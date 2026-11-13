// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that run reinforcement learning on
// a local language model against verifiers that do not trust it, for its
// clients. If your team needs expertise in in-process rollout collection and
// policy-gradient training you can procure our services by sending an email
// to info@swedishembedded.com.

//! The in-process brain path: a loaded model sampled at temperature 1
//! against a verifier, brain's policy-gradient trainer, and a greedy dev
//! exam.
//!
//! A [`TaskFamily`] is what the path needs to know about the tasks: how a
//! task is put to the policy and how a completion is graded. It is the
//! seam where a family's own verifier meets the campaign; nothing here knows
//! what a task is about.

use std::path::{Path, PathBuf};

use futures::StreamExt;
use serde::Serialize;
use serde_json::{json, Value};
use splinter_core::chat::WireMessage;
use splinter_core::digest::Digest;
use splinter_data::{write_dataset, Objective, Projection};
use splinter_eval::campaign::gate::{DevResult, Outcomes};
use splinter_model::answer::{Answerer, Decoding};
use splinter_model::policy_gradient::{
    train_policy_gradient, PolicyGradientOptions, PolicyGradientTune,
};
use splinter_model::precision::BasePrecision;
use splinter_model::BrainDatasetCheck;

use super::types::{
    CampaignError, CollectRequest, PolicyRef, Rollout, RolloutCollector, RolloutGroup, TrainReport,
    TrainRequest, Trained, Trainer, Validator,
};

/// How a completion was graded.
#[derive(Clone, Debug, PartialEq)]
pub struct Graded {
    /// The reward, in `[0, 1]`.
    pub reward: f64,
    /// Whether it passed.
    pub passed: bool,
    /// What the verifier shows an attempt it failed.
    pub feedback: Option<String>,
    /// What only the verifier knows, for distillation.
    pub privileged: Option<String>,
}

/// The tasks of a campaign and their verifier.
pub trait TaskFamily: Send + Sync {
    /// The family's settings, for the configuration digest.
    fn describe(&self) -> Value;
    /// The ids of the tasks campaigns train on.
    fn pool(&self) -> Vec<String>;
    /// The ids of the dev tasks, never trained on.
    fn dev(&self) -> Vec<String>;
    /// The ids of tasks of earlier work to keep; may be empty.
    fn retention(&self) -> Vec<String>;
    /// The conversation put to the policy for `task`, as (role, text).
    ///
    /// # Errors
    /// The task is not in the family.
    fn prompt(&self, task: &str) -> Result<Vec<(String, String)>, CampaignError>;
    /// Grades `completion` of `task`.
    ///
    /// # Errors
    /// The task is not in the family, or the verifier failed (not the same as
    /// a wrong answer).
    fn grade(&self, task: &str, completion: &str) -> Result<Graded, CampaignError>;
}

/// How the model is loaded and sampled.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ModelSettings {
    /// The base checkpoint directory.
    pub base: PathBuf,
    /// The context the model is loaded with.
    pub context_tokens: Option<u32>,
    /// Tokens a completion may take.
    pub max_tokens: u32,
    /// The sampling temperature of training rollouts: 1, so the log-probability
    /// the sampler gave a token is the policy's own.
    pub temperature: f32,
    /// Requests in flight at once.
    pub concurrency: usize,
}

impl ModelSettings {
    /// Settings for `base`, sampling at temperature 1.
    #[must_use]
    pub fn new(base: &Path) -> Self {
        Self {
            base: base.to_path_buf(),
            context_tokens: Some(2048),
            max_tokens: 96,
            temperature: 1.0,
            concurrency: 16,
        }
    }
}

fn first_turns(turns: &[(String, String)]) -> Vec<(&str, &str)> {
    turns
        .iter()
        .map(|(r, t)| (r.as_str(), t.as_str()))
        .collect()
}

fn prompt_messages(turns: &[(String, String)]) -> Vec<WireMessage> {
    turns
        .iter()
        .map(|(role, text)| WireMessage {
            role: role.clone(),
            content: text.clone(),
            images: Vec::new(),
            tool_calls: Vec::new(),
            tool_call_id: None,
            reasoning_content: None,
            train: false,
        })
        .collect()
}

fn answerer(
    settings: &ModelSettings,
    policy: &PolicyRef,
    decoding: Decoding,
) -> Result<Answerer, CampaignError> {
    Answerer::load_with(
        &settings.base,
        policy.adapter.as_deref(),
        settings.context_tokens,
        "campaign",
        decoding,
    )
    .map_err(|e| CampaignError::Collector(format!("loading {}: {e}", settings.base.display())))
}

/// The in-process collector: the policy sampled at temperature 1, one draw per
/// attempt under a seed derived from the request's.
pub struct ChatCollector<'f> {
    family: &'f dyn TaskFamily,
    settings: ModelSettings,
    policy: PolicyRef,
    model: Option<Answerer>,
    runtime: tokio::runtime::Runtime,
}

impl<'f> ChatCollector<'f> {
    /// A collector over `family`.
    ///
    /// # Errors
    /// The async runtime could not start.
    pub fn new(family: &'f dyn TaskFamily, settings: ModelSettings) -> Result<Self, CampaignError> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| CampaignError::Collector(format!("starting the runtime: {e}")))?;
        Ok(Self {
            family,
            settings,
            policy: PolicyRef::default(),
            model: None,
            runtime,
        })
    }
}

impl RolloutCollector for ChatCollector<'_> {
    fn describe(&self) -> Value {
        json!({"path": "in-process", "model": self.settings, "family": self.family.describe()})
    }

    fn tasks(&self) -> Vec<String> {
        self.family.pool()
    }

    fn set_policy(&mut self, policy: &PolicyRef) -> Result<(), CampaignError> {
        // Dropped first: two copies of a large model do not share a card.
        self.model = None;
        self.model = Some(answerer(&self.settings, policy, Decoding::Greedy)?);
        self.policy = policy.clone();
        Ok(())
    }

    fn collect(&mut self, request: &CollectRequest) -> Result<Vec<RolloutGroup>, CampaignError> {
        if self.model.is_none() || self.policy != request.policy {
            self.set_policy(&request.policy)?;
        }
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| CampaignError::Collector("no model is loaded".into()))?;
        let family = self.family;
        let settings = &self.settings;
        let mut jobs = Vec::new();
        for (t, task) in request.tasks.iter().enumerate() {
            let turns = family.prompt(task)?;
            for attempt in 0..request.attempts {
                let seed = request
                    .seed
                    .wrapping_add((t as u64) << 20)
                    .wrapping_add(attempt as u64);
                jobs.push((t, turns.clone(), seed));
            }
        }
        let replies = self.runtime.block_on(async {
            futures::stream::iter(jobs.into_iter().map(|(t, turns, seed)| async move {
                let decoding = Decoding::Sampled {
                    temperature: settings.temperature,
                    seed: Some(seed),
                };
                let reply = model
                    .ask_turns(decoding, &first_turns(&turns), settings.max_tokens)
                    .await
                    .map_err(|e| CampaignError::Collector(e.to_string()))?;
                Ok::<_, CampaignError>((t, turns, reply))
            }))
            .buffered(settings.concurrency.max(1))
            .collect::<Vec<_>>()
            .await
        });
        let mut groups: Vec<RolloutGroup> = request
            .tasks
            .iter()
            .map(|task| RolloutGroup {
                task: task.clone(),
                rollouts: Vec::new(),
            })
            .collect();
        for reply in replies {
            let (t, turns, reply) = reply?;
            let completion = reply.text.trim().to_string();
            let graded = family.grade(&request.tasks[t], &completion)?;
            groups[t].rollouts.push(Rollout {
                reward: graded.reward,
                passed: graded.passed,
                prompt: prompt_messages(&turns),
                completion,
                feedback: graded.feedback,
                privileged: graded.privileged,
                policy_version: request.policy.version,
                trace: None,
            });
        }
        Ok(groups)
    }
}

/// The dev exam: the policy answers every dev and retention task once,
/// greedily, so a task has one answer for given weights and the champion and
/// the candidate differ only by the weights.
pub struct ChatValidator<'f> {
    family: &'f dyn TaskFamily,
    settings: ModelSettings,
    runtime: tokio::runtime::Runtime,
}

impl<'f> ChatValidator<'f> {
    /// A validator over `family`.
    ///
    /// # Errors
    /// The async runtime could not start.
    pub fn new(family: &'f dyn TaskFamily, settings: ModelSettings) -> Result<Self, CampaignError> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| CampaignError::Validator(format!("starting the runtime: {e}")))?;
        Ok(Self {
            family,
            settings,
            runtime,
        })
    }

    fn pass_each(&self, model: &Answerer, tasks: &[String]) -> Result<Outcomes, CampaignError> {
        let family = self.family;
        let max_tokens = self.settings.max_tokens;
        let answers = self.runtime.block_on(async {
            futures::stream::iter(tasks.iter().map(|task| async move {
                let turns = family.prompt(task)?;
                let reply = model
                    .ask_turns(Decoding::Greedy, &first_turns(&turns), max_tokens)
                    .await
                    .map_err(|e| CampaignError::Validator(e.to_string()))?;
                let graded = family.grade(task, reply.text.trim())?;
                Ok::<_, CampaignError>((task.clone(), graded.passed))
            }))
            .buffered(self.settings.concurrency.max(1))
            .collect::<Vec<_>>()
            .await
        });
        answers.into_iter().collect()
    }
}

impl Validator for ChatValidator<'_> {
    fn describe(&self) -> Value {
        json!({"path": "in-process greedy", "model": self.settings, "family": self.family.describe()})
    }

    fn units(&self) -> usize {
        self.family.dev().len()
    }

    fn retention_units(&self) -> usize {
        self.family.retention().len()
    }

    fn validate(&mut self, policy: &PolicyRef) -> Result<DevResult, CampaignError> {
        let model = answerer(&self.settings, policy, Decoding::Greedy)
            .map_err(|e| CampaignError::Validator(e.to_string()))?;
        Ok(DevResult {
            dev: self.pass_each(&model, &self.family.dev())?,
            retention: self.pass_each(&model, &self.family.retention())?,
        })
    }
}

/// How brain trains a candidate.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TrainerSettings {
    /// The base checkpoint directory.
    pub base: PathBuf,
    /// Optimiser steps per arm.
    pub steps: u32,
    /// Adapter rank.
    pub rank: u32,
    /// Adapter alpha.
    pub alpha: f32,
    /// Learning rate.
    pub learning_rate: f32,
    /// Records averaged into one optimiser step.
    pub records_per_step: u32,
    /// The longest training row; longer records are refused.
    pub max_block: Option<u32>,
    /// The objective's settings; unset ones are brain's defaults, recorded.
    pub options: PolicyGradientOptions,
    /// The precision of the frozen base.
    pub base_precision: BasePrecision,
}

/// The call into brain: a clipped policy-gradient run, continuing the
/// champion's adapter.
pub struct BrainTrainer {
    settings: TrainerSettings,
}

impl BrainTrainer {
    /// A trainer with `settings`.
    #[must_use]
    pub fn new(settings: TrainerSettings) -> Self {
        Self { settings }
    }
}

fn directory_bytes(dir: &Path) -> u64 {
    std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| match entry.metadata() {
            Ok(meta) if meta.is_dir() => directory_bytes(&entry.path()),
            Ok(meta) => meta.len(),
            Err(_) => 0,
        })
        .sum()
}

impl Trainer for BrainTrainer {
    fn describe(&self) -> Value {
        json!({"path": "brain policy-gradient", "settings": self.settings})
    }

    fn train(&mut self, request: &TrainRequest) -> Result<Trained, CampaignError> {
        let fail = |why: String| CampaignError::Trainer(why);
        std::fs::create_dir_all(&request.out_dir)
            .map_err(|e| fail(format!("creating {}: {e}", request.out_dir.display())))?;
        let mut projection = Projection::new("campaign", Objective::PolicyGradient, None, None);
        projection.records.clone_from(&request.records);
        let records_path = request.out_dir.join("records.jsonl");
        write_dataset(&records_path, &projection, &BrainDatasetCheck)
            .map_err(|e| fail(format!("writing the training records: {e}")))?;
        let attempt = request.out_dir.join("run");
        let s = &self.settings;
        let trained = train_policy_gradient(&PolicyGradientTune {
            model_dir: &s.base,
            train: &records_path,
            attempt_dir: &attempt,
            steps: s.steps,
            rank: s.rank,
            alpha: s.alpha,
            weight_decay: 0.0,
            grad_accum: s.records_per_step,
            continue_from: request.continue_from.as_deref(),
            cancel: None,
            base: s.base_precision,
            learning_rate: Some(s.learning_rate),
            seed: request.seed,
            max_block: s.max_block,
            thinking: false,
            options: s.options,
        })
        .map_err(|e| fail(e.to_string()))?;
        let stats = &trained.statistics;
        Ok(Trained {
            digest: Digest::parse(&trained.adapter_digest)
                .map_err(|e| fail(format!("the adapter's digest: {e}")))?,
            bytes: directory_bytes(&request.out_dir),
            adapter: trained.adapter,
            dir: request.out_dir.clone(),
            report: TrainReport {
                steps: u32::try_from(stats.steps.len()).unwrap_or(u32::MAX),
                kl: stats.mean_kl.map(f64::from),
                clip_fraction: stats.mean_clip_fraction.map(f64::from),
                skipped_nonfinite: stats.steps_skipped_nonfinite,
            },
        })
    }
}
