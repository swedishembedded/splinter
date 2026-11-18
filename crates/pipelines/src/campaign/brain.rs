// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that run reinforcement learning on
// a local language model against verifiers that do not trust it, for its
// clients. If your team needs expertise in in-process rollout collection and
// policy-gradient training you can procure our services by sending an email
// to info@swedishembedded.com.

//! What a campaign over brain is made of, whichever way it reaches brain: the
//! task family, the settings of the model sampled and of the trainer, and the
//! files a training call writes and reads. The collector, validator and
//! trainer that use them over brain's API are in [`super::served`].
//!
//! A [`TaskFamily`] is what the path needs to know about the tasks: how a
//! task is put to the policy and how a completion is graded. It is the
//! seam where a family's own verifier meets the campaign; nothing here knows
//! what a task is about.

use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::Value;
use splinter_core::digest::Digest;
use splinter_data::{write_dataset, AdvantageKind, Objective, Projection};
use splinter_model::policy_gradient::{PolicyGradientOptions, PolicyGradientTune};
use splinter_model::precision::BasePrecision;

use super::types::{CampaignError, TrainReport, TrainRequest, Trained};

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

/// The directory a training call writes its candidate into, named after the
/// arm (`<out_dir>/<arm>`). Its name is the tag of the adapter's card id
/// (`<base>:splinter:candidate:<arm>`): a server offers every adapter of a
/// directory under its card id and refuses a second file with an id it already
/// serves, so each arm's candidate must carry its own tag. A fixed name made
/// every candidate after the first unservable beside it.
pub(super) fn attempt_dir(request: &TrainRequest) -> Result<PathBuf, CampaignError> {
    let arm = request
        .out_dir
        .file_name()
        .and_then(|n| n.to_str())
        .filter(|n| !n.is_empty() && !n.contains([':', '/']))
        .ok_or_else(|| {
            CampaignError::Trainer(format!(
                "{}: an arm's directory needs a name to tag its adapter with",
                request.out_dir.display()
            ))
        })?;
    Ok(request.out_dir.join(arm))
}

/// Writes the request's records as the policy-gradient dataset the trainer
/// reads, in the arm's own directory.
pub(super) fn write_records(request: &TrainRequest) -> Result<PathBuf, CampaignError> {
    let fail = |why: String| CampaignError::Trainer(why);
    std::fs::create_dir_all(&request.out_dir)
        .map_err(|e| fail(format!("creating {}: {e}", request.out_dir.display())))?;
    let mut projection = Projection::new("campaign", Objective::PolicyGradient, None, None);
    projection.records.clone_from(&request.records);
    projection.advantage = request.advantage;
    let records_path = request.out_dir.join("records.jsonl");
    write_dataset(&records_path, &projection, &splinter_data::Unchecked)
        .map_err(|e| fail(format!("writing the training records: {e}")))?;
    Ok(records_path)
}

/// The policy-gradient run `settings` ask for on `records`, continuing the
/// champion's adapter, with its files under `attempt`.
pub(super) fn tune<'a>(
    s: &'a TrainerSettings,
    request: &'a TrainRequest,
    records: &'a Path,
    attempt: &'a Path,
) -> PolicyGradientTune<'a> {
    PolicyGradientTune {
        model_dir: &s.base,
        train: records,
        attempt_dir: attempt,
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
        pack_answers: false,
        options: PolicyGradientOptions {
            advantage_norm: request.advantage.map(AdvantageKind::norm),
            advantage_floor: request.advantage.map(AdvantageKind::floor),
            ..s.options
        },
    }
}

/// What the controller keeps of a finished run.
pub(super) fn trained_from(
    request: &TrainRequest,
    trained: &splinter_model::policy_gradient::TrainedPolicyGradient,
) -> Result<Trained, CampaignError> {
    let fail = |why: String| CampaignError::Trainer(why);
    let stats = &trained.statistics;
    Ok(Trained {
        digest: Digest::parse(&trained.adapter_digest)
            .map_err(|e| fail(format!("the adapter's digest: {e}")))?,
        bytes: directory_bytes(&request.out_dir),
        adapter: trained.adapter.clone(),
        dir: request.out_dir.clone(),
        report: TrainReport {
            steps: u32::try_from(stats.steps.len()).unwrap_or(u32::MAX),
            kl: stats.mean_kl.map(f64::from),
            clip_fraction: stats.mean_clip_fraction.map(f64::from),
            skipped_nonfinite: stats.steps_skipped_nonfinite,
            skipped_guard: stats.steps_skipped_guard,
            entropy: stats.mean_entropy.map(f64::from),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(out_dir: &str) -> TrainRequest {
        TrainRequest {
            records: Vec::new(),
            continue_from: None,
            out_dir: PathBuf::from(out_dir),
            seed: u64::from(std::process::id()),
            advantage: None,
        }
    }

    /// Two arms' candidates must not share a card id: the served directory
    /// refuses a second adapter with an id it already serves, so the arm names
    /// the attempt directory, whose name is the card id's tag.
    #[test]
    fn each_arm_trains_into_a_directory_named_after_it() {
        let first = attempt_dir(&request("/run/adapters/arm-0001-policy-gradient")).unwrap();
        let second = attempt_dir(&request("/run/adapters/arm-0002-policy-gradient")).unwrap();
        assert_eq!(
            first,
            PathBuf::from("/run/adapters/arm-0001-policy-gradient/arm-0001-policy-gradient")
        );
        assert_ne!(first.file_name(), second.file_name());
        assert!(attempt_dir(&request("/")).is_err());
    }
}
