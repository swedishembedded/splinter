// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that run reinforcement learning on
// a language model served by a continuously batching engine, for its clients.
// If your team needs expertise in throughput-bound rollout collection for
// policy-gradient training you can procure our services by sending an email
// to info@swedishembedded.com.

//! The served path: rollouts and the dev exam are asked of a model a serving
//! endpoint holds, many requests at once, so the engine's continuous batching
//! decodes them together. Which endpoint serves which policy is the host's
//! business ([`PolicyModels`]); nothing here knows how a server is started.
//!
//! The request is exactly the family's conversation, sent through sven's
//! provider with no system prompt of Splinter's own added.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::Duration;

use futures::StreamExt;
use serde_json::{json, Value};
use splinter_agent::chat::{reply_bare, ChatError, ChatOptions, Reply, Turn};
use splinter_agent::solve::Model;
use splinter_eval::campaign::gate::{DevResult, Outcomes};

use super::brain::{
    attempt_dir, trained_from, tune, write_records, ModelSettings, TaskFamily, TrainerSettings,
};
use super::types::{
    CampaignError, CollectRequest, PolicyRef, Rollout, RolloutCollector, RolloutGroup,
    TrainRequest, Trained, Trainer, Validator,
};
use splinter_core::chat::WireMessage;
use splinter_model::policy_gradient::PolicyGradientTune;
use splinter_model::served_train::train_policy_gradient_served;
use splinter_model::BrainEndpoints;

/// How long one reply may take, queueing behind the batch included.
const REPLY_DEADLINE: Duration = Duration::from_secs(600);

/// Whoever serves policies: a model for a policy, sampling at temperature one
/// (training rollouts) or greedily (the dev exam).
pub trait PolicyModels: Send + Sync {
    /// The model that answers as `policy` does.
    ///
    /// # Errors
    /// The policy cannot be served.
    fn model(&self, policy: &PolicyRef, greedy: bool) -> Result<Model, CampaignError>;
}

fn turns(family: &dyn TaskFamily, task: &str) -> Result<Vec<Turn>, CampaignError> {
    family
        .prompt(task)?
        .into_iter()
        .map(|(role, text)| match role.as_str() {
            "user" => Ok(Turn::user(text)),
            "assistant" => Ok(Turn::assistant(text)),
            other => Err(CampaignError::Config(format!(
                "the served path sends user and assistant turns, not {other:?} (task {task})"
            ))),
        })
        .collect()
}

fn wire(turns: &[Turn]) -> Vec<WireMessage> {
    turns
        .iter()
        .map(|turn| WireMessage {
            role: if turn.by_assistant {
                "assistant"
            } else {
                "user"
            }
            .to_string(),
            content: turn.text.clone(),
            images: Vec::new(),
            tool_calls: Vec::new(),
            tool_call_id: None,
            reasoning_content: None,
            train: false,
        })
        .collect()
}

async fn ask(
    model: &Model,
    turns: &[Turn],
    max_tokens: u32,
    trace: bool,
) -> Result<Reply, ChatError> {
    let mut options = ChatOptions::new(REPLY_DEADLINE);
    options.max_output_tokens = Some(max_tokens);
    options.trace = trace;
    reply_bare(model, turns, &options).await
}

/// The served collector: a group's attempts are concurrent requests at
/// temperature one.
pub struct ServedCollector<'f> {
    family: &'f dyn TaskFamily,
    models: &'f dyn PolicyModels,
    settings: ModelSettings,
    policy: PolicyRef,
    runtime: tokio::runtime::Runtime,
}

impl<'f> ServedCollector<'f> {
    /// A collector over `family`, asking the models of `models`.
    ///
    /// # Errors
    /// The async runtime could not start.
    pub fn new(
        family: &'f dyn TaskFamily,
        models: &'f dyn PolicyModels,
        settings: ModelSettings,
    ) -> Result<Self, CampaignError> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .map_err(|e| CampaignError::Collector(format!("starting the runtime: {e}")))?;
        Ok(Self {
            family,
            models,
            settings,
            policy: PolicyRef::default(),
            runtime,
        })
    }
}

impl RolloutCollector for ServedCollector<'_> {
    fn describe(&self) -> Value {
        json!({"path": "served", "model": self.settings, "family": self.family.describe()})
    }

    fn tasks(&self) -> Vec<String> {
        self.family.pool()
    }

    fn set_policy(&mut self, policy: &PolicyRef) -> Result<(), CampaignError> {
        // The host serves it when the first request names it.
        self.policy = policy.clone();
        Ok(())
    }

    fn collect(&mut self, request: &CollectRequest) -> Result<Vec<RolloutGroup>, CampaignError> {
        let model = self.models.model(&request.policy, false)?;
        let max_tokens = self.settings.max_tokens;
        let mut jobs = Vec::new();
        for (t, task) in request.tasks.iter().enumerate() {
            let conversation = turns(self.family, task)?;
            for _ in 0..request.attempts {
                jobs.push((t, conversation.clone()));
            }
        }
        let model = &model;
        let replies = self.runtime.block_on(async {
            futures::stream::iter(jobs.into_iter().map(|(t, conversation)| async move {
                let reply = ask(model, &conversation, max_tokens, false)
                    .await
                    .map_err(|e| CampaignError::Collector(e.to_string()))?;
                Ok::<_, CampaignError>((t, conversation, reply.text))
            }))
            .buffered(self.settings.concurrency.max(1))
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
            let (t, conversation, completion) = reply?;
            let graded = self.family.grade(&request.tasks[t], &completion)?;
            groups[t].rollouts.push(Rollout {
                reward: graded.reward,
                passed: graded.passed,
                prompt: wire(&conversation),
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

/// The served dev exam: every dev and retention task answered once, greedily.
/// A policy that was already examined keeps its result, so a champion is not
/// asked twice for the exam its candidate was just compared on.
pub struct ServedValidator<'f> {
    family: &'f dyn TaskFamily,
    models: &'f dyn PolicyModels,
    settings: ModelSettings,
    runtime: tokio::runtime::Runtime,
    kept: Mutex<BTreeMap<String, DevResult>>,
    trace: bool,
}

impl<'f> ServedValidator<'f> {
    /// A validator over `family`, asking the models of `models`.
    ///
    /// # Errors
    /// The async runtime could not start.
    pub fn new(
        family: &'f dyn TaskFamily,
        models: &'f dyn PolicyModels,
        settings: ModelSettings,
    ) -> Result<Self, CampaignError> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .map_err(|e| CampaignError::Validator(format!("starting the runtime: {e}")))?;
        Ok(Self {
            family,
            models,
            settings,
            runtime,
            kept: Mutex::new(BTreeMap::new()),
            trace: false,
        })
    }

    /// The same validator, asking every reply's policy trace (the prompt ids
    /// the model read, the generated ids and their log-probabilities), so an
    /// answer records exactly what was put to the model and who served it.
    #[must_use]
    pub fn traced(mut self) -> Self {
        self.trace = true;
        self
    }

    /// Every task of `tasks` answered once by `policy`, greedily, as many
    /// requests at once as the settings allow, each with the completion and
    /// whether the family's verifier passed it.
    ///
    /// # Errors
    /// The policy cannot be served, a request fails, or the verifier does.
    pub fn answer(
        &self,
        policy: &PolicyRef,
        tasks: &[String],
    ) -> Result<Vec<Answered>, CampaignError> {
        let model = self.models.model(policy, true)?;
        self.answer_with(&model, tasks)
    }

    /// Every task of `tasks` answered once by `model`, as many requests at
    /// once as the settings allow; a task listed several times is answered
    /// that many times, independently. How the model decodes (greedily or
    /// sampled, at what temperature) is the model's own.
    ///
    /// # Errors
    /// A request fails (the first failure is named), or the verifier does.
    pub fn answer_with(
        &self,
        model: &Model,
        tasks: &[String],
    ) -> Result<Vec<Answered>, CampaignError> {
        self.attempt_with(model, tasks)?
            .into_iter()
            .map(|attempt| {
                attempt.outcome.map_err(|failure| {
                    CampaignError::Validator(format!("task {}: {failure}", attempt.task))
                })
            })
            .collect()
    }

    /// Every task of `tasks` put to `model` as [`Self::answer_with`] does,
    /// with each call's outcome kept apart: a request that produced no reply
    /// is its own [`CallFailure`], never a wrong answer, and does not stop the
    /// other tasks.
    ///
    /// # Errors
    /// A task is not in the family, or the verifier failed.
    pub fn attempt_with(
        &self,
        model: &Model,
        tasks: &[String],
    ) -> Result<Vec<Attempt>, CampaignError> {
        let (family, max_tokens, trace) = (self.family, self.settings.max_tokens, self.trace);
        let attempts = self.runtime.block_on(async {
            futures::stream::iter(tasks.iter().map(|task| async move {
                let conversation = turns(family, task)?;
                let outcome = match ask(model, &conversation, max_tokens, trace).await {
                    Ok(reply) => {
                        let passed = family.grade(task, &reply.text)?.passed;
                        Ok(Answered {
                            task: task.clone(),
                            reply,
                            passed,
                        })
                    }
                    Err(e) => Err(CallFailure::of(&e)),
                };
                Ok::<_, CampaignError>(Attempt {
                    task: task.clone(),
                    outcome,
                })
            }))
            .buffered(self.settings.concurrency.max(1))
            .collect::<Vec<_>>()
            .await
        });
        attempts.into_iter().collect()
    }

    fn pass_each(&self, model: &Model, tasks: &[String]) -> Result<Outcomes, CampaignError> {
        Ok(self
            .answer_with(model, tasks)?
            .into_iter()
            .map(|a| (a.task, a.passed))
            .collect())
    }
}

/// One task answered by a policy: what it wrote and whether the verifier
/// passed it.
#[derive(Clone, Debug, PartialEq)]
pub struct Answered {
    /// The task id.
    pub task: String,
    /// The reply: the completion as the policy wrote it, whether the output
    /// limit cut it, and the trace and served identity the server reported.
    pub reply: Reply,
    /// Whether the family's verifier passed it.
    pub passed: bool,
}

/// One task put to a policy: its answer, or why the call produced none.
#[derive(Clone, Debug, PartialEq)]
pub struct Attempt {
    /// The task id.
    pub task: String,
    /// The answer, or the failed call.
    pub outcome: Result<Answered, CallFailure>,
}

/// Why a call to the model produced no reply to grade: an infrastructure
/// outcome, never a wrong answer.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum CallFailure {
    /// The reply was not whole within the deadline.
    #[error("timed out: {0}")]
    TimedOut(String),
    /// The server could not be reached, refused the request or failed it.
    #[error("the server failed: {0}")]
    Server(String),
    /// The call was cancelled.
    #[error("cancelled")]
    Cancelled,
}

impl CallFailure {
    fn of(error: &ChatError) -> Self {
        match error {
            ChatError::TimedOut(_) => Self::TimedOut(error.to_string()),
            ChatError::Cancelled => Self::Cancelled,
            ChatError::Model(_) | ChatError::Empty => Self::Server(error.to_string()),
        }
    }
}

/// What identifies a policy: its adapter's digest, else its version.
fn key(policy: &PolicyRef) -> String {
    policy.digest.as_ref().map_or_else(
        || format!("version-{}", policy.version),
        ToString::to_string,
    )
}

impl Validator for ServedValidator<'_> {
    fn describe(&self) -> Value {
        json!({"path": "served greedy", "model": self.settings, "family": self.family.describe()})
    }

    fn units(&self) -> usize {
        self.family.dev().len()
    }

    fn retention_units(&self) -> usize {
        self.family.retention().len()
    }

    fn dev_tasks(&self) -> Vec<String> {
        self.family.dev()
    }

    /// Asks only what `policy` was not asked before: a policy's answers are
    /// kept, so a champion graded in one round is not graded again.
    fn validate_on(
        &mut self,
        policy: &PolicyRef,
        dev: &[String],
    ) -> Result<DevResult, CampaignError> {
        let held = key(policy);
        let mut known = self
            .kept
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&held)
            .cloned()
            .unwrap_or_default();
        let new_dev: Vec<String> = dev
            .iter()
            .filter(|t| !known.dev.contains_key(*t))
            .cloned()
            .collect();
        let new_retention: Vec<String> = self
            .family
            .retention()
            .into_iter()
            .filter(|t| !known.retention.contains_key(t))
            .collect();
        if !new_dev.is_empty() || !new_retention.is_empty() {
            let model = self.models.model(policy, true)?;
            known.dev.extend(self.pass_each(&model, &new_dev)?);
            known
                .retention
                .extend(self.pass_each(&model, &new_retention)?);
            self.kept
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .insert(held, known.clone());
        }
        let wanted: std::collections::BTreeSet<&String> = dev.iter().collect();
        known.dev.retain(|task, _| wanted.contains(task));
        Ok(known)
    }
}

/// The call into brain as a training job of the service that also serves the
/// policy: a clipped policy gradient on the batch's records, continuing the
/// champion's adapter, with datasets and adapters exchanged through the
/// endpoints' artifact store.
pub struct ServedTrainer {
    settings: TrainerSettings,
    endpoints: BrainEndpoints,
}

impl ServedTrainer {
    /// A trainer with `settings` on the service at `endpoints`.
    ///
    /// # Errors
    /// The endpoints name no artifact store, which a job needs.
    pub fn new(
        settings: TrainerSettings,
        endpoints: BrainEndpoints,
    ) -> Result<Self, CampaignError> {
        endpoints
            .artifact_store()
            .map_err(|e| CampaignError::Config(e.to_string()))?;
        Ok(Self {
            settings,
            endpoints,
        })
    }
}

impl Trainer for ServedTrainer {
    fn describe(&self) -> Value {
        json!({"path": "brain training job", "settings": self.settings})
    }

    fn train(&mut self, request: &TrainRequest) -> Result<Trained, CampaignError> {
        let records = write_records(request)?;
        let attempt = attempt_dir(request)?;
        let tune: PolicyGradientTune<'_> = tune(&self.settings, request, &records, &attempt);
        let trained = train_policy_gradient_served(&self.endpoints, &tune)
            .map_err(|e| CampaignError::Trainer(e.to_string()))?;
        trained_from(request, &trained)
    }
}
