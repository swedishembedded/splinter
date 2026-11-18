// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that run model learning as one
// continuous, resumable campaign, for its clients. If your team needs
// expertise in rollout collection and trainer interfaces for language-model
// agents you can procure our services by sending an email to
// info@swedishembedded.com.

//! The seams of in-memory rollout collection over a served policy: what a
//! collector, a trainer and a validator are, and the plain data that crosses
//! them (rollout groups with the sampler's own token ids and
//! log-probabilities). Samples and the episode executors are built on them;
//! the campaign controller's own seams are [`super::work`]. The campaign's
//! error, [`CampaignError`], is shared by both.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use splinter_core::chat::WireMessage;
use splinter_core::digest::Digest;
use splinter_data::Record;
use splinter_eval::campaign::gate::DevResult;
use splinter_eval::campaign::sampling::{Attempt, Group};

/// Why a campaign operation failed.
#[derive(Debug, thiserror::Error)]
pub enum CampaignError {
    /// The configuration cannot run a campaign.
    #[error("invalid campaign configuration: {0}")]
    Config(String),
    /// The collector failed.
    #[error("rollout collection failed: {0}")]
    Collector(String),
    /// The trainer failed.
    #[error("training failed: {0}")]
    Trainer(String),
    /// The validator failed.
    #[error("dev validation failed: {0}")]
    Validator(String),
    /// The campaign was cancelled; a training job in progress was cancelled
    /// on the server.
    #[error("cancelled")]
    Cancelled,
    /// A stage that runs Splinter's own pipelines failed.
    #[error(transparent)]
    Orchestrator(Box<splinter_orchestrator::error::OrchestratorError>),
    /// The campaign's stage ledger failed, or another process holds it.
    #[error(transparent)]
    Ledger(#[from] splinter_store::stage_ledger::LedgerError),
}

/// The policy rollouts are drawn from: the base model and, after the first
/// accepted arm, an adapter on it.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicyRef {
    /// How many arms have been accepted: the version of the champion. A
    /// rollout carries the version it was drawn from.
    pub version: u64,
    /// The adapter on the base; `None` for the base itself.
    pub adapter: Option<PathBuf>,
    /// The adapter's content digest, when it has one.
    pub digest: Option<Digest>,
}

/// One sampled call of a rollout, as the serving endpoint returned it:
/// nothing here is re-tokenized, and a call the endpoint did not return
/// everything for is not provable.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TraceSequence {
    /// The prompt token ids the model read.
    pub prompt_token_ids: Vec<u32>,
    /// The token ids it generated.
    pub generation_token_ids: Vec<u32>,
    /// The log-probability of each generated token under the policy that
    /// sampled it (temperature 1, no truncation).
    pub generation_log_probs: Vec<f32>,
    /// Whether the generation was cut off by the token budget.
    pub truncated: bool,
}

/// The lossless policy trace of a rollout.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RolloutTrace {
    /// The sampled calls, in order.
    pub sequences: Vec<TraceSequence>,
    /// Whether every call is provable. A rollout that is not is dropped, never
    /// repaired by re-tokenizing.
    pub provable: bool,
}

/// One graded attempt at a task.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rollout {
    /// The reward its verifier gave it, in `[0, 1]`.
    pub reward: f64,
    /// Whether the verifier called it a pass.
    pub passed: bool,
    /// The conversation before the policy's completion, as the policy read
    /// it; none of it is supervised.
    pub prompt: Vec<WireMessage>,
    /// What the policy wrote.
    pub completion: String,
    /// What the verifier showed the attempt after it failed, when it shows
    /// anything: the text a repair turn is built from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feedback: Option<String>,
    /// What only the verifier knows about the task (the reference answer, the
    /// hidden check): shown to a teacher, never to the policy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub privileged: Option<String>,
    /// The policy version it was drawn from.
    pub policy_version: u64,
    /// The sampled calls, when the collector records them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace: Option<RolloutTrace>,
}

/// The attempts one policy made at one task.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RolloutGroup {
    /// The task.
    pub task: String,
    /// Its graded attempts; an attempt nothing graded is not a rollout.
    pub rollouts: Vec<Rollout>,
}

impl RolloutGroup {
    /// The group's rewards as the sampling decisions read them, tagged
    /// `tag`.
    #[must_use]
    pub fn summary(&self, tag: u64) -> Group {
        Group {
            task: self.task.clone(),
            attempts: self
                .rollouts
                .iter()
                .map(|r| Attempt {
                    reward: r.reward,
                    passed: r.passed,
                })
                .collect(),
            tag,
        }
    }
}

/// What the collector is asked for.
#[derive(Clone, Debug, PartialEq)]
pub struct CollectRequest {
    /// The tasks to attempt.
    pub tasks: Vec<String>,
    /// Attempts per task.
    pub attempts: usize,
    /// The policy to attempt them with.
    pub policy: PolicyRef,
    /// The seed the draws derive from; recorded by the campaign.
    pub seed: u64,
}

/// Where rollouts come from.
pub trait RolloutCollector {
    /// The collector's settings, for the configuration digest.
    fn describe(&self) -> Value;
    /// The tasks the collector can attempt: the campaign's pool.
    fn tasks(&self) -> Vec<String>;
    /// Makes `policy` the policy later requests are drawn from: loads the
    /// adapter, or hot-swaps it on a server.
    ///
    /// # Errors
    /// The policy cannot be served.
    fn set_policy(&mut self, policy: &PolicyRef) -> Result<(), CampaignError>;
    /// A group per task of the request, under the policy it names. A task the
    /// collector cannot attempt is an error, not an empty group.
    ///
    /// # Errors
    /// The attempts could not be made or graded.
    fn collect(&mut self, request: &CollectRequest) -> Result<Vec<RolloutGroup>, CampaignError>;
}

/// What the trainer is asked for.
#[derive(Debug)]
pub struct TrainRequest {
    /// The records to train on: rollouts with their rewards and groups,
    /// imitation records and replay.
    pub records: Vec<Record>,
    /// The adapter to continue; `None` trains a fresh one on the base.
    pub continue_from: Option<PathBuf>,
    /// Where the candidate adapter goes: a directory the trainer creates.
    pub out_dir: PathBuf,
    /// The seed of the optimiser's batches; recorded by the campaign.
    pub seed: u64,
    /// How the trainer turns the rollout records' rewards into advantages;
    /// `None` when no record carries a reward.
    pub advantage: Option<splinter_data::AdvantageKind>,
}

/// What a training call measured, each absent when not measured.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TrainReport {
    /// Optimiser steps taken.
    pub steps: u32,
    /// Mean KL to the reference over the steps.
    pub kl: Option<f64>,
    /// Mean clip fraction over the steps.
    pub clip_fraction: Option<f64>,
    /// Steps skipped because a loss or gradient was not finite.
    pub skipped_nonfinite: usize,
    /// Steps skipped because the step's KL or gradient norm was over its
    /// bound.
    #[serde(default)]
    pub skipped_guard: usize,
    /// Mean entropy estimate of the sampled tokens over the steps.
    #[serde(default)]
    pub entropy: Option<f64>,
}

/// A trained candidate.
#[derive(Clone, Debug, PartialEq)]
pub struct Trained {
    /// The candidate adapter file a policy loads.
    pub adapter: PathBuf,
    /// The directory the arm's training wrote, which collection removes whole
    /// when the checkpoint is superseded.
    pub dir: PathBuf,
    /// Its content digest.
    pub digest: Digest,
    /// Bytes it occupies.
    pub bytes: u64,
    /// What the run measured.
    pub report: TrainReport,
}

/// The call into brain.
pub trait Trainer {
    /// The trainer's settings, for the configuration digest.
    fn describe(&self) -> Value;
    /// Trains a candidate.
    ///
    /// # Errors
    /// The training failed; nothing was adopted.
    fn train(&mut self, request: &TrainRequest) -> Result<Trained, CampaignError>;
}

/// The dev-validation exam.
pub trait Validator {
    /// The validator's settings, for the configuration digest.
    fn describe(&self) -> Value;
    /// Independent units the dev set has: families, not tasks.
    fn units(&self) -> usize;
    /// Tasks of the retention set; zero when the campaign keeps none.
    fn retention_units(&self) -> usize {
        0
    }
    /// The dev task ids, in the validator's order: the pool the evidence
    /// gate draws a round's items from.
    fn dev_tasks(&self) -> Vec<String>;
    /// Attempts `policy` at the dev tasks `dev` (ids of [`Self::dev_tasks`],
    /// never trained on) and, when the campaign keeps one, at every retention
    /// task, and returns pass or fail per task. No other dev task is
    /// attempted, so the evidence gate's unspent items stay unseen.
    ///
    /// # Errors
    /// The policy could not be attempted or graded.
    fn validate_on(
        &mut self,
        policy: &PolicyRef,
        dev: &[String],
    ) -> Result<DevResult, CampaignError>;
    /// [`Self::validate_on`] every dev task: what the greedy gate compares.
    ///
    /// # Errors
    /// The policy could not be attempted or graded.
    fn validate(&mut self, policy: &PolicyRef) -> Result<DevResult, CampaignError> {
        let dev = self.dev_tasks();
        self.validate_on(policy, &dev)
    }
}

/// What the strategies that need more than the batch's own rollouts call.
pub trait AllFailHandler {
    /// Verified answers for the tasks whose groups all failed, as imitation
    /// records (a teacher solving them, graded by the same verifier).
    ///
    /// # Errors
    /// The teacher could not be asked.
    fn teach(&mut self, groups: &[&RolloutGroup]) -> Result<Vec<Record>, CampaignError>;
    /// Easier tasks the policy might solve, split from the failed ones; they
    /// join the pool.
    ///
    /// # Errors
    /// The decomposition could not be made.
    fn decompose(&mut self, groups: &[&RolloutGroup]) -> Result<Vec<String>, CampaignError>;
}
