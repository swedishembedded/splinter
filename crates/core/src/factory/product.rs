// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training-data lineage from every model
// back to the agent trials it learned from, for its clients. If your team
// needs expertise in auditable machine-learning pipelines, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The products a recipe's stages hand each other: trial results, datasets,
//! model candidates and evaluation results.

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::execution::{FileRef, SlotKey};
use super::trial::Grade;
use crate::digest::{digest_of, Digest};

/// The schema of a [`TrialResult`].
pub const TRIAL_SCHEMA: &str = "splinter.trial/1";
/// The schema of a [`DatasetProduct`].
pub const DATASET_SCHEMA: &str = "splinter.dataset/1";
/// The schema of a [`ModelCandidate`].
pub const CANDIDATE_SCHEMA: &str = "splinter.candidate/1";
/// The schema of an [`EvalResult`].
pub const EVAL_SCHEMA: &str = "splinter.eval/1";

/// Who acted in a trial.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct AgentIdentity {
    /// The agent adapter.
    pub name: String,
    /// The version the agent reported.
    #[serde(default)]
    pub version: Option<String>,
    /// The digest of the agent's invocation (argv, files, environment
    /// names).
    pub invocation: Digest,
}

/// Tokens and cost an attempt's model requests consumed; a value nobody
/// measured is absent, never 0.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TrialUsage {
    /// Prompt tokens.
    #[serde(default)]
    pub prompt_tokens: Option<u64>,
    /// Completion tokens.
    #[serde(default)]
    pub completion_tokens: Option<u64>,
    /// Prompt tokens served from cache.
    #[serde(default)]
    pub cached_tokens: Option<u64>,
    /// Cost in US dollars, when the provider reported it.
    #[serde(default)]
    pub cost_usd: Option<f64>,
}

/// The validated outcome of one physical attempt.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TrialResult {
    /// Always [`TRIAL_SCHEMA`].
    pub schema: String,
    /// The slot it fills.
    pub slot: SlotKey,
    /// The physical attempt.
    pub attempt: u32,
    /// The task's id.
    pub task: String,
    /// The task's content digest.
    pub task_digest: Digest,
    /// Who acted.
    pub agent: AgentIdentity,
    /// The model, `connection:model`, when the recipe bound one.
    #[serde(default)]
    pub model: Option<String>,
    /// The classified, scored outcome.
    pub grade: Grade,
    /// The ATIF trajectory, relative to the rollout product.
    #[serde(default)]
    pub trace: Option<FileRef>,
    /// The backend's own trial directory, relative to the rollout product.
    pub native_dir: String,
    /// The backend's files the grade was read from.
    #[serde(default)]
    pub native: Vec<FileRef>,
    /// When the attempt started (RFC 3339).
    #[serde(default)]
    pub started_at: Option<String>,
    /// When it finished (RFC 3339).
    #[serde(default)]
    pub finished_at: Option<String>,
    /// What it consumed.
    #[serde(default)]
    pub usage: TrialUsage,
}

impl TrialResult {
    /// The result's content digest, its identity in a collection.
    #[must_use]
    pub fn digest(&self) -> Digest {
        digest_of(self)
    }

    /// Whether the trial is a verified success: counted, passed, no veto.
    #[must_use]
    pub fn verified_success(&self) -> bool {
        self.grade.counted && self.grade.passed == Some(true) && self.grade.vetoes.is_empty()
    }
}

/// One row of a dataset's exclusion waterfall.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct WaterfallStep {
    /// What the step counts (`raw_trials`, `graded_successes`, ...).
    pub step: String,
    /// How many remained after it.
    pub remaining: u64,
}

/// A training dataset built from trial collections.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct DatasetProduct {
    /// Always [`DATASET_SCHEMA`].
    pub schema: String,
    /// The dataset's name in its recipe.
    pub name: String,
    /// The training objective (`sft`).
    pub objective: String,
    /// The projection (`native-sft`, `portable-sft`).
    pub projection: String,
    /// The admission rule (`verified-success`).
    pub admission: String,
    /// The model the records are rendered for.
    pub target: String,
    /// The rollout products it was built from.
    pub sources: Vec<Digest>,
    /// The training records.
    pub train: FileRef,
    /// The validation records.
    pub validation: FileRef,
    /// Trials admitted, one line each.
    pub accepted: FileRef,
    /// Trials excluded, each with its reason.
    pub excluded: FileRef,
    /// Supervised spans mapped back to their trials and messages.
    pub supervision: FileRef,
    /// The exclusion waterfall, in order.
    pub waterfall: Vec<WaterfallStep>,
    /// Exclusions by exact reason.
    pub exclusions: BTreeMap<String, u64>,
}

/// The form a trained model takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CandidateKind {
    /// An adapter over the base.
    Adapter,
    /// Full weights.
    Full,
}

/// A trained model, ready to be served.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ModelCandidate {
    /// Always [`CANDIDATE_SCHEMA`].
    pub schema: String,
    /// The training variant's name.
    pub name: String,
    /// Its form.
    pub kind: CandidateKind,
    /// The base model reference (`brain:Qwen/Qwen3-4B`).
    pub base: String,
    /// The trained files (adapter or weights), relative to the product.
    pub artifacts: Vec<FileRef>,
    /// The dataset product it was trained on.
    pub dataset: Digest,
    /// The training method.
    pub method: String,
    /// The parameters as the trainer accepted them.
    pub params: Value,
    /// The trainer and its version.
    pub trainer: String,
    /// The training seed.
    pub seed: u64,
    /// The trainer's report (losses, steps), as it wrote it.
    #[serde(default)]
    pub report: Value,
    /// The model id the candidate is served under once loaded.
    #[serde(default)]
    pub served_as: Option<String>,
    /// The serving check passed: the trainer's server loaded the candidate
    /// and answered.
    pub load_checked: bool,
}

/// One evaluated arm.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ArmResult {
    /// The arm (`baseline` or a candidate's name).
    pub arm: String,
    /// The model the arm served.
    pub model: String,
    /// Trials run.
    pub trials: u64,
    /// Trials counted.
    pub counted: u64,
    /// Trials passed.
    pub passed: u64,
    /// The pass rate over counted trials; absent when none was counted.
    #[serde(default)]
    pub pass_rate: Option<f64>,
    /// The rollout product holding its trials.
    pub trials_product: Digest,
}

/// What an evaluation decided.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Decision {
    /// The candidate is better under the plan's rule.
    Improved,
    /// It is not.
    NotImproved,
    /// The evidence does not decide.
    Inconclusive,
}

/// A candidate's evaluation against a baseline under a frozen plan.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct EvalResult {
    /// Always [`EVAL_SCHEMA`].
    pub schema: String,
    /// The plan's digest.
    pub plan: Digest,
    /// Whether the plan is protected (sealed tasks, signed verdict).
    pub protected: bool,
    /// The baseline arm.
    pub baseline: ArmResult,
    /// The candidate arm.
    pub candidate: ArmResult,
    /// The decision under the plan's rule.
    pub decision: Decision,
    /// The rule's statistics (difference, interval, p-value), as computed.
    #[serde(default)]
    pub statistics: Value,
    /// The evaluation authority's signature over the result, on a protected
    /// plan.
    #[serde(default)]
    pub signature: Option<String>,
}
