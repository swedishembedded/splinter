// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training datasets whose every
// parameter is checked before a run spends anything, for its clients. If your
// team needs expertise in machine-learning data pipelines then you can procure
// our services by sending an email to info@swedishembedded.com.

//! The typed parameters of the dataset operations: `dataset.import`,
//! `dataset.extend`, `dataset.text` and `dataset.project`.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// `dataset.import`: training records made elsewhere, as a dataset
/// ([`crate::run::dataset_import`]).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ImportDatasetParams {
    /// The model role the records are for.
    pub target: String,
    /// The longest record, in the target's tokens, when bounded: a longer
    /// one is excluded and listed.
    #[serde(default)]
    pub max_block: Option<u64>,
    /// The records keep the target's reasoning (it is trained to reason
    /// before it answers); off: no-think mode, and a record that carries
    /// reasoning is refused.
    #[serde(default)]
    pub thinking: bool,
}

/// `dataset.extend`: new records admitted into an existing dataset only as
/// new solutions of their task ([`crate::run::dataset_extend`]).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ExtendDatasetParams {
    /// The model role the dataset is for.
    pub target: String,
    /// The most records a task has in the result, the base's included.
    #[serde(default = "four")]
    pub per_task: usize,
    /// Argument names left out when two solutions' tool calls are compared:
    /// the free text a call carries beside what it does (a stated intent).
    #[serde(default)]
    pub ignore_call_arguments: Vec<String>,
    /// Prefixes taken off a record's task id (its `metadata.task`, or its `metadata.group` when it has no task)
    /// to make the key two datasets compare tasks by: records written from a benchmark and records projected from a
    /// run name one task in different ways (`swedishembedded/vim-001-x`, `train/001-x`). A prefix is taken off
    /// repeatedly, in the order given, until none applies.
    #[serde(default)]
    pub task_prefixes: Vec<String>,
    /// The share, in percent, of the tasks new to the base that are held out
    /// (as whole tasks). A task the base already has stays on its side.
    #[serde(default = "ten")]
    pub holdout_percent: u8,
    /// The seed the held-out draw of a new task is made by; keep the base's.
    pub holdout_seed: u64,
}

fn four() -> usize {
    4
}
fn ten() -> u8 {
    10
}

/// How a text dataset holds documents out for validation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DocumentSplit {
    /// The share of documents held out, in percent.
    pub percent: u32,
}

/// `dataset.text`: documents as a continued-pretraining dataset
/// ([`crate::run::dataset_text`]).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TextDatasetParams {
    /// The model role the records are for.
    pub target: String,
    /// How documents are held out for validation; none when unset.
    #[serde(default)]
    pub validation: Option<DocumentSplit>,
    /// The seed the held-out documents are drawn by.
    pub seed: u64,
}

fn sft() -> String {
    "sft".into()
}
fn verified_success() -> String {
    "verified-success".into()
}
fn native_sft() -> String {
    "native-sft".into()
}
fn by_task() -> String {
    "task".into()
}

/// How validation records are held out.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ValidationSplit {
    /// Group records by this before splitting (`task`, `task_family`).
    #[serde(default = "by_task")]
    pub group_by: String,
    /// The fraction of groups held out, in percent.
    pub percent: u32,
}

/// The objectives `dataset.project` projects attempts for.
pub const OBJECTIVES: &[&str] = &[
    "sft",
    "preference",
    "feedback",
    "policy-gradient",
    "distill",
    "retrieval",
];

/// How a preference dataset pairs attempts.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PreferenceOptions {
    /// The most pairs one task (by one policy) gives, drawn by the stage's
    /// seed when it has more; every pair when unset.
    #[serde(default)]
    pub pairs_per_task: Option<usize>,
}

/// How a policy-gradient dataset rewards an attempt beyond its grade.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RewardOptions {
    /// Behaviour metrics of an attempt (`loop`: a loop watchdog ended it; `repeat_after_error`: the share of the
    /// calls after a failed call that repeated it) and, for each, the percent of a pass's reward it costs the
    /// attempt at 1, whatever the attempt's outcome.
    #[serde(default)]
    pub penalties_percent: std::collections::BTreeMap<String, u32>,
}

/// Which assistant actions of an SFT record are supervised.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum Supervise {
    /// Every action of the agent.
    #[default]
    All,
    /// Every action but one whose tool calls were all answered with an error:
    /// that one stays in the context, unsupervised. The action that recovers
    /// and the final answer stay supervised.
    Productive,
}

/// Which admitted trials of a task an SFT dataset keeps when it keeps only
/// `per_task` of them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum Prefer {
    /// The trials with the fewest actions, ties drawn by the stage's seed.
    #[default]
    FewestActions,
    /// Trials drawn by the stage's seed.
    Random,
}

/// A negative control an SFT dataset is built as.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum Control {
    /// Each record's task request (its first user message) is the request of
    /// another admitted task, by a derangement of the tasks drawn by the
    /// stage's seed; everything else is the record's own.
    ShuffledRequests,
}

impl Control {
    /// The name a recipe and the manifest give it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::ShuffledRequests => "shuffled-requests",
        }
    }
}

/// `dataset.project`: training records projected from attempts.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ProjectParams {
    /// The objective (`sft`, `preference`, `feedback`, `policy-gradient`,
    /// `distill`, `retrieval`).
    #[serde(default = "sft")]
    pub objective: String,
    /// The admission rule of an SFT dataset.
    #[serde(default = "verified_success")]
    pub admission: String,
    /// The projection of an SFT dataset.
    #[serde(default = "native_sft")]
    pub projection: String,
    /// How a preference dataset pairs attempts.
    #[serde(default)]
    pub preference: Option<PreferenceOptions>,
    /// How a policy-gradient dataset rewards an attempt beyond its grade.
    #[serde(default)]
    pub reward: Option<RewardOptions>,
    /// The model role the records are for.
    pub target: String,
    /// The longest record, in the target's tokens, when bounded.
    #[serde(default)]
    pub max_block: Option<u64>,
    /// Train the target to reason before it answers (off: no-think mode).
    #[serde(default)]
    pub thinking: bool,
    /// How validation records are held out.
    #[serde(default)]
    pub validation: Option<ValidationSplit>,
    /// Which actions of an SFT record are supervised (`all`, `productive`).
    #[serde(default)]
    pub supervise: Supervise,
    /// The most admitted trials an SFT dataset keeps of one task; every one
    /// when unset. The others are excluded as `per_task_limit`.
    #[serde(default)]
    pub per_task: Option<usize>,
    /// Which trials `per_task` keeps: `fewest-actions` (the default) or
    /// `random`.
    #[serde(default)]
    pub prefer: Option<Prefer>,
    /// Build the SFT dataset as a negative control (`shuffled-requests`).
    #[serde(default)]
    pub control: Option<Control>,
    /// The seed of the split and of every draw the projection makes.
    pub seed: u64,
}
