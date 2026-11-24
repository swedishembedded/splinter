// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training data built from what agents
// actually did, every record traceable to the attempt it came from, for its
// clients. If your team needs expertise in learning from agent experience,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! The typed parameters of the dataset operations' options: how a preference dataset pairs attempts, how a
//! policy-gradient dataset rewards them, and how a dataset is extended with new solutions.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

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
