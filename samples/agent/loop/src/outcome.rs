// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements self-improving coding agents whose every
// step is auditable, for its clients. If your team needs expertise in agent
// evaluation or locally operated coding agents, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The structured result a run hands back, and the checkpoint it leaves.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::acceptance::CheckResult;
use crate::repo::FileChange;

/// The version of the outcome record.
pub const OUTCOME_SCHEMA: u32 = 1;

/// How a run ended.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    /// A candidate passed every acceptance check and touched nothing protected.
    Accepted,
    /// Every attempt ended and none was accepted.
    Rejected,
    /// A time, token, tool-call or cost limit stopped the run.
    LimitExhausted,
    /// A person or process asked it to stop.
    Cancelled,
    /// The loop itself, or the model's provider, failed.
    Error,
}

/// How one attempt ended.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AttemptSummary {
    /// 1-based.
    pub attempt: u32,
    /// How sven concluded the run: `Success`, `BudgetExhausted`, `Timeout`,
    /// `Cancelled`, `AgentError` or `Waiting`.
    pub conclusion: String,
    /// Why the attempt was not accepted, when it was not.
    pub failure: Option<String>,
    /// Tool calls the model requested.
    pub tool_calls: u32,
    /// Whether the tool-call cap stopped it.
    pub stopped_by_tool_cap: bool,
    /// Files it changed, as git reports them.
    pub changed_files: Vec<FileChange>,
    /// Every acceptance check it was judged by.
    pub checks: Vec<CheckResult>,
    /// What the worker said when it finished.
    pub final_message: Option<String>,
    /// Seconds the attempt took.
    pub seconds: f64,
    /// Input tokens, where the provider reported them.
    pub input_tokens: Option<u64>,
    /// Output tokens, where the provider reported them.
    pub output_tokens: Option<u64>,
}

/// What the run spent in all, failed attempts included.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Usage {
    /// Input tokens the providers reported.
    pub input_tokens: u64,
    /// Output tokens the providers reported.
    pub output_tokens: u64,
    /// Dollars the provider reported; absent when none did.
    pub cost_usd: Option<f64>,
    /// Model calls that reported no price (never counted as free).
    pub calls_without_price: u32,
    /// Seconds from the first attempt to the last.
    pub wall_seconds: f64,
}

/// Where a run's evidence is.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Artifacts {
    /// The run's directory.
    pub run_dir: PathBuf,
    /// The append-only event stream.
    pub events: PathBuf,
    /// The patch of the last candidate.
    pub patch: Option<PathBuf>,
    /// The checkout the last candidate lives in.
    pub worktree: PathBuf,
    /// The agent trajectory of each attempt (ATIF).
    pub trajectories: Vec<PathBuf>,
}

/// The final record of a run.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Outcome {
    /// The record's version.
    pub schema: u32,
    /// The run's id.
    pub run: String,
    /// How it ended.
    pub status: Status,
    /// `unaided` when no supervisor hint was given, else `assisted`.
    pub assistance: String,
    /// The model as asked for, and as it identified itself.
    pub model_requested: String,
    /// The identity the model reported.
    pub model_identity: String,
    /// Each attempt, in order.
    pub attempts: Vec<AttemptSummary>,
    /// Files the last candidate changed.
    pub changed_files: Vec<FileChange>,
    /// The acceptance results of the last candidate.
    pub validation: Vec<CheckResult>,
    /// What is not settled: a failed check, a protected path touched, a
    /// limit that stopped the work.
    pub unresolved: Vec<String>,
    /// What it cost.
    pub usage: Usage,
    /// Where the evidence is.
    pub artifacts: Artifacts,
    /// Why it stopped, in a sentence.
    pub termination: String,
}

/// The last durable point of a run, enough to resume it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Attempts that finished.
    pub attempts_done: u32,
    /// Summaries of those attempts.
    pub attempts: Vec<AttemptSummary>,
    /// Usage so far.
    pub usage: Usage,
    /// Seconds of the total budget already used.
    pub elapsed_seconds: f64,
    /// Whether the run reached a final state.
    pub finished: bool,
}
