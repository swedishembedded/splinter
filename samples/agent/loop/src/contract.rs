// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements self-improving coding agents whose every
// step is auditable, for its clients. If your team needs expertise in agent
// evaluation or locally operated coding agents, you can procure our
// services by sending an email to info@swedishembedded.com.

//! What a run was asked to do, and what it may spend doing it.
//!
//! The contract is written before the first model call and never changed:
//! a later reader compares what was done with exactly what was agreed. The
//! budgets are the loop's own and visible in the record; nothing raises
//! them silently.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::acceptance::Check;

/// The version of the contract record.
pub const CONTRACT_SCHEMA: u32 = 1;

/// The model a run uses when it is not told: a local checkpoint, never a
/// remote model.
pub const DEFAULT_MODEL: &str = "local:Qwen/Qwen3-8B";

/// Defaults of the per-run limits, shown by `--help` and recorded.
pub mod defaults {
    /// Seconds one attempt may take.
    pub const ATTEMPT_SECS: u64 = 900;
    /// Seconds the whole run may take, failed attempts included.
    pub const TOTAL_SECS: u64 = 3600;
    /// Output tokens one attempt's model may generate.
    pub const MAX_OUTPUT_TOKENS: u64 = 24_000;
    /// Tool calls one attempt may request.
    pub const MAX_TOOL_CALLS: u32 = 60;
    /// Attempts, the first included.
    pub const MAX_ATTEMPTS: u32 = 3;
    /// Retries of a model call that failed in transit within one attempt.
    pub const PROVIDER_RETRIES: u32 = 2;
    /// Rounds an attempt may go on in the same conversation, told what
    /// failed, after the worker stops with the checks still red.
    pub const FOLLOW_UPS: u32 = 2;
    /// Seconds an acceptance check may take.
    pub const CHECK_SECS: u64 = 300;
}

/// The limits of one run.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Limits {
    /// Seconds one attempt may take.
    pub attempt_secs: u64,
    /// Seconds the whole run may take.
    pub total_secs: u64,
    /// Output tokens per attempt.
    pub max_output_tokens: u64,
    /// Tool calls per attempt.
    pub max_tool_calls: u32,
    /// Attempts, the first included.
    pub max_attempts: u32,
    /// Retries of a model call that failed in transit, per attempt.
    pub provider_retries: u32,
    /// Further rounds in the same conversation after the worker stops with
    /// the checks red, per attempt.
    #[serde(default)]
    pub follow_ups: u32,
    /// The most a remote model may cost, in US dollars, as the provider
    /// reports it. `None` for a local model; for a remote model it is the
    /// cap, and a provider that reports no price is stopped by the token
    /// budget alone (an unknown price is never treated as free).
    pub max_cost_usd: Option<f64>,
}

/// The agreement a run is carried out under.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Contract {
    /// The record's version.
    pub schema: u32,
    /// The task, in the requester's words.
    pub task: String,
    /// The repository the task is about, as authorised.
    pub repository: PathBuf,
    /// The revision the work starts from.
    pub baseline_rev: String,
    /// Shell commands run once in the fresh checkout before any attempt,
    /// for what the repository does not carry itself (an untracked build
    /// configuration, say). Their output is part of the record.
    #[serde(default)]
    pub setup: Vec<String>,
    /// The checks that decide acceptance.
    pub acceptance: Vec<Check>,
    /// Paths (files or directory prefixes) the worker must not change.
    pub protected: Vec<String>,
    /// The limits.
    pub limits: Limits,
    /// The model reference as asked for.
    pub model: String,
    /// Whether models reached over an API were allowed.
    pub allow_api_models: bool,
    /// SHA-256 of the system prompt in effect.
    pub system_prompt_digest: String,
    /// Where the system prompt came from: `built-in` or a file.
    pub system_prompt_source: String,
    /// SHA-256 over the subagent, skill, command and project-context
    /// definitions sven finds for the repository (see `definitions`); empty
    /// in a contract written before this was recorded.
    #[serde(default)]
    pub definitions_digest: String,
    /// Hints the supervisor supplied; a run with any is an assisted one.
    pub hints: Vec<String>,
}

impl Contract {
    /// Whether the supervisor helped: unaided results are counted apart.
    #[must_use]
    pub fn assisted(&self) -> bool {
        !self.hints.is_empty()
    }
}
