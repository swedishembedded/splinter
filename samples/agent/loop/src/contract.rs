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

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use splinter_sdk::sandbox::CommandCall;

/// The version of the contract record.
pub const CONTRACT_SCHEMA: u32 = 2;

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
    /// Times one call may return the same answer before the attempt is
    /// stopped as going nowhere (0 sets no bound).
    pub const MAX_REPEATED_RESULTS: u32 = 4;
}

/// The limits of one run.
/// One named check.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Check {
    /// What the check is called in the record.
    pub name: String,
    /// The shell command, run in a fresh copy of the repository's snapshot
    /// with the candidate's patch applied. It may point at a script anywhere
    /// the supervisor can read: that script is never in the worker's tree.
    pub command: String,
    /// Whether the worker is shown the command, and told the output of the
    /// check when it fails. A hidden check is named in the prompt and
    /// nothing more.
    #[serde(default)]
    pub visible: bool,
}

impl Check {
    /// `NAME=COMMAND`, as the command line gives it.
    pub fn parse(text: &str, visible: bool) -> Result<Self> {
        let (name, command) = text
            .split_once('=')
            .filter(|(n, c)| !n.trim().is_empty() && !c.trim().is_empty())
            .with_context(|| format!("an acceptance check is NAME=COMMAND, not {text:?}"))?;
        Ok(Self {
            name: name.trim().to_string(),
            command: command.trim().to_string(),
            visible,
        })
    }

    /// The command as the sandbox runs it: through `sh -c`.
    #[must_use]
    pub fn call(&self) -> CommandCall {
        CommandCall {
            argv: vec!["sh".into(), "-c".into(), self.command.clone()],
            stdin: None,
        }
    }
}

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
    /// Times one call (same tool, same arguments) may return the same
    /// answer before the attempt is stopped as going nowhere; 0 sets no
    /// bound. A repeated call whose answer changes is not counted.
    #[serde(default)]
    pub max_repeated_results: u32,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_check_is_name_equals_command() {
        assert!(Check::parse("unit=python3 -m unittest", true).is_ok());
        assert!(Check::parse("no-equals", true).is_err());
        assert!(Check::parse("=cmd", true).is_err());
    }
}
