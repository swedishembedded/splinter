// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agent execution backends whose every
// attempt is accounted for, for its clients. If your team needs expertise
// in reliable agent orchestration, you can procure our services by sending
// an email to info@swedishembedded.com.

//! What an execution worker reports back for one physical attempt: the raw
//! evidence the controller grades, and where the backend's own files are.
//!
//! A worker never grades: it reports what it saw, and the controller applies
//! the one classifier ([`super::trial::grade`]) under the run's policy, so a
//! worker of another version cannot change what a trial counts as.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::execution::SlotKey;
use super::product::TrialUsage;
use super::trial::TrialEvidence;

/// The schema of an [`AttemptReport`].
pub const ATTEMPT_SCHEMA: &str = "splinter.attempt/1";

/// One physical attempt, as its worker saw it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AttemptReport {
    /// Always [`ATTEMPT_SCHEMA`].
    pub schema: String,
    /// The slot the attempt filled.
    pub slot: SlotKey,
    /// The attempt's number.
    pub attempt: u32,
    /// The facts the classifier reads.
    pub evidence: TrialEvidence,
    /// The backend's own trial directory, relative to the worker's output
    /// directory.
    pub native_dir: String,
    /// The ATIF trajectory, relative to the worker's output directory, when
    /// the agent left one.
    #[serde(default)]
    pub trace: Option<String>,
    /// Every other file of the trial the evidence was read from, relative to
    /// the worker's output directory.
    #[serde(default)]
    pub native_files: Vec<String>,
    /// The version the agent reported.
    #[serde(default)]
    pub agent_version: Option<String>,
    /// When the attempt started (RFC 3339).
    #[serde(default)]
    pub started_at: Option<String>,
    /// When it finished (RFC 3339).
    #[serde(default)]
    pub finished_at: Option<String>,
    /// What it consumed.
    #[serde(default)]
    pub usage: TrialUsage,
    /// The sandbox it ran in; `None` from a worker that does not say.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sandbox: Option<super::execution::SandboxRecord>,
}
