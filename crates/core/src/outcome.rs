// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements measurements of what an agent really
// completed, and at what cost, for its clients. If your team needs expertise
// in benchmarking coding agents, you can procure our services by sending an
// email to info@swedishembedded.com.

//! What one attempt at a task came to, as a measurement.
//!
//! A [`TaskOutcome`] is not stored experience. It is an evaluation derived
//! from an experience, the annotations about it and the facts of the attempt
//! ([`AttemptReport`], recorded as an annotation because an experience is
//! immutable), so it can be recomputed from the log and never disagrees
//! with it. A value that was not measured is `None`, never `0`.

use serde::{Deserialize, Serialize};

use crate::annotation::Strength;
use crate::digest::Digest;
use crate::experience::ExperienceId;

/// Why an attempt did not complete its task.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureClass {
    /// The agent never had the information the change needs.
    InformationMissing,
    /// It searched and the relevant material never surfaced.
    Retrieval,
    /// A tool failed or was used wrongly.
    Tool,
    /// The harness policy (when to test, when to stop) was wrong.
    Strategy,
    /// The plan of the change was wrong.
    Planning,
    /// The right files were touched and the tests still fail.
    Implementation,
    /// The verifier, not the work, was at fault.
    Verification,
    /// A time, token or tool-call bound stopped the run.
    Resource,
    /// The model could not do what the task needs.
    ModelCompetence,
    /// The task itself is unclear or unsound.
    UnclearTask,
    /// The environment failed (build, runner, sandbox). Such an attempt
    /// says nothing about the agent and is left out of the denominators.
    Environment,
    /// No objective evidence classifies it.
    Unclassified,
}

/// The facts of an attempt that its trajectory and verdicts do not carry.
/// Every field is optional: what the run did not record is absent.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AttemptReport {
    /// Wall-clock seconds the attempt took.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wall_seconds: Option<f64>,
    /// Device-busy seconds, when the attempt ran on a local device.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_seconds: Option<f64>,
    /// Tool calls the model requested.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<u32>,
    /// Tool calls that failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_errors: Option<u32>,
    /// Actions the agent attempted that the gate checked.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attempted_actions: Option<u32>,
    /// Of those, the ones the gate refused as unauthorised.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unauthorised_actions: Option<u32>,
    /// Whether the run was stopped by the cap on tool calls.
    #[serde(default)]
    pub stopped_by_tool_cap: bool,
    /// Whether the run was stopped for repeating one call.
    #[serde(default)]
    pub stopped_by_repetition: bool,
    /// Whether the agent handed in a result.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub submitted: Option<bool>,
    /// Minutes a person spent in the loop, when one was.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub human_minutes: Option<f64>,
    /// The failure class, when something classified the attempt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure: Option<FailureClass>,
}

/// One attempt at one task, measured.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TaskOutcome {
    /// The task attempted.
    pub task_id: Digest,
    /// The attempt.
    pub experience_id: ExperienceId,
    /// Whether the verdicts decide the work is correct. An attempt no
    /// verifier decided is not verified.
    pub verified_success: bool,
    /// The strength of the verdicts that decided it; `None` when none did.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verification_strength: Option<Strength>,
    /// Whether the attempt handed in a result.
    pub submitted: bool,
    /// Wall-clock seconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wall_seconds: Option<f64>,
    /// Device-busy seconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_seconds: Option<f64>,
    /// Prompt tokens over the attempt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u64>,
    /// Completion tokens over the attempt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u64>,
    /// Tool calls requested.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<u32>,
    /// Tool calls that failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_errors: Option<u32>,
    /// Actions the gate checked.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attempted_actions: Option<u32>,
    /// Actions the gate refused as unauthorised.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unauthorised_actions: Option<u32>,
    /// Why the attempt failed, when it did and something classified it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure: Option<FailureClass>,
    /// Minutes a person spent in the loop; absent when none was.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub human_minutes: Option<f64>,
}
