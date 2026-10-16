// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gated self-improvement loops where a model
// update is adopted only on held-out evidence. If your team needs expertise
// in evaluation-gated fine-tuning, you can procure our services by sending
// an email to info@swedishembedded.com.

//! How the gate probes a candidate that works in a workspace, and what the
//! probe set aside: the bounds an attempt runs under, and the attempts that
//! were counted instead of graded.

use serde::{Deserialize, Serialize};

/// How many times a workspace task is attempted per arm when a gate names no
/// other number.
pub const DEFAULT_PROBE_ATTEMPTS: u32 = 1;
/// How many follow-up turns an agent is sent after the checks it was shown
/// stay red, when nothing names another number.
pub const DEFAULT_FOLLOW_UPS: u32 = 2;
/// How many tool calls an agent may make in one attempt, follow-ups
/// included, when nothing names another number.
pub const DEFAULT_MAX_TOOL_CALLS: u32 = 60;
/// How many times one tool call may return the same answer in an attempt
/// before the attempt is stopped as going nowhere, when nothing names
/// another number.
pub const DEFAULT_MAX_REPEATED_RESULTS: u32 = 3;

/// How an agent is probed on tasks that are changes to a repository: the
/// same for both arms of a comparison and for the served candidate, so a
/// difference between them is the weights'.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgenticConfig {
    /// Attempts per task. A task's verdict is what a strict majority of its
    /// graded attempts say; with no majority it has none.
    pub attempts: u32,
    /// Follow-up turns after the checks the task shows stay red.
    pub follow_ups: u32,
    /// Tool calls in one attempt, follow-ups included; `None` sets no cap.
    pub max_tool_calls: Option<u32>,
    /// Identical answers to one call before an attempt is stopped; `None`
    /// sets no bound.
    pub max_repeated_results: Option<u32>,
}

impl Default for AgenticConfig {
    fn default() -> Self {
        Self {
            attempts: DEFAULT_PROBE_ATTEMPTS,
            follow_ups: DEFAULT_FOLLOW_UPS,
            max_tool_calls: Some(DEFAULT_MAX_TOOL_CALLS),
            max_repeated_results: Some(DEFAULT_MAX_REPEATED_RESULTS),
        }
    }
}

/// The attempts at workspace tasks one model's probe set aside instead of
/// grading. A count of attempts, so zero is a measurement: nothing was set
/// aside.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Withheld {
    /// Attempts whose trajectory shows the solver saw what only the grader
    /// holds (the exposure audit failed them). They are no one's win or
    /// loss, whatever their patch did.
    pub exposed: usize,
    /// Attempts the harness could not carry out or grade: the workspace
    /// could not be laid out, or the verifier could not run. They say
    /// nothing about the model.
    pub environment_failures: usize,
}

impl Withheld {
    /// This and `other` together.
    #[must_use]
    pub fn plus(self, other: Self) -> Self {
        Self {
            exposed: self.exposed + other.exposed,
            environment_failures: self.environment_failures + other.environment_failures,
        }
    }
}

/// What probing both arms on workspace tasks set aside; see [`Withheld`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgenticProbe {
    /// Attempts per task.
    pub attempts: u32,
    /// What the candidate's probe set aside.
    pub candidate: Withheld,
    /// What the champion's probe set aside.
    pub champion: Withheld,
}
