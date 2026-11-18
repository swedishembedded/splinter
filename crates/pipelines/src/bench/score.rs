// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements tamper-evident hand-over of agent benchmark
// results into training pipelines, for its clients. If your team needs
// expertise in turning benchmark runs into verified training data, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The examiner's score record of a trial (`bench.score/1`): how the trial
//! ended, whether it is counted in pass rates, and the score the verifier
//! gave it. An exported manifest carries it as `score`; every reader of the
//! examiner reads this record, so the intake does too.
//!
//! Only a counted trial has a verdict. A trial the record does not count -
//! the sandbox, the model server, the agent or the verifier failed, a budget
//! ran out, the task was defective or unsupported - has no grade, and an
//! infrastructure failure is never a score of zero.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::artifact::Manifest;

/// The score record schema this reader understands.
pub const SCORE_SCHEMA: &str = "bench.score/1";

/// How a trial ended, as the score record says.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    /// The verifier graded the agent's work.
    Graded,
    /// The task is defective: no tests, or its verifier refuses its truth.
    InvalidTask,
    /// The sandbox, the staging, the gateway or a cancelled run failed.
    EnvironmentError,
    /// The model server refused, failed or could not be reached.
    ModelError,
    /// The agent crashed for a reason other than its model.
    AgentError,
    /// The verifier crashed, ran out of time, or left no reward.
    VerifierError,
    /// A time limit ended the agent; counted when the task grades timeouts.
    Timeout,
    /// A context, output or usage budget ran out.
    BudgetExhausted,
    /// The agent reached for the hidden answers: a hard veto.
    SecurityViolation,
    /// The agent or model cannot take what the task needs.
    Unsupported,
}

impl Status {
    /// The status named `name`, as the record writes it.
    ///
    /// # Errors
    /// The name is not a status of [`SCORE_SCHEMA`].
    pub fn parse(name: &str) -> Result<Self, String> {
        serde_json::from_value(Value::String(name.to_string()))
            .map_err(|_| format!("unknown score status {name:?}"))
    }

    /// The name the record writes.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Graded => "graded",
            Self::InvalidTask => "invalid_task",
            Self::EnvironmentError => "environment_error",
            Self::ModelError => "model_error",
            Self::AgentError => "agent_error",
            Self::VerifierError => "verifier_error",
            Self::Timeout => "timeout",
            Self::BudgetExhausted => "budget_exhausted",
            Self::SecurityViolation => "security_violation",
            Self::Unsupported => "unsupported",
        }
    }
}

/// What the trial was, as the score record identifies it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScoreIdentity {
    /// The task's name.
    pub task: String,
    /// The digest of the task's content.
    #[serde(default)]
    pub digest: Option<String>,
    /// The attempt: its place among the task's trials by start time.
    #[serde(default)]
    pub attempt: Option<u64>,
    /// The collection the task is of (its domain).
    #[serde(default)]
    pub collection: Option<String>,
    /// The model of each role.
    #[serde(default)]
    pub models: BTreeMap<String, String>,
}

/// A trial's score record; the fields this reader does not use are kept as
/// they are in the manifest, which stays the record of them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScoreRecord {
    /// Must be [`SCORE_SCHEMA`].
    pub schema: String,
    /// How the trial ended ([`Status`]).
    pub status: String,
    /// Whether the trial is in pass rates.
    pub counted: bool,
    /// The verifier's reward in `[0, 1]`; absent when it left none or the
    /// status makes it meaningless.
    #[serde(default)]
    pub score: Option<f64>,
    /// Whether a counted trial passed; absent for one not counted.
    #[serde(default)]
    pub pass: Option<bool>,
    /// Why a trial that is not graded ended as it did.
    #[serde(default)]
    pub cause: Option<Value>,
    /// What the trial used: requests, tokens, turns, tool calls, seconds.
    #[serde(default)]
    pub usage: Option<Value>,
    /// What the trial was.
    #[serde(default)]
    pub identity: Option<ScoreIdentity>,
}

impl ScoreRecord {
    /// How the trial ended.
    ///
    /// # Errors
    /// The record names a status [`SCORE_SCHEMA`] does not have.
    pub fn status(&self) -> Result<Status, String> {
        Status::parse(&self.status)
    }

    /// The cause's message, when the record states one.
    #[must_use]
    pub fn cause_message(&self) -> Option<&str> {
        self.cause
            .as_ref()
            .and_then(|c| c.get("message"))
            .and_then(Value::as_str)
    }

    /// Checks that the record is one this reader understands and that it
    /// says what the rest of `manifest` says: the same task, and for a
    /// counted trial the same verdict and reward.
    ///
    /// # Errors
    /// The reason the manifest is refused.
    pub fn agrees_with(&self, manifest: &Manifest) -> Result<(), String> {
        if self.schema != SCORE_SCHEMA {
            return Err(format!(
                "score schema {:?} is not {SCORE_SCHEMA:?}",
                self.schema
            ));
        }
        self.status()?;
        if let Some(identity) = &self.identity {
            if identity.task != manifest.task.id {
                return Err(format!(
                    "the score is of task {:?}, the manifest of {:?}",
                    identity.task, manifest.task.id
                ));
            }
            if let Some(digest) = &identity.digest {
                if !digest.eq_ignore_ascii_case(&manifest.task.digest) {
                    return Err(format!(
                        "the score is of task digest {digest}, the manifest of {}",
                        manifest.task.digest
                    ));
                }
            }
        }
        if !self.counted {
            return Ok(());
        }
        let passed = manifest.grade.verdict == "pass";
        if self.pass != Some(passed) {
            return Err(format!(
                "the score says pass {:?}, the grade's verdict is {:?}",
                self.pass, manifest.grade.verdict
            ));
        }
        if let Some(score) = self.score {
            if (score - manifest.grade.reward).abs() > f64::EPSILON {
                return Err(format!(
                    "the score is {score}, the grade's reward {}",
                    manifest.grade.reward
                ));
            }
        }
        Ok(())
    }
}
