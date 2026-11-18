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
    /// The environment failed (build, runner, sandbox, an agent runtime that
    /// crashed for a reason other than its model). Such an attempt says
    /// nothing about the agent and is left out of the denominators.
    Environment,
    /// No objective evidence classifies it.
    Unclassified,
    /// The model gateway between the agent and the model server failed: it
    /// timed out waiting for the server (HTTP 504), could not reach it (502),
    /// or the agent could not reach the gateway. Infrastructure.
    Gateway,
    /// The model server refused or failed the request (another 5xx, a refusal
    /// or an unusable reply). Infrastructure.
    ModelServer,
    /// The agent never wrote the artifact the task grades: it stopped, or
    /// answered in its reply instead of the file.
    NoAnswer,
    /// The agent wrote the artifact, and the verifier could not read it.
    MalformedArtifact,
    /// The artifact was read and its content is wrong.
    WrongContent,
    /// The agent read the repository's history or hidden material the task
    /// withholds (the examiner's security veto).
    HistoryProbing,
    /// The agent repeated one action, unchanged, with the same result.
    Repetition,
    /// The agent or model cannot take a capability the task needs: coverage,
    /// not a failure of the work.
    Unsupported,
}

/// Who an attempt's failure is about.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Attribution {
    /// The model did the wrong thing: training material for a repair.
    Model,
    /// The run's infrastructure (sandbox, gateway, model server, verifier)
    /// failed; the attempt says nothing about the model.
    Infrastructure,
    /// The task itself is unsound, or asks for a capability the agent does
    /// not have; the attempt says nothing about the model either.
    Task,
}

impl FailureClass {
    /// Every class, in declaration order.
    pub const ALL: [Self; 20] = [
        Self::InformationMissing,
        Self::Retrieval,
        Self::Tool,
        Self::Strategy,
        Self::Planning,
        Self::Implementation,
        Self::Verification,
        Self::Resource,
        Self::ModelCompetence,
        Self::UnclearTask,
        Self::Environment,
        Self::Unclassified,
        Self::Gateway,
        Self::ModelServer,
        Self::NoAnswer,
        Self::MalformedArtifact,
        Self::WrongContent,
        Self::HistoryProbing,
        Self::Repetition,
        Self::Unsupported,
    ];

    /// Who the failure is about; `None` for [`FailureClass::Unclassified`],
    /// which no evidence attributes.
    #[must_use]
    pub fn attribution(self) -> Option<Attribution> {
        match self {
            Self::Verification | Self::Environment | Self::Gateway | Self::ModelServer => {
                Some(Attribution::Infrastructure)
            }
            Self::UnclearTask | Self::Unsupported => Some(Attribution::Task),
            Self::Unclassified => None,
            Self::InformationMissing
            | Self::Retrieval
            | Self::Tool
            | Self::Strategy
            | Self::Planning
            | Self::Implementation
            | Self::Resource
            | Self::ModelCompetence
            | Self::NoAnswer
            | Self::MalformedArtifact
            | Self::WrongContent
            | Self::HistoryProbing
            | Self::Repetition => Some(Attribution::Model),
        }
    }

    /// Whether the class is a failure of the model, never of the
    /// infrastructure or the task.
    #[must_use]
    pub fn is_model(self) -> bool {
        self.attribution() == Some(Attribution::Model)
    }

    /// Whether the class clears the agent: the infrastructure or the task
    /// decided the attempt, so it is left out of the agent's denominators.
    #[must_use]
    pub fn exonerates_agent(self) -> bool {
        matches!(
            self.attribution(),
            Some(Attribution::Infrastructure | Attribution::Task)
        )
    }

    /// The name the records write (`snake_case`).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::InformationMissing => "information_missing",
            Self::Retrieval => "retrieval",
            Self::Tool => "tool",
            Self::Strategy => "strategy",
            Self::Planning => "planning",
            Self::Implementation => "implementation",
            Self::Verification => "verification",
            Self::Resource => "resource",
            Self::ModelCompetence => "model_competence",
            Self::UnclearTask => "unclear_task",
            Self::Environment => "environment",
            Self::Unclassified => "unclassified",
            Self::Gateway => "gateway",
            Self::ModelServer => "model_server",
            Self::NoAnswer => "no_answer",
            Self::MalformedArtifact => "malformed_artifact",
            Self::WrongContent => "wrong_content",
            Self::HistoryProbing => "history_probing",
            Self::Repetition => "repetition",
            Self::Unsupported => "unsupported",
        }
    }

    /// The class the records name `name`.
    ///
    /// # Errors
    /// No class has that name.
    pub fn parse(name: &str) -> Result<Self, String> {
        Self::ALL
            .into_iter()
            .find(|c| c.as_str() == name)
            .ok_or_else(|| format!("{name:?} is not a failure class"))
    }
}

/// One unit of memory an attempt retrieved.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetrievedUnit {
    /// The unit.
    pub unit_id: String,
    /// The revision of the source it was cut from.
    pub revision: String,
    /// Whether it was put into the model's context.
    pub inserted: bool,
}

/// The memory an attempt read: which frozen snapshot, and which units it
/// retrieved from it. With the model's identity this is what an attempt's
/// result is about; it is how retrieval-assisted behaviour is told from a
/// change of weights.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryUse {
    /// The snapshot every retrieval read, when the memory had one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snapshot_id: Option<String>,
    /// Retrievals the attempt made.
    pub retrievals: u32,
    /// The distinct units retrieved, in the order first retrieved.
    pub units: Vec<RetrievedUnit>,
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
    /// The memory the attempt read, when it was given any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory: Option<MemoryUse>,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_class_is_named_as_it_serialises_and_parses_back() {
        for class in FailureClass::ALL {
            let json = serde_json::to_value(class).unwrap();
            assert_eq!(json, serde_json::Value::String(class.as_str().into()));
            assert_eq!(FailureClass::parse(class.as_str()), Ok(class));
        }
    }

    #[test]
    fn an_infrastructure_or_task_class_is_never_a_model_failure() {
        for class in FailureClass::ALL {
            assert!(
                !(class.is_model() && class.exonerates_agent()),
                "{class:?} is both"
            );
        }
        for class in [
            FailureClass::Environment,
            FailureClass::Gateway,
            FailureClass::ModelServer,
            FailureClass::Verification,
        ] {
            assert_eq!(class.attribution(), Some(Attribution::Infrastructure));
            assert!(!class.is_model());
        }
    }
}
