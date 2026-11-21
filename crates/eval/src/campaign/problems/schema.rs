// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for choosing the next training
// experiment from the evidence already paid for, for its clients. If your team
// needs expertise in planning model-improvement campaigns you can procure our
// services by sending an email to info@swedishembedded.com.

//! The problem state's schema (`splinter.problem-state/1` documents and
//! `splinter.problem-event/1` events): the parameters, problems, candidates,
//! evidence links, verdicts and revisions.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::rules::{DEFAULT_ALPHA, DEFAULT_BELIEF_DOWN, DEFAULT_BELIEF_UP};
use super::{invalid, valid_path, ProblemError, DEFAULT_ENGINEERING, EVENT_SCHEMA, EVIDENCE_KINDS};

/// The rules' parameters. Every one is stated in the document or takes the
/// default named here; a document's parameters replace these whole.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Parameters {
    /// What a supported verdict adds to a belief.
    #[serde(default = "default_belief_up")]
    pub belief_up: f64,
    /// What a falsified verdict takes from a belief.
    #[serde(default = "default_belief_down")]
    pub belief_down: f64,
    /// The level every interval is at (`1 - alpha`) and every test decides
    /// at.
    #[serde(default = "default_alpha")]
    pub alpha: f64,
    /// The least effect a hypothesis must show, per metric (`delta_target`).
    #[serde(default = "default_targets")]
    pub targets: BTreeMap<String, f64>,
    /// The configuration paths an engineering change may touch.
    #[serde(default = "default_engineering")]
    pub engineering: Vec<String>,
}

fn default_belief_up() -> f64 {
    DEFAULT_BELIEF_UP
}

fn default_belief_down() -> f64 {
    DEFAULT_BELIEF_DOWN
}

fn default_alpha() -> f64 {
    DEFAULT_ALPHA
}

fn default_targets() -> BTreeMap<String, f64> {
    BTreeMap::from([("pass@1".to_string(), 0.05)])
}

fn default_engineering() -> Vec<String> {
    DEFAULT_ENGINEERING
        .iter()
        .map(ToString::to_string)
        .collect()
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            belief_up: DEFAULT_BELIEF_UP,
            belief_down: DEFAULT_BELIEF_DOWN,
            alpha: DEFAULT_ALPHA,
            targets: default_targets(),
            engineering: default_engineering(),
        }
    }
}

impl Parameters {
    pub(super) fn validate(&self) -> Result<(), ProblemError> {
        let step = |name: &str, v: f64| {
            if v.is_finite() && v > 0.0 && v <= 1.0 {
                Ok(())
            } else {
                invalid(format!("parameters.{name} is {v}, not in (0, 1]"))
            }
        };
        step("belief_up", self.belief_up)?;
        step("belief_down", self.belief_down)?;
        if !(self.alpha.is_finite() && self.alpha > 0.0 && self.alpha < 1.0) {
            return invalid(format!("parameters.alpha is {}, not in (0, 1)", self.alpha));
        }
        for (metric, target) in &self.targets {
            if metric.trim().is_empty() || !(target.is_finite() && *target > 0.0) {
                return invalid(format!(
                    "parameters.targets.{metric:?} is {target}: a metric names a target above 0"
                ));
            }
        }
        for path in &self.engineering {
            valid_path("parameters.engineering", path)?;
        }
        Ok(())
    }
}

/// A link to where evidence lives: `<kind>:<id>` and a note. It never copies
/// the evidence.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Evidence {
    /// `<kind>:<id>`, the kind one of [`EVIDENCE_KINDS`].
    #[serde(rename = "ref")]
    pub reference: String,
    /// What it shows.
    #[serde(default)]
    pub note: String,
}

impl Evidence {
    pub(super) fn validate(&self) -> Result<(), ProblemError> {
        match self.reference.split_once(':') {
            Some((kind, id)) if EVIDENCE_KINDS.contains(&kind) && !id.trim().is_empty() => Ok(()),
            _ => invalid(format!(
                "evidence {:?}: a link is <kind>:<id> with kind one of {}",
                self.reference,
                EVIDENCE_KINDS.join(", ")
            )),
        }
    }
}

/// Whether a problem still limits the model.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProblemStatus {
    /// It does: its candidates are proposed.
    #[default]
    Open,
    /// It no longer does: none of its candidates is proposed.
    Resolved,
}

/// Where a candidate stands.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateStatus {
    /// Not yet supported: a hypothesis.
    #[default]
    Pending,
    /// Supported: part of the champion.
    Adopted,
}

/// Which phase a candidate belongs to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    /// A change to the model, the algorithm or the data.
    #[default]
    Hypothesis,
    /// A throughput or serving change that may touch only the engineering
    /// paths.
    Engineering,
}

/// What one experiment showed about one candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// The effect is shown and at least the target (or the gate promoted).
    Supported,
    /// An effect the size of the target is ruled out.
    Falsified,
    /// Measured, but neither: the interval holds both.
    Inconclusive,
    /// Not evidence: a crash, a measurement the infrastructure decided, a
    /// verdict the gate did not accept.
    Invalid,
}

/// A measured paired effect at the state's level.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Effect {
    /// Arm less champion.
    pub difference: f64,
    /// The interval's lower bound.
    pub low: f64,
    /// Its upper bound.
    pub high: f64,
    /// The p-value of the paired test, when there is one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub p: Option<f64>,
}

/// One verdict in a candidate's history.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Outcome {
    /// The experiment that reached it.
    pub experiment: String,
    /// The verdict.
    pub verdict: Verdict,
    /// The belief before.
    pub belief_before: f64,
    /// The belief after.
    pub belief_after: f64,
    /// The effect it rests on, when measured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effect: Option<Effect>,
    /// Where it is recorded.
    #[serde(default)]
    pub evidence: Vec<Evidence>,
    /// Why, in words.
    pub reason: String,
}

/// A coordinator's revision of a belief or a status.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Revision {
    /// The value before, as text.
    pub before: String,
    /// The value after, as text.
    pub after: String,
    /// Why.
    pub reason: String,
}

/// A candidate intervention against a problem.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Candidate {
    /// Its id, unique across problems.
    pub id: String,
    /// Its kind (`teacher-distillation`, `concise-traces`, `star`, ...).
    pub intervention: String,
    /// What it does.
    pub statement: String,
    /// The one configuration path its arm changes.
    pub variable: String,
    /// The value at that path, when the configuration can express it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<serde_json::Value>,
    /// Its phase.
    #[serde(default)]
    pub phase: Phase,
    /// The belief that it closes the problem.
    pub belief: f64,
    /// Why the starting belief is what it is.
    #[serde(default)]
    pub prior: String,
    /// Where it stands.
    #[serde(default)]
    pub status: CandidateStatus,
    /// Where its evidence lives.
    #[serde(default)]
    pub evidence: Vec<Evidence>,
    /// Its verdicts, oldest first; never written in a document.
    #[serde(default)]
    pub history: Vec<Outcome>,
    /// Its belief revisions, oldest first; never written in a document.
    #[serde(default)]
    pub revisions: Vec<Revision>,
}

/// A problem that limits the model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Problem {
    /// Its id.
    pub id: String,
    /// What limits the model.
    pub statement: String,
    /// The metric a verdict on it is measured in; the parameters name its
    /// target.
    pub metric: String,
    /// Whether it still limits the model.
    #[serde(default)]
    pub status: ProblemStatus,
    /// Experiments run against it: those before the state existed, as the
    /// document says, plus every verdict since (`N` of the priority).
    #[serde(default)]
    pub attempts: u32,
    /// Where its evidence lives.
    #[serde(default)]
    pub evidence: Vec<Evidence>,
    /// The candidates against it.
    #[serde(default)]
    pub candidates: Vec<Candidate>,
    /// Its status revisions, oldest first; never written in a document.
    #[serde(default)]
    pub revisions: Vec<Revision>,
}

/// A document that seeds or extends a problem state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Document {
    /// [`STATE_SCHEMA`].
    pub schema: String,
    /// The parameters from this document on; the state's stay when absent
    /// (the defaults for a seed).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Parameters>,
    /// New problems, or a known problem repeated (same statement and metric)
    /// with new candidates and evidence links.
    #[serde(default)]
    pub problems: Vec<Problem>,
}

/// One change to a problem state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case", deny_unknown_fields)]
pub enum Event {
    /// A document: the first seeds the state, later ones extend it.
    Add {
        /// The document.
        document: Document,
    },
    /// One experiment's verdict on one candidate.
    Verdict {
        /// The candidate.
        candidate: String,
        /// The experiment.
        experiment: String,
        /// The verdict.
        verdict: Verdict,
        /// The effect it rests on, when measured.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        effect: Option<Effect>,
        /// Where it is recorded.
        #[serde(default)]
        evidence: Vec<Evidence>,
        /// Why.
        reason: String,
    },
    /// A coordinator sets a candidate's belief.
    Belief {
        /// The candidate.
        candidate: String,
        /// The new belief.
        belief: f64,
        /// Why.
        reason: String,
    },
    /// A coordinator resolves or reopens a problem.
    Status {
        /// The problem.
        problem: String,
        /// The new status.
        status: ProblemStatus,
        /// Why.
        reason: String,
    },
}

/// An event as the ledger records it: with its schema.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Recorded {
    /// [`EVENT_SCHEMA`].
    pub schema: String,
    /// The event.
    pub event: Event,
}

impl Recorded {
    /// `event` under the current schema.
    #[must_use]
    pub fn new(event: Event) -> Self {
        Self {
            schema: EVENT_SCHEMA.to_string(),
            event,
        }
    }

    /// The event, when its schema is this build's.
    ///
    /// # Errors
    /// Another schema.
    pub fn into_event(self) -> Result<Event, ProblemError> {
        if self.schema == EVENT_SCHEMA {
            Ok(self.event)
        } else {
            invalid(format!(
                "an event of schema {:?}: this build reads {EVENT_SCHEMA}",
                self.schema
            ))
        }
    }
}
