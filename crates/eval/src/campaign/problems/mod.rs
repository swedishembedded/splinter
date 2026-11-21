// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for choosing the next training
// experiment from the evidence already paid for, for its clients. If your team
// needs expertise in planning model-improvement campaigns you can procure our
// services by sending an email to info@swedishembedded.com.

//! A campaign's problem state: the problems that limit the model, each with
//! its evidence, and the candidate interventions against each with a belief
//! in `[0, 1]`, revised by every verdict.
//!
//! The state is never edited. It is the fold of an append-only list of
//! [`Event`]s ([`ProblemState::fold`]): a document that seeds or extends it,
//! one experiment's verdict on one candidate, or a coordinator's revision of a
//! belief or of a problem's status. Folding the same events gives the same
//! state, byte for byte ([`ProblemState::digest`]).
//!
//! * [`rules`] - the priority `b / (N + 1)`, the belief update by verdict, the
//!   verdict of a measured effect against the problem's target, the verdict of
//!   the promotion gate, and the ranked proposals.
//! * [`variables`] - whether an arm's configuration differs from the
//!   champion's in the one declared variable, and whether an engineering
//!   change stays inside the paths it may touch.
//!
//! No files and no clock: the campaign in `splinter-pipelines` stores the
//! events and reads the configurations.

pub mod rules;
mod schema;
pub mod variables;

#[cfg(test)]
mod tests;

use serde::{Deserialize, Serialize};
use splinter_core::digest::{digest_of, Digest};

pub use rules::{
    classify_effect, classify_promotion, priority, updated_belief, Proposal, DEFAULT_ALPHA,
    DEFAULT_BELIEF_DOWN, DEFAULT_BELIEF_UP,
};
pub use schema::{
    Candidate, CandidateStatus, Document, Effect, Event, Evidence, Outcome, Parameters, Phase,
    Problem, ProblemStatus, Recorded, Revision, Verdict,
};
pub use variables::{changed, firewall, leaves, single_variable, under, VariableError};

/// The schema of a problem-state document.
pub const STATE_SCHEMA: &str = "splinter.problem-state/1";

/// The schema of a recorded event.
pub const EVENT_SCHEMA: &str = "splinter.problem-event/1";

/// The kinds an evidence link may name, as `<kind>:<id>`.
pub const EVIDENCE_KINDS: [&str; 9] = [
    "knowledge",
    "ledger",
    "result",
    "diagnosis",
    "commit",
    "experience",
    "dataset",
    "campaign",
    "paper",
];

/// The configuration paths an engineering change may touch when a document
/// names none: how many trials run at once and which `harbor` program runs
/// them, the two campaign settings that change how fast a generation runs and
/// nothing it learns or is measured on.
pub const DEFAULT_ENGINEERING: [&str; 2] = ["practice.jobs", "practice.harbor"];

/// Why a document or an event is refused.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum ProblemError {
    /// A document, parameter or event field is out of range, named.
    #[error("{0}")]
    Invalid(String),
    /// An event other than a document came before any document.
    #[error("no problem state has been seeded: the first event adds a document")]
    Unseeded,
    /// No candidate has this id.
    #[error("no candidate {0:?} in the problem state")]
    UnknownCandidate(String),
    /// No problem has this id.
    #[error("no problem {0:?} in the problem state")]
    UnknownProblem(String),
    /// The candidate already has a verdict from this experiment.
    #[error("candidate {candidate:?} already has a verdict from experiment {experiment:?}: one experiment, one verdict")]
    DuplicateExperiment {
        /// The candidate.
        candidate: String,
        /// The experiment.
        experiment: String,
    },
    /// The candidate was adopted by an earlier verdict.
    #[error("candidate {0:?} is adopted: it is part of the champion and no longer a hypothesis")]
    Adopted(String),
}

/// The fold of every event so far.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProblemState {
    /// The parameters in force.
    pub parameters: Parameters,
    /// Every problem, in the order added.
    pub problems: Vec<Problem>,
}

fn invalid<T>(why: String) -> Result<T, ProblemError> {
    Err(ProblemError::Invalid(why))
}

fn valid_id(what: &str, id: &str) -> Result<(), ProblemError> {
    let sound = !id.is_empty()
        && id.len() <= 64
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "-_.@".contains(c));
    if sound {
        Ok(())
    } else {
        invalid(format!(
            "{what} id {id:?}: 1 to 64 letters, digits, `-`, `_`, `.` and `@`"
        ))
    }
}

fn valid_path(what: &str, path: &str) -> Result<(), ProblemError> {
    if !path.is_empty() && path.split('.').all(|s| !s.trim().is_empty()) {
        Ok(())
    } else {
        invalid(format!(
            "{what} {path:?}: a configuration path is dotted names, none empty"
        ))
    }
}

fn valid_belief(what: &str, belief: f64) -> Result<(), ProblemError> {
    if belief.is_finite() && (0.0..=1.0).contains(&belief) {
        Ok(())
    } else {
        invalid(format!("{what}: belief {belief} is not in [0, 1]"))
    }
}

fn reasoned(reason: &str) -> Result<(), ProblemError> {
    if reason.trim().is_empty() {
        invalid("a verdict or revision is recorded with its reason: give one".into())
    } else {
        Ok(())
    }
}

impl ProblemState {
    /// The state `events` fold to; `None` when there are none.
    ///
    /// # Errors
    /// The first event that cannot be applied, as [`Self::fold`].
    pub fn replay<'a>(
        events: impl IntoIterator<Item = &'a Event>,
    ) -> Result<Option<Self>, ProblemError> {
        let mut state = None;
        for event in events {
            state = Some(Self::fold(state, event)?);
        }
        Ok(state)
    }

    /// `state` with `event` applied; a document seeds a state that is
    /// `None`.
    ///
    /// # Errors
    /// An event before any document, or one [`Self::apply`] refuses.
    pub fn fold(state: Option<Self>, event: &Event) -> Result<Self, ProblemError> {
        let mut state = match (state, event) {
            (Some(state), _) => state,
            (None, Event::Add { .. }) => Self {
                parameters: Parameters::default(),
                problems: Vec::new(),
            },
            (None, _) => return Err(ProblemError::Unseeded),
        };
        state.apply(event)?;
        Ok(state)
    }

    /// Applies `event`, or refuses it and changes nothing.
    ///
    /// # Errors
    /// A document out of range; a verdict, belief or status naming what is
    /// not there; a second verdict from one experiment; a verdict on an
    /// adopted candidate; a revision without a reason.
    pub fn apply(&mut self, event: &Event) -> Result<(), ProblemError> {
        let mut next = self.clone();
        match event {
            Event::Add { document } => next.add(document)?,
            Event::Verdict {
                candidate,
                experiment,
                verdict,
                effect,
                evidence,
                reason,
            } => next.verdict(candidate, experiment, *verdict, effect, evidence, reason)?,
            Event::Belief {
                candidate,
                belief,
                reason,
            } => {
                reasoned(reason)?;
                valid_belief(candidate, *belief)?;
                let (_, c) = next.candidate_mut(candidate)?;
                c.revisions.push(Revision {
                    before: c.belief.to_string(),
                    after: belief.to_string(),
                    reason: reason.clone(),
                });
                c.belief = *belief;
            }
            Event::Status {
                problem,
                status,
                reason,
            } => {
                reasoned(reason)?;
                let p = next
                    .problems
                    .iter_mut()
                    .find(|p| p.id == *problem)
                    .ok_or_else(|| ProblemError::UnknownProblem(problem.clone()))?;
                p.revisions.push(Revision {
                    before: status_text(p.status),
                    after: status_text(*status),
                    reason: reason.clone(),
                });
                p.status = *status;
            }
        }
        *self = next;
        Ok(())
    }

    fn add(&mut self, document: &Document) -> Result<(), ProblemError> {
        if document.schema != STATE_SCHEMA {
            return invalid(format!(
                "a problem-state document of schema {:?}: this build reads {STATE_SCHEMA}",
                document.schema
            ));
        }
        if let Some(parameters) = &document.parameters {
            parameters.validate()?;
            self.parameters = parameters.clone();
        }
        for problem in &document.problems {
            self.add_problem(problem)?;
        }
        Ok(())
    }

    fn add_problem(&mut self, problem: &Problem) -> Result<(), ProblemError> {
        valid_id("problem", &problem.id)?;
        if !self.parameters.targets.contains_key(&problem.metric) {
            return invalid(format!(
                "problem {:?}: its metric {:?} has no target in parameters.targets",
                problem.id, problem.metric
            ));
        }
        if !problem.revisions.is_empty() {
            return invalid(format!(
                "problem {:?}: revisions are recorded by events, not written in a document",
                problem.id
            ));
        }
        for evidence in &problem.evidence {
            evidence.validate()?;
        }
        for candidate in &problem.candidates {
            self.validate_candidate(&problem.id, candidate)?;
        }
        let mut seen: Vec<&str> = Vec::new();
        for candidate in &problem.candidates {
            if seen.contains(&candidate.id.as_str()) {
                return invalid(format!("candidate {:?} is listed twice", candidate.id));
            }
            seen.push(&candidate.id);
        }
        if let Some(known) = self.problems.iter_mut().find(|p| p.id == problem.id) {
            if known.statement != problem.statement || known.metric != problem.metric {
                return invalid(format!(
                    "problem {:?} is known with another statement or metric: add a new problem instead",
                    problem.id
                ));
            }
            for evidence in &problem.evidence {
                if !known.evidence.contains(evidence) {
                    known.evidence.push(evidence.clone());
                }
            }
            known.candidates.extend(problem.candidates.iter().cloned());
        } else {
            self.problems.push(problem.clone());
        }
        Ok(())
    }

    fn validate_candidate(&self, problem: &str, candidate: &Candidate) -> Result<(), ProblemError> {
        valid_id("candidate", &candidate.id)?;
        if self.candidate(&candidate.id).is_some() {
            return invalid(format!(
                "candidate {:?} is already in the problem state",
                candidate.id
            ));
        }
        valid_path(
            &format!("candidate {:?} variable", candidate.id),
            &candidate.variable,
        )?;
        valid_belief(&format!("candidate {:?}", candidate.id), candidate.belief)?;
        if candidate.intervention.trim().is_empty() || candidate.statement.trim().is_empty() {
            return invalid(format!(
                "candidate {:?} of problem {problem:?}: name its intervention and state it",
                candidate.id
            ));
        }
        if !candidate.history.is_empty() || !candidate.revisions.is_empty() {
            return invalid(format!(
                "candidate {:?}: verdicts and revisions are recorded by events, not written in a document",
                candidate.id
            ));
        }
        if candidate.phase == Phase::Engineering
            && !self
                .parameters
                .engineering
                .iter()
                .any(|path| under(&candidate.variable, path))
        {
            return invalid(format!(
                "engineering candidate {:?} changes {:?}, which is not an engineering path ({})",
                candidate.id,
                candidate.variable,
                self.parameters.engineering.join(", ")
            ));
        }
        for evidence in &candidate.evidence {
            evidence.validate()?;
        }
        Ok(())
    }

    fn verdict(
        &mut self,
        candidate: &str,
        experiment: &str,
        verdict: Verdict,
        effect: &Option<Effect>,
        evidence: &[Evidence],
        reason: &str,
    ) -> Result<(), ProblemError> {
        reasoned(reason)?;
        valid_id("experiment", experiment)?;
        for link in evidence {
            link.validate()?;
        }
        let parameters = self.parameters.clone();
        let (attempts, c) = self.candidate_mut(candidate)?;
        if c.history.iter().any(|o| o.experiment == experiment) {
            return Err(ProblemError::DuplicateExperiment {
                candidate: candidate.to_string(),
                experiment: experiment.to_string(),
            });
        }
        if c.status == CandidateStatus::Adopted {
            return Err(ProblemError::Adopted(candidate.to_string()));
        }
        let before = c.belief;
        c.belief = updated_belief(before, verdict, &parameters);
        if verdict == Verdict::Supported {
            c.status = CandidateStatus::Adopted;
        }
        c.history.push(Outcome {
            experiment: experiment.to_string(),
            verdict,
            belief_before: before,
            belief_after: c.belief,
            effect: effect.clone(),
            evidence: evidence.to_vec(),
            reason: reason.to_string(),
        });
        *attempts = attempts.saturating_add(1);
        Ok(())
    }

    /// The problem and candidate with id `candidate`.
    #[must_use]
    pub fn candidate(&self, candidate: &str) -> Option<(&Problem, &Candidate)> {
        self.problems.iter().find_map(|p| {
            p.candidates
                .iter()
                .find(|c| c.id == candidate)
                .map(|c| (p, c))
        })
    }

    fn candidate_mut(
        &mut self,
        candidate: &str,
    ) -> Result<(&mut u32, &mut Candidate), ProblemError> {
        self.problems
            .iter_mut()
            .find_map(|p| {
                p.candidates
                    .iter_mut()
                    .find(|c| c.id == candidate)
                    .map(|c| (&mut p.attempts, c))
            })
            .ok_or_else(|| ProblemError::UnknownCandidate(candidate.to_string()))
    }

    /// The target `delta_target` of `problem`'s metric.
    ///
    /// # Errors
    /// A metric the parameters in force name no target for (they were
    /// replaced by a later document).
    pub fn target(&self, problem: &Problem) -> Result<f64, ProblemError> {
        self.parameters
            .targets
            .get(&problem.metric)
            .copied()
            .ok_or_else(|| {
                ProblemError::Invalid(format!(
                    "problem {:?}: its metric {:?} has no target in the parameters in force",
                    problem.id, problem.metric
                ))
            })
    }

    /// The state's identity: the digest of its canonical JSON.
    #[must_use]
    pub fn digest(&self) -> Digest {
        digest_of(self)
    }
}

fn status_text(status: ProblemStatus) -> String {
    match status {
        ProblemStatus::Open => "open",
        ProblemStatus::Resolved => "resolved",
    }
    .to_string()
}
