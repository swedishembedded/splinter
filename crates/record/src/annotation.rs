// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements grading records that judge agent work
// without rewriting it, for its clients. If your team needs expertise in
// reward modelling or verifier design, you can procure our services by
// sending an email to info@swedishembedded.com.

//! What is said about an experience after the fact: verdicts, step labels
//! and relations to other experiences.
//!
//! Annotations are append-only and never rewrite the experience they
//! describe, so re-grading is a new annotation, and every view derived from
//! them can be recomputed from the log.
//!
//! # The decision rule
//!
//! Reward is derived, never stored. [`decide`] reads the verdicts:
//!
//! 1. abstentions decide nothing and are ignored;
//! 2. among the remaining pass/fail verdicts, only those at the strongest
//!    [`Strength`] present count (`Executable > Formal > Consistency >
//!    Judged`): a formal check outranks any number of judged opinions;
//! 3. if those all agree, that is the decision; if they conflict, there is
//!    no decision - two equally strong graders disagreeing is not evidence
//!    either way.
//!
//! [`reward`] is the decision as a number: pass `1.0`, fail `0.0`, no
//! decision `None` (unmeasured is not zero).

use serde::{Deserialize, Serialize};

use crate::experience::ExperienceId;

/// What produced an annotation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Producer {
    /// The producer's name, e.g. `splinter-lab/denoise-formal`.
    pub name: String,
    /// Its version, so a changed grader is distinguishable from the old one.
    pub version: String,
}

/// One statement about one experience.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Annotation {
    /// The experience it is about.
    pub experience: ExperienceId,
    /// Who said it.
    pub producer: Producer,
    /// What was said.
    pub body: AnnotationBody,
}

/// What an annotation says.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnnotationBody {
    /// A judgement of the whole experience.
    Verdict {
        /// Pass, fail, or no judgement.
        outcome: Outcome,
        /// How the judgement was reached.
        strength: Strength,
        /// What the judgement rests on, as the producer records it.
        evidence: serde_json::Value,
    },
    /// A judgement of one trajectory step.
    StepLabel {
        /// The ATIF `step_id` of the labelled step.
        step: u64,
        /// The label.
        label: Label,
        /// What the label rests on.
        evidence: serde_json::Value,
    },
    /// How this experience relates to another.
    Relation {
        /// The relation, read as "this experience <kind> `other`".
        kind: RelationKind,
        /// The other experience.
        other: ExperienceId,
    },
}

/// A verdict's outcome.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    /// The work is correct.
    Pass,
    /// The work is wrong.
    Fail,
    /// The grader could not judge it.
    Abstain,
}

/// How a verdict was reached, weakest first: the derived ordering is the
/// strength ordering.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Strength {
    /// A model's or person's opinion.
    Judged,
    /// Agreement between independent attempts.
    Consistency,
    /// A formal comparison against a reference.
    Formal,
    /// The work was executed and checked.
    Executable,
}

/// A step label.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Label {
    /// The step helped.
    Good,
    /// The step hurt.
    Bad,
    /// Neither.
    Neutral,
}

/// How one experience relates to another.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationKind {
    /// This experience is preferred over the other.
    PreferredOver,
    /// This experience retries the other.
    RetryOf,
    /// This experience critiques the other.
    CritiqueOf,
    /// This experience revises the other.
    RevisionOf,
    /// This experience is a variant of the other. (A task set records the
    /// same relation between tasks, which are not experiences, on its
    /// entries.)
    VariantOf,
}

/// The verdict the annotations add up to under the module's decision rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Decision {
    /// `true` for pass, `false` for fail.
    pub passed: bool,
    /// The strength of the verdicts that decided it.
    pub strength: Strength,
}

/// The decision `notes` add up to; `None` when there is no pass/fail
/// verdict, or the strongest ones conflict.
#[must_use]
pub fn decide(notes: &[Annotation]) -> Option<Decision> {
    let verdicts = notes.iter().filter_map(|note| match &note.body {
        AnnotationBody::Verdict {
            outcome: Outcome::Pass,
            strength,
            ..
        } => Some((true, *strength)),
        AnnotationBody::Verdict {
            outcome: Outcome::Fail,
            strength,
            ..
        } => Some((false, *strength)),
        _ => None,
    });
    let strongest = verdicts.clone().map(|(_, s)| s).max()?;
    let mut deciding = verdicts.filter(|(_, s)| *s == strongest).map(|(p, _)| p);
    let passed = deciding.next()?;
    if deciding.any(|p| p != passed) {
        return None;
    }
    Some(Decision {
        passed,
        strength: strongest,
    })
}

/// The reward `notes` derive: `1.0` for a pass decision, `0.0` for a fail,
/// `None` when [`decide`] reaches no decision.
#[must_use]
pub fn reward(notes: &[Annotation]) -> Option<f64> {
    decide(notes).map(|d| if d.passed { 1.0 } else { 0.0 })
}
