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
//! The rule that turns verdicts into a decision is the store's: it is the
//! experience database's own, and lives beside it.

use serde::{Deserialize, Serialize};

use crate::experience::ExperienceId;

/// What produced an annotation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Producer {
    /// The producer's name, e.g. `splinter-eval/denoise-formal`.
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
    /// A rule the work must not break, such as inventing nothing the source
    /// lacks. It refutes and never establishes: a pass says only that nothing
    /// was found wrong, which does not make an answer right, so it counts as
    /// an abstention ([`Strength::counted`]); a fail refutes whatever else
    /// says, so it outranks every other strength. It is therefore the
    /// strongest in the ordering, and no verdict of it says "right".
    Constraint = 255,
}

impl Strength {
    /// The rank the experience database orders this evidence by.
    #[must_use]
    pub fn rank(self) -> u8 {
        self as u8
    }

    /// The strength that has `rank`, if any.
    #[must_use]
    pub fn from_rank(rank: u8) -> Option<Self> {
        [
            Strength::Judged,
            Strength::Consistency,
            Strength::Formal,
            Strength::Executable,
            Strength::Constraint,
        ]
        .into_iter()
        .find(|s| s.rank() == rank)
    }

    /// The outcome a verdict of this strength counts as in a decision: its
    /// own, except that a constraint that passed counts as an abstention.
    #[must_use]
    pub fn counted(self, outcome: Outcome) -> Outcome {
        match (self, outcome) {
            (Strength::Constraint, Outcome::Pass) => Outcome::Abstain,
            _ => outcome,
        }
    }
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
