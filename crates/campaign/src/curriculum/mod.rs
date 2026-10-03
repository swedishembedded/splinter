// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements curricula that spend a learner's solver and
// training budget where it still learns, for its clients. If your team needs
// expertise in curriculum design or continual learning, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The curriculum: solver and training budget spent where learning
//! happens.
//!
//! * [`frontier`] - before a task's attempts become training data, the
//!   policy's pass@k on it is measured: k solves, each graded by the task's
//!   own verifiers and stored as an experience. A task is worth training
//!   on when the student fails it at least sometimes and a verified answer
//!   exists: a passing attempt, or a teacher's. A task always solved
//!   carries no signal, and one never solved that no teacher answered
//!   verifiably has nothing to learn from; both are dropped. Every
//!   measurement is recorded, per task: attempts, passes, rate, the
//!   teacher's solves, and the release measured.
//! * [`teacher`] - a task the student never solves closed-book is solved
//!   once more by a teacher, open-book, with the task's grounding material
//!   shown; a verified teacher's answer is how the policy learns what it
//!   does not know.
//! * [`mastery`] - each concept's rolling pass rate per release, from the
//!   student's closed-book solves; `status` shows the weakest.
//! * [`queue`] - concepts the release gate saw forgotten (a retention suite
//!   that dropped), queued for new tasks, which the next `learn` generates.
//! * [`quota`] - a round's training set, near-duplicates removed with the
//!   generator's own rule, capped per concept, task kind and verification
//!   strength.
//!
//! What a task exercises is [`splinter_knowledge::concepts`]' rule: the
//! concepts it declares, else the (source, section) pairs its evidence
//! falls in, else its kind.
//!
//! The curriculum keeps its records under `<root>/curriculum/`:
//!
//! ```text
//! measurements/<hex>.json   one pass@k measurement, content-addressed, written once
//! queue/<hex>.json          one concept queued for generation, named by the concept's digest
//! ```

pub mod frontier;
pub mod mastery;
pub mod queue;
pub mod quota;
pub mod teacher;

use splinter_core::digest::Digest;

use crate::release::ReleaseId;

/// What an experience's provenance records as its policy when the base
/// solved it through a policy alias that pointed at no release.
pub const BASE_POLICY: &str = "base";

/// What an experience's provenance records as its policy when a policy
/// alias solved it: the release the alias pointed at, or [`BASE_POLICY`].
#[must_use]
pub fn policy_label(release: Option<&ReleaseId>) -> String {
    release.map_or_else(|| BASE_POLICY.to_string(), ToString::to_string)
}

/// The release a [`policy_label`] names: `Some(None)` for the base,
/// `Some(Some(id))` for a release, `None` for anything else (an experience
/// a policy alias did not solve).
#[must_use]
pub fn release_of_label(label: &str) -> Option<Option<ReleaseId>> {
    if label == BASE_POLICY {
        return Some(None);
    }
    Digest::parse(label).ok().map(|d| Some(ReleaseId(d)))
}
