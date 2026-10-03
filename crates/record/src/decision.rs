// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements grading records that judge agent work
// without rewriting it, for its clients. If your team needs expertise in
// reward modelling or verifier design, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The decision rule: what an experience's verdicts add up to.
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
//!
//! The rule is the experience database's own, applied to Splinter's
//! annotations, so the store and every query over it can never disagree
//! about which verdict decides.

use splinter_core::annotation::{Annotation, AnnotationBody, Outcome, RelationKind, Strength};
use splinter_expdb::analyze::resolve_verdicts;
use splinter_expdb::model::{Rel, Ruling, Verdict};

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
    let verdicts: Vec<Verdict> = notes
        .iter()
        .filter_map(|note| match &note.body {
            AnnotationBody::Verdict {
                outcome, strength, ..
            } => Some(Verdict::new(
                match outcome {
                    Outcome::Pass => Ruling::Pass,
                    Outcome::Fail => Ruling::Fail,
                    Outcome::Abstain => Ruling::Abstain,
                },
                strength.rank(),
            )),
            _ => None,
        })
        .collect();
    let resolved = resolve_verdicts(&verdicts)?;
    Some(Decision {
        passed: resolved.passed,
        strength: Strength::from_rank(resolved.rank)?,
    })
}

/// The reward `notes` derive: `1.0` for a pass decision, `0.0` for a fail,
/// `None` when [`decide`] reaches no decision.
#[must_use]
pub fn reward(notes: &[Annotation]) -> Option<f64> {
    decide(notes).map(|d| if d.passed { 1.0 } else { 0.0 })
}

/// The edge kind the experience database records `kind` as.
#[must_use]
pub fn relation_edge(kind: RelationKind) -> Rel {
    match kind {
        RelationKind::PreferredOver => Rel::PreferredOver,
        RelationKind::RetryOf => Rel::RetryOf,
        RelationKind::CritiqueOf => Rel::CritiqueOf,
        RelationKind::RevisionOf => Rel::RevisionOf,
        RelationKind::VariantOf => Rel::VariantOf,
    }
}
