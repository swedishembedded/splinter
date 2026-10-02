// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The decision view: at a state two attempts shared, the action on the
//! passing path.
//!
//! A candidate is a [`RelationKind::RetryOf`] or [`RelationKind::RevisionOf`]
//! relation between two experiences in the corpus. It establishes a
//! preferred action only when one side is decided pass at the minimum
//! strength and the other decided fail at it (otherwise
//! [`Exclusion::NoPreferredPath`]). Both trajectories are rendered with the
//! passing side's student turn (see [`Strip`] and `crate::trajectory`) and
//! compared message by message, ignoring call ids (a fresh run draws fresh
//! ones). Where they first differ, if both took an action there, the
//! passing side's action is the record's supervised reply and the shared
//! messages before it are the context.
//!
//! What this can derive is bounded by what the relations record. The
//! divergence point is the one state both attempts provably reached and
//! acted on differently; after it the passing attempt's states have no
//! recorded failing alternative, so no later state yields a record. When
//! the attempts part at an observation rather than an action, or one is a
//! prefix of the other, no preferred action is established
//! ([`Exclusion::NoDivergence`]). A step labelled bad is not supervised.
//!
//! The record's experiences are the passing, then the failing one.

use splinter_record::annotation::{Label, RelationKind, Strength};
use splinter_record::experience::ExperienceId;

use crate::render::student_turn;
use crate::trajectory::{conversation, same_message};
use crate::{
    require_pass, Corpus, Entry, Exclusion, Objective, Projection, Provenance, RecordBody, Strip,
    View, ViewError,
};

/// The name every decision record carries.
const NAME: &str = "decision";

/// Supervised fine-tuning on the action where a passing attempt parted
/// from a failed one.
#[derive(Clone, Debug)]
pub struct DecisionView {
    min_strength: Strength,
    strip: Strip,
}

impl DecisionView {
    /// The view, accepting pass and fail decisions at `min_strength` or
    /// stronger, showing the student only the instruction.
    #[must_use]
    pub fn new(min_strength: Strength) -> Self {
        Self {
            min_strength,
            strip: Strip::default(),
        }
    }

    /// The same view under `strip`.
    #[must_use]
    pub fn with_strip(self, strip: Strip) -> Self {
        Self { strip, ..self }
    }

    fn failed(&self, entry: &Entry) -> bool {
        entry
            .decision()
            .is_some_and(|d| !d.passed && d.strength >= self.min_strength)
    }

    fn record(
        &self,
        corpus: &Corpus,
        entry: &Entry,
        other: &ExperienceId,
    ) -> Result<(RecordBody, Provenance), Exclusion> {
        let other = corpus.get(other).ok_or(Exclusion::RelatedMissing)?;
        let (winner, loser) =
            if require_pass(entry, self.min_strength).is_ok() && self.failed(other) {
                (entry, other)
            } else if require_pass(other, self.min_strength).is_ok() && self.failed(entry) {
                (other, entry)
            } else {
                return Err(Exclusion::NoPreferredPath);
            };
        let turn = student_turn(winner, &self.strip)?;
        let won =
            conversation(&winner.experience.trajectory, &turn).ok_or(Exclusion::Unrepresentable)?;
        let lost =
            conversation(&loser.experience.trajectory, &turn).ok_or(Exclusion::Unrepresentable)?;
        let parted = won
            .messages
            .iter()
            .zip(&lost.messages)
            .position(|(a, b)| !same_message(a, b))
            .ok_or(Exclusion::NoDivergence)?;
        let action = won
            .actions
            .iter()
            .find(|a| a.index == parted && !a.copied)
            .ok_or(Exclusion::NoDivergence)?;
        if lost.messages.get(parted).map(|m| m.role.as_str()) != Some("assistant") {
            return Err(Exclusion::NoDivergence);
        }
        if winner.labelled(action.step, Label::Bad) {
            return Err(Exclusion::BadStep);
        }
        Ok((
            RecordBody::Chat {
                messages: won.up_to(action),
            },
            Provenance::of(vec![winner.id.clone(), loser.id.clone()]),
        ))
    }
}

impl View for DecisionView {
    fn name(&self) -> &str {
        NAME
    }

    fn objective(&self) -> Objective {
        Objective::Sft
    }

    fn project(&self, corpus: &Corpus) -> Result<Projection, ViewError> {
        let mut projection = Projection::new(
            NAME,
            Objective::Sft,
            Some(self.strip.clone()),
            Some(self.min_strength),
        );
        for entry in corpus.entries() {
            let chained = entry
                .related(RelationKind::RetryOf)
                .chain(entry.related(RelationKind::RevisionOf));
            for other in chained {
                projection.take(self.record(corpus, entry, other));
            }
        }
        Ok(projection)
    }
}
