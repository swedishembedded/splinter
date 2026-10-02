// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The outcome view: a whole trajectory and the reward it earned.
//!
//! Each experience is a candidate. Its reward is the store's derived
//! `reward` (pass `1.0`, fail `0.0`); an experience with no reward - no
//! verdict, or conflicting strongest verdicts - is left out
//! ([`Exclusion::NoReward`]), never written as `0`: unmeasured is not zero.
//! A reward decided below the minimum strength is left out too. The
//! trajectory is rendered as `crate::trajectory` describes, under the
//! student's turn (see [`Strip`]); its actions are the messages marked
//! `train`.

use splinter_record::annotation::{reward, Strength};

use crate::render::student_turn;
use crate::trajectory::conversation;
use crate::{
    Corpus, Entry, Exclusion, Objective, Projection, Provenance, RecordBody, Strip, View, ViewError,
};

/// The name every outcome record carries.
const NAME: &str = "outcome";

/// Whole trajectories with their derived reward.
#[derive(Clone, Debug)]
pub struct OutcomeView {
    min_strength: Strength,
    strip: Strip,
}

impl OutcomeView {
    /// The view, accepting rewards decided at `min_strength` or stronger,
    /// showing the student only the instruction.
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

    fn record(&self, entry: &Entry) -> Result<(RecordBody, Provenance), Exclusion> {
        let earned = reward(&entry.notes).ok_or(Exclusion::NoReward)?;
        // A reward is derived from a decision, so there is one here.
        let decided = entry.decision().ok_or(Exclusion::NoReward)?;
        if decided.strength < self.min_strength {
            return Err(Exclusion::TooWeak);
        }
        let turn = student_turn(entry, &self.strip)?;
        let conversation =
            conversation(&entry.experience.trajectory, &turn).ok_or(Exclusion::Unrepresentable)?;
        let mut messages = conversation.messages;
        let mut acted = false;
        for action in conversation.actions.iter().filter(|a| !a.copied) {
            if let Some(message) = messages.get_mut(action.index) {
                message.train = true;
                acted = true;
            }
        }
        if !acted {
            return Err(Exclusion::NoAction);
        }
        Ok((
            RecordBody::Rewarded {
                messages,
                reward: earned,
            },
            Provenance::of(vec![entry.id.clone()]),
        ))
    }
}

impl View for OutcomeView {
    fn name(&self) -> &str {
        NAME
    }

    fn objective(&self) -> Objective {
        Objective::Reward
    }

    fn project(&self, corpus: &Corpus) -> Result<Projection, ViewError> {
        let mut projection = Projection::new(
            NAME,
            Objective::Reward,
            Some(self.strip.clone()),
            Some(self.min_strength),
        );
        for entry in corpus.entries() {
            projection.take(self.record(entry));
        }
        Ok(projection)
    }
}
