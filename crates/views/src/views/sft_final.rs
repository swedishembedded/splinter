// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The sft-final view: a passed experience's final answer, supervised.
//!
//! An experience qualifies when its annotations decide pass (under the
//! decision rule of `splinter_store::annotation`: the strongest pass/fail
//! verdicts decide, a conflict among them decides nothing) at or above the
//! view's minimum strength, and it has a final output. It yields one record:
//! the student's turn (see [`Strip`]), not supervised, and the final output
//! as the assistant turn, the only supervised one. The trajectory's
//! intermediate steps stay out of the record. One candidate per experience.

use splinter_store::annotation::Strength;

use crate::render::{message, student_turn};
use crate::{
    require_pass, Corpus, Entry, Exclusion, Objective, Projection, Provenance, RecordBody, Strip,
    View, ViewError,
};

/// The name every sft-final record carries.
const NAME: &str = "sft-final";

/// Supervised fine-tuning on the final answer of a passed experience.
#[derive(Clone, Debug)]
pub struct SftFinal {
    min_strength: Strength,
    strip: Strip,
}

impl SftFinal {
    /// The view, accepting pass decisions at `min_strength` or stronger,
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
        require_pass(entry, self.min_strength)?;
        let answer = entry
            .experience
            .final_output
            .as_deref()
            .ok_or(Exclusion::NoFinalOutput)?;
        let turn = student_turn(entry, &self.strip)?;
        let messages = vec![
            message("user", &turn, false),
            message("assistant", answer, true),
        ];
        Ok((
            RecordBody::Chat { messages },
            Provenance::of(vec![entry.id.clone()]),
        ))
    }
}

impl View for SftFinal {
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
            projection.take(self.record(entry));
        }
        Ok(projection)
    }
}
