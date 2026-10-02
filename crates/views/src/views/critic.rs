// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The critic view: a verified critique, supervised as the reply to the
//! task and the answer it critiques.
//!
//! A candidate is a [`RelationKind::CritiqueOf`] relation, recorded on the
//! critique experience and naming the critiqued one. It yields one record
//! when the critique is itself decided pass at the minimum strength (a
//! verified critique; otherwise [`Exclusion::UnverifiedCritique`]), both
//! have a final output, and the critiqued experience is in the corpus:
//! the user turn is the critiqued experience's student turn (see
//! [`Strip`]) followed by its answer under `Candidate answer:`, and the
//! supervised reply is the critique's final output.
//!
//! The record's experiences are the critique, then the critiqued one.

use splinter_record::annotation::{RelationKind, Strength};
use splinter_record::experience::ExperienceId;

use crate::render::{message, student_turn, with_candidate};
use crate::{
    require_pass, Corpus, Entry, Exclusion, Objective, Projection, Provenance, RecordBody, Strip,
    View, ViewError,
};

/// The name every critic record carries.
const NAME: &str = "critic";

/// Supervised fine-tuning on verified critiques.
#[derive(Clone, Debug)]
pub struct Critic {
    min_strength: Strength,
    strip: Strip,
}

impl Critic {
    /// The view, accepting critiques decided pass at `min_strength` or
    /// stronger, showing the student only the critiqued instruction and
    /// answer.
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

    fn record(
        &self,
        corpus: &Corpus,
        critique: &Entry,
        critiqued: &ExperienceId,
    ) -> Result<(RecordBody, Provenance), Exclusion> {
        require_pass(critique, self.min_strength).map_err(|_| Exclusion::UnverifiedCritique)?;
        let text = critique
            .experience
            .final_output
            .as_deref()
            .ok_or(Exclusion::NoFinalOutput)?;
        let candidate = corpus.get(critiqued).ok_or(Exclusion::RelatedMissing)?;
        let answer = candidate
            .experience
            .final_output
            .as_deref()
            .ok_or(Exclusion::NoFinalOutput)?;
        let turn = student_turn(candidate, &self.strip)?;
        let messages = vec![
            message("user", &with_candidate(&turn, answer), false),
            message("assistant", text, true),
        ];
        Ok((
            RecordBody::Chat { messages },
            Provenance::of(vec![critique.id.clone(), candidate.id.clone()]),
        ))
    }
}

impl View for Critic {
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
        for critique in corpus.entries() {
            for critiqued in critique.related(RelationKind::CritiqueOf) {
                projection.take(self.record(corpus, critique, critiqued));
            }
        }
        Ok(projection)
    }
}
