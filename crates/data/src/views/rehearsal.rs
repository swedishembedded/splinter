// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fine-tuning that rehearses a model's own
// answers so new training does not cost it what it could already do, for
// its clients. If your team needs expertise in continual learning without
// catastrophic forgetting, you can procure our services by sending an email
// to info@swedishembedded.com.

//! The rehearsal view: a model's own answers as supervised records, taken as
//! they were given, so that training on new material can replay what the
//! model already does and not move it off that.
//!
//! An experience qualifies when it has a final answer and no verifier
//! decided that answer wrong: a pass qualifies, and so does an answer no
//! verifier could grade (a general request has no reference), since the
//! point of a rehearsal record is that it is what the model answered, not
//! that it is right. An answer decided fail is left out
//! ([`Exclusion::Failed`]): rehearsing a wrong sum would teach it. The
//! record has the shape of [`super::SftFinal`]'s - the instruction alone,
//! the answer supervised, under the default system prompt - because a
//! rehearsal record shows the model what it sees when it answers as the
//! assistant it already is.

use crate::views::sft_final::final_answer_record;
use crate::{
    Corpus, Entry, Exclusion, Objective, Projection, Provenance, RecordBody, Strip, View, ViewError,
};

/// The name every rehearsal record carries.
pub const NAME: &str = "rehearsal";

/// A model's own answers, supervised as they were given.
#[derive(Clone, Debug, Default)]
pub struct Rehearsal {
    strip: Strip,
}

impl Rehearsal {
    /// The view, showing the student only the instruction.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn record(&self, entry: &Entry) -> Result<(RecordBody, Provenance), Exclusion> {
        if entry.decision().is_some_and(|decision| !decision.passed) {
            return Err(Exclusion::Failed);
        }
        final_answer_record(entry, &self.strip)
    }
}

impl View for Rehearsal {
    fn name(&self) -> &str {
        NAME
    }

    fn objective(&self) -> Objective {
        Objective::Sft
    }

    fn project(&self, corpus: &Corpus) -> Result<Projection, ViewError> {
        let mut projection = Projection::new(NAME, Objective::Sft, Some(self.strip.clone()), None);
        for entry in corpus.entries() {
            projection.take(self.record(entry));
        }
        Ok(projection)
    }
}
