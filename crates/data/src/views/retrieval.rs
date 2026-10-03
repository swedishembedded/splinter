// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The retrieval view: an instruction and the source span it is grounded
//! in, for contrastive training.
//!
//! An experience decided pass at the minimum strength yields one record per
//! evidence span, each span a candidate: the query is the student's turn
//! (see [`Strip`]), the positive is the span's text resolved through the
//! source store. The negatives are what the corpus shows of the same
//! content that this experience is not grounded in: the evidence spans of
//! other experiences over the same content digest that overlap none of its
//! own, resolved, deduplicated, and different from the positive text.
//! Nothing else is invented as a negative, so a record may have none.
//!
//! A span whose source or content the store does not hold is counted as
//! [`Exclusion::SourceMissing`], one that is not UTF-8 text as
//! [`Exclusion::NotText`]; a store that cannot be read is an error.

use splinter_core::annotation::Strength;
use splinter_core::experience::Span;
use splinter_store::sources::SourceStore;

use super::source_text;
use crate::render::student_turn;
use crate::{
    require_pass, Corpus, Entry, Objective, Projection, Provenance, RecordBody, Strip, View,
    ViewError,
};

/// The name every retrieval record carries.
const NAME: &str = "retrieval";

/// Contrastive (query, positive, negatives) triples from grounded
/// experiences.
#[derive(Clone, Debug)]
pub struct Retrieval<'s> {
    min_strength: Strength,
    strip: Strip,
    sources: &'s SourceStore,
}

impl<'s> Retrieval<'s> {
    /// The view over experiences decided pass at `min_strength` or
    /// stronger, resolving spans through `sources`, the query being the
    /// instruction alone.
    #[must_use]
    pub fn new(min_strength: Strength, sources: &'s SourceStore) -> Self {
        Self {
            min_strength,
            strip: Strip::default(),
            sources,
        }
    }

    /// The same view under `strip`.
    #[must_use]
    pub fn with_strip(self, strip: Strip) -> Self {
        Self { strip, ..self }
    }

    fn project_entry(
        &self,
        corpus: &Corpus,
        entry: &Entry,
        projection: &mut Projection,
    ) -> Result<(), ViewError> {
        if let Err(reason) = require_pass(entry, self.min_strength) {
            projection.exclude(reason);
            return Ok(());
        }
        let evidence = &entry.experience.evidence;
        if evidence.is_empty() {
            projection.exclude(crate::Exclusion::NoEvidence);
            return Ok(());
        }
        let query = match student_turn(entry, &self.strip) {
            Ok(query) => query,
            Err(reason) => {
                projection.exclude(reason);
                return Ok(());
            }
        };
        for span in evidence {
            let positive = match source_text(self.sources.read_span(span))? {
                Ok(text) => text,
                Err(reason) => {
                    projection.exclude(reason);
                    continue;
                }
            };
            let mut negatives: Vec<String> = Vec::new();
            for other in unused_spans(corpus, span, evidence) {
                if let Ok(text) = source_text(self.sources.read_span(other))? {
                    if text != positive && !negatives.contains(&text) {
                        negatives.push(text);
                    }
                }
            }
            projection.push(
                RecordBody::Contrastive {
                    query: query.clone(),
                    positive,
                    negatives,
                },
                Provenance {
                    experiences: vec![entry.id.clone()],
                    task: None,
                    sources: vec![span.source.clone()],
                },
            );
        }
        Ok(())
    }
}

/// The corpus's evidence spans over `span`'s content that overlap none of
/// `own`, in corpus order.
fn unused_spans<'c>(
    corpus: &'c Corpus,
    span: &'c Span,
    own: &'c [Span],
) -> impl Iterator<Item = &'c Span> {
    let overlaps = |a: &Span, b: &Span| a.source == b.source && a.start < b.end && b.start < a.end;
    corpus
        .entries()
        .iter()
        .flat_map(|e| &e.experience.evidence)
        .filter(move |other| other.source == span.source)
        .filter(move |other| !own.iter().any(|mine| overlaps(mine, other)))
}

impl View for Retrieval<'_> {
    fn name(&self) -> &str {
        NAME
    }

    fn objective(&self) -> Objective {
        Objective::Contrastive
    }

    fn project(&self, corpus: &Corpus) -> Result<Projection, ViewError> {
        let mut projection = Projection::new(
            NAME,
            Objective::Contrastive,
            Some(self.strip.clone()),
            Some(self.min_strength),
        );
        for entry in corpus.entries() {
            self.project_entry(corpus, entry, &mut projection)?;
        }
        Ok(projection)
    }
}
