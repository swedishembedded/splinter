// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The cpt view: the raw text of source parts, for continued pretraining.
//!
//! Every part of every source in the corpus, in corpus order and each
//! source's part order, is a candidate. A part whose content digest was
//! already projected is a [`Exclusion::Duplicate`] - one content shared by
//! many sources is trained on once; content that is not UTF-8 text is
//! [`Exclusion::NotText`], empty content [`Exclusion::Empty`]. There is no
//! student input, so no strip policy, and no verdict is read.

use std::collections::HashSet;

use splinter_record::digest::Digest;
use splinter_record::sources::SourceStore;

use super::source_text;
use crate::{Corpus, Exclusion, Objective, Projection, Provenance, RecordBody, View, ViewError};

/// The name every cpt record carries.
const NAME: &str = "cpt";

/// Raw source text for continued pretraining.
#[derive(Clone, Debug)]
pub struct Cpt<'s> {
    sources: &'s SourceStore,
}

impl<'s> Cpt<'s> {
    /// The view reading source content from `sources`.
    #[must_use]
    pub fn new(sources: &'s SourceStore) -> Self {
        Self { sources }
    }
}

impl View for Cpt<'_> {
    fn name(&self) -> &str {
        NAME
    }

    fn objective(&self) -> Objective {
        Objective::Cpt
    }

    fn project(&self, corpus: &Corpus) -> Result<Projection, ViewError> {
        let mut projection = Projection::new(NAME, Objective::Cpt, None, None);
        let mut seen: HashSet<Digest> = HashSet::new();
        for id in corpus.sources() {
            let source = self.sources.get_source(id)?;
            for part in &source.parts {
                if !seen.insert(part.content.clone()) {
                    projection.exclude(Exclusion::Duplicate);
                    continue;
                }
                let outcome = source_text(self.sources.read_blob(&part.content))?.map(|text| {
                    (
                        RecordBody::Text { text },
                        Provenance {
                            sources: vec![part.content.clone()],
                            ..Provenance::default()
                        },
                    )
                });
                projection.take(outcome);
            }
        }
        Ok(projection)
    }
}
