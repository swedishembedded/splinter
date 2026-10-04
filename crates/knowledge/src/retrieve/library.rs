// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements retrieval over a person's writing that
// finds the passages a task or a principle bears on, for its clients. If
// your team needs expertise in hybrid lexical and semantic search over a
// large corpus, you can procure our services by sending an email to
// info@swedishembedded.com.

//! A library of passages searched both ways.
//!
//! A situation put in plain words shares few words with the paragraph it
//! bears on, so meaning ranks first: measured over the paragraphs of a
//! large body of letters, semantic search found the source paragraph at the
//! top far more often than lexical search, and fusing the two with equal
//! weight let the weaker ranking pull the better one down there. Lexical
//! search keeps what meaning blurs - a name, a place, a date - so the words
//! that are in a passage and nowhere near it in meaning fill the tail of the
//! answer, a fixed share of it, and never its head.

use super::{Bm25, Dense, EmbedError, Embedder, Passage};

/// How a library's answer is shared between the two searches: one slot in
/// this many, past the first, is for what only the words find.
const LEXICAL_ONE_SLOT_IN: usize = 4;

/// Passages with the indexes to search them by meaning and by words.
pub struct Library {
    passages: Vec<Passage>,
    bm25: Bm25,
    dense: Dense,
}

impl Library {
    /// The library of `passages`, embedded by `embedder`.
    pub fn new(passages: Vec<Passage>, embedder: &dyn Embedder) -> Result<Self, EmbedError> {
        let dense = Dense::new(&passages, embedder)?;
        let bm25 = Bm25::new(&passages);
        Ok(Self {
            passages,
            bm25,
            dense,
        })
    }

    /// Every passage, in the order the library was made from.
    #[must_use]
    pub fn passages(&self) -> &[Passage] {
        &self.passages
    }

    /// Up to `k` passages bearing on `query`: nearest in meaning first, the
    /// tail given to the best of the passages only the words of `query` find.
    /// `embedder` must be the one the library was made with.
    pub fn find(
        &self,
        query: &str,
        embedder: &dyn Embedder,
        k: usize,
    ) -> Result<Vec<&Passage>, EmbedError> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let slots = (k - 1).div_ceil(LEXICAL_ONE_SLOT_IN);
        let semantic: Vec<usize> = self
            .dense
            .search(query, embedder, k)?
            .into_iter()
            .map(|h| h.passage)
            .collect();
        let head = &semantic[..semantic.len().min(k - slots)];
        let words: Vec<usize> = self
            .bm25
            .search(query, k)
            .into_iter()
            .map(|h| h.passage)
            .filter(|p| !head.contains(p))
            .take(slots)
            .collect();
        let found = semantic
            .iter()
            .filter(|p| !words.contains(p))
            .take(k - words.len())
            .chain(&words);
        Ok(found.map(|&p| &self.passages[p]).collect())
    }
}
