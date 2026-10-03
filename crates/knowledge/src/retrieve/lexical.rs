// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements retrieval over a person's writing that
// finds the passages a task or a principle bears on, for its clients. If
// your team needs expertise in hybrid lexical and semantic search over a
// large corpus, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Lexical ranking: BM25 over the passages' words.

use std::collections::HashMap;

use super::{Hit, Passage};

/// How fast repeating a word stops adding to a score.
const K1: f64 = 1.2;

/// How much a passage's length counts against it.
const B: f64 = 0.75;

/// The lower-case alphanumeric words of `text`.
fn tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// BM25 over a fixed set of passages.
pub struct Bm25 {
    /// Per passage, how often each word occurs in it, and its length.
    frequencies: Vec<(HashMap<String, u32>, usize)>,
    /// In how many passages each word occurs.
    holders: HashMap<String, usize>,
    average_length: f64,
}

impl Bm25 {
    /// An index over `passages`; a hit's `passage` is an index into them.
    #[must_use]
    pub fn new(passages: &[Passage]) -> Self {
        let mut holders: HashMap<String, usize> = HashMap::new();
        let mut frequencies = Vec::with_capacity(passages.len());
        for passage in passages {
            let words = tokens(&passage.text);
            let mut counts: HashMap<String, u32> = HashMap::new();
            for word in &words {
                *counts.entry(word.clone()).or_default() += 1;
            }
            for word in counts.keys() {
                *holders.entry(word.clone()).or_default() += 1;
            }
            frequencies.push((counts, words.len()));
        }
        let total: usize = frequencies.iter().map(|(_, n)| n).sum();
        Self {
            average_length: if frequencies.is_empty() {
                0.0
            } else {
                total as f64 / frequencies.len() as f64
            },
            frequencies,
            holders,
        }
    }

    /// The `k` passages that best match the words of `query`, best first;
    /// passages sharing no word with it are not hits.
    #[must_use]
    pub fn search(&self, query: &str, k: usize) -> Vec<Hit> {
        let mut terms = tokens(query);
        terms.sort();
        terms.dedup();
        let n = self.frequencies.len() as f64;
        let mut hits: Vec<Hit> = self
            .frequencies
            .iter()
            .enumerate()
            .filter_map(|(passage, (counts, length))| {
                let score: f64 = terms
                    .iter()
                    .filter_map(|term| {
                        let count = f64::from(*counts.get(term)?);
                        let holders = *self.holders.get(term)? as f64;
                        let idf = ((n - holders + 0.5) / (holders + 0.5) + 1.0).ln();
                        let norm = 1.0 - B + B * (*length as f64 / self.average_length);
                        Some(idf * count * (K1 + 1.0) / (count + K1 * norm))
                    })
                    .sum();
                (score > 0.0).then_some(Hit { passage, score })
            })
            .collect();
        hits.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.passage.cmp(&b.passage)));
        hits.truncate(k);
        hits
    }
}
