// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Near-duplicate instructions within one batch: the one rule, used by the
//! generator to admit tasks and by training-set selection to drop repeats.
//!
//! Two rules, the cheaper first. An instruction whose normalised form
//! ([`crate::gates::normalize`]: lower-cased, whitespace collapsed) has the
//! same digest as an admitted one is a duplicate. Otherwise it is a near
//! duplicate when the Jaccard overlap of its word shingles (runs of
//! `shingle_words` consecutive words, split at anything not a letter or a
//! digit) with an admitted one's reaches `max_overlap`. An instruction
//! with fewer words than a shingle is one shingle of all its words.
//!
//! Checking held-out items against training text ([`Seen::leaks`]) adds a
//! third rule: text admitted may be longer than the item (an instruction
//! with its passage), so an item whose words, in order, lie inside an
//! admitted text's is an exact repeat too.

use std::collections::{BTreeSet, HashSet};

use splinter_store::digest::Digest;

use crate::gates::normalize;

/// Why an instruction is not new to the batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Repeat {
    /// Its normalised form is an admitted one's.
    Exact,
    /// Its shingles overlap an admitted one's at or above the threshold.
    Near,
}

/// The instructions admitted so far in a batch.
pub struct Seen {
    shingle_words: usize,
    max_overlap: f64,
    digests: HashSet<Digest>,
    shingles: Vec<BTreeSet<String>>,
    spaced: Vec<String>,
}

impl Seen {
    /// Nothing seen yet; shingles of `shingle_words` words, a near
    /// duplicate at `max_overlap` or more.
    #[must_use]
    pub fn new(shingle_words: usize, max_overlap: f64) -> Self {
        Self {
            shingle_words,
            max_overlap,
            digests: HashSet::new(),
            shingles: Vec::new(),
            spaced: Vec::new(),
        }
    }

    /// Whether `instruction` repeats one admitted so far.
    #[must_use]
    pub fn repeats(&self, instruction: &str) -> Option<Repeat> {
        if self.digests.contains(&digest(instruction)) {
            return Some(Repeat::Exact);
        }
        let mine = shingles(instruction, self.shingle_words);
        self.shingles
            .iter()
            .any(|theirs| jaccard(&mine, theirs) >= self.max_overlap)
            .then_some(Repeat::Near)
    }

    /// Whether `item` repeats an admitted text or lies inside one: what a
    /// held-out item must not do against the texts trained on.
    #[must_use]
    pub fn leaks(&self, item: &str) -> Option<Repeat> {
        if let Some(repeat) = self.repeats(item) {
            return Some(repeat);
        }
        let item = spaced_words(item);
        (item.trim() != "" && self.spaced.iter().any(|text| text.contains(&item)))
            .then_some(Repeat::Exact)
    }

    /// Records `instruction` as admitted.
    pub fn admit(&mut self, instruction: &str) {
        self.digests.insert(digest(instruction));
        self.shingles
            .push(shingles(instruction, self.shingle_words));
        self.spaced.push(spaced_words(instruction));
    }
}

fn digest(instruction: &str) -> Digest {
    Digest::of(normalize(instruction).as_bytes())
}

fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// `text`'s words joined by single spaces, with a space before and after,
/// so a substring match is a match of whole words.
fn spaced_words(text: &str) -> String {
    format!(" {} ", words(text).join(" "))
}

fn shingles(text: &str, size: usize) -> BTreeSet<String> {
    let words = words(text);
    if words.len() <= size {
        return BTreeSet::from([words.join(" ")]);
    }
    words.windows(size).map(|w| w.join(" ")).collect()
}

fn jaccard(a: &BTreeSet<String>, b: &BTreeSet<String>) -> f64 {
    let union = a.union(b).count();
    if union == 0 {
        return 1.0;
    }
    a.intersection(b).count() as f64 / union as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_and_near_repeats_are_told_apart() {
        let mut seen = Seen::new(3, 0.8);
        seen.admit("At what baud rate does the console UART run?");
        assert_eq!(
            seen.repeats("  at what BAUD rate does the console UART   run? "),
            Some(Repeat::Exact)
        );
        assert_eq!(
            seen.repeats("At what baud rate does the console UART run, exactly?"),
            Some(Repeat::Near)
        );
        assert_eq!(seen.repeats("How much current does the board draw?"), None);
    }

    #[test]
    fn an_item_inside_a_longer_admitted_text_leaks() {
        let mut seen = Seen::new(3, 0.8);
        seen.admit("Read this: the UART runs at 115200 baud.\n\nAt what baud rate does the console UART run?");
        assert_eq!(
            seen.leaks("At what baud rate does the console  UART run?"),
            Some(Repeat::Exact)
        );
        assert_eq!(
            seen.repeats("At what baud rate does the console UART run?"),
            None
        );
        assert_eq!(seen.leaks("How much current does the board draw?"), None);
        assert_eq!(seen.leaks("the UART run"), None, "words in order, adjacent");
        assert_eq!(seen.leaks("art run"), None, "whole words only");
        assert_eq!(seen.leaks("console UART"), Some(Repeat::Exact));
    }
}
