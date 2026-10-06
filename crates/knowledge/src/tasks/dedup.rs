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
//!
//! A repeat with another answer is worse than a repeat: training on both
//! teaches a model two answers to one question. [`contradictions`] finds
//! them among questions that name a subject: two that name the same
//! subject (case and whitespace aside) and ask the same question, by the
//! two rules above, while their references disagree
//! ([`references_agree`]). Two that name no subject contradict only when
//! their instructions are the same, since such an instruction shows all its
//! answer depends on and a near copy of it may well have another answer.

use std::collections::{BTreeSet, HashSet};

use splinter_core::digest::Digest;

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

    /// Whether an admitted instruction's words, in order, lie inside `text`:
    /// the mirror of [`Self::leaks`], for a text that must not carry an
    /// admitted item within it.
    #[must_use]
    pub fn encloses(&self, text: &str) -> bool {
        let text = spaced_words(text);
        self.spaced
            .iter()
            .any(|admitted| admitted.trim() != "" && text.contains(admitted))
    }

    /// Records `instruction` as admitted.
    pub fn admit(&mut self, instruction: &str) {
        self.digests.insert(digest(instruction));
        self.shingles
            .push(shingles(instruction, self.shingle_words));
        self.spaced.push(spaced_words(instruction));
    }
}

/// A question as the contradiction rule compares it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Asked<'a> {
    /// What the student is asked.
    pub instruction: &'a str,
    /// What it is about; `None` for a question of a kind that names none.
    pub subject: Option<&'a str>,
    /// Its reference answer.
    pub reference: &'a str,
}

/// Whether two references state the same answer: equal once lower-cased,
/// whitespace collapsed and closing punctuation dropped, or the words of
/// one, in order, inside the other's (`115200` and `115200 baud`).
#[must_use]
pub fn references_agree(a: &str, b: &str) -> bool {
    let canonical = |text: &str| {
        normalize(text)
            .trim_end_matches(|c: char| c.is_ascii_punctuation())
            .to_string()
    };
    if canonical(a) == canonical(b) {
        return true;
    }
    let (a, b) = (spaced_words(a), spaced_words(b));
    a.trim() != "" && b.trim() != "" && (a.contains(&b) || b.contains(&a))
}

/// Whether `a` and `b` ask the same question of the same subject and
/// disagree on its answer; see the module documentation.
#[must_use]
pub fn contradicts(a: &Asked<'_>, b: &Asked<'_>, shingle_words: usize, max_overlap: f64) -> bool {
    let same_question = match (a.subject, b.subject) {
        (Some(x), Some(y)) => {
            normalize(x) == normalize(y)
                && (digest(a.instruction) == digest(b.instruction)
                    || jaccard(
                        &shingles(a.instruction, shingle_words),
                        &shingles(b.instruction, shingle_words),
                    ) >= max_overlap)
        }
        (None, None) => digest(a.instruction) == digest(b.instruction),
        _ => false,
    };
    same_question && !references_agree(a.reference, b.reference)
}

/// Every pair `(i, j)`, `i < j`, of `items` that contradict each other
/// ([`contradicts`]).
#[must_use]
pub fn contradictions(
    items: &[Asked<'_>],
    shingle_words: usize,
    max_overlap: f64,
) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for (i, a) in items.iter().enumerate() {
        for (j, b) in items.iter().enumerate().skip(i + 1) {
            if contradicts(a, b, shingle_words, max_overlap) {
                pairs.push((i, j));
            }
        }
    }
    pairs
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
    fn the_same_question_of_the_same_subject_with_another_answer_contradicts() {
        let asked = |instruction, subject, reference| Asked {
            instruction,
            subject,
            reference,
        };
        let items = [
            asked(
                "At what baud rate does the Frobnicator console UART run?",
                Some("Frobnicator"),
                "115200 baud",
            ),
            // The same question and subject, another answer.
            asked(
                "At what baud rate does the  frobnicator console UART run?",
                Some("frobnicator"),
                "9600 baud",
            ),
            // The same answer, worded more briefly: no contradiction.
            asked(
                "At what baud rate does the Frobnicator console UART run, exactly?",
                Some("Frobnicator"),
                "115200",
            ),
            // Another subject may have another answer.
            asked(
                "At what baud rate does the Widget console UART run?",
                Some("Widget"),
                "9600 baud",
            ),
            // Questions that name no subject repeat only when identical.
            asked("What does this print?\n\nprint(1 + 2)", None, "3"),
            asked("What does this print?\n\nprint(1 + 3)", None, "4"),
        ];
        assert_eq!(contradictions(&items, 3, 0.8), [(0, 1), (1, 2)]);
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
        let mut short = Seen::new(3, 0.8);
        short.admit("What is the capital of France?");
        assert!(short.encloses("Quick one: what is the capital of France? Answer briefly."));
        assert!(!short.encloses("What is the capital of Spain?"));
    }
}
