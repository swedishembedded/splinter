// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Whether a reference answer is supported by its evidence, decided by
//! code and conservatively: most of the reference's content words, and
//! every number it states, must be found in the evidence text.
//!
//! A content word is a maximal run of letters and digits, lower-cased,
//! that holds a digit or is at least three characters long and not one of
//! [`STOPWORDS`]; a plural `s` is then dropped from one longer than three
//! characters. Support is the share of the reference's distinct content
//! words that the evidence's content words include. Numbers are held to
//! the fact extractor's traceability rule
//! ([`crate::gates::answer_numbers_traceable`]): a number the evidence
//! does not carry is an invented number, however many words match.

use std::collections::BTreeSet;

use crate::gates::answer_numbers_traceable;

/// Words too common to show that a text is supported.
pub const STOPWORDS: &[&str] = &[
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "can", "had", "her", "was",
    "one", "our", "out", "has", "his", "how", "its", "may", "new", "now", "see", "two", "who",
    "did", "get", "let", "put", "say", "she", "too", "use", "with", "that", "this", "from", "they",
    "will", "would", "there", "their", "what", "about", "which", "when", "make", "like", "than",
    "then", "them", "these", "some", "into", "only", "over", "also", "such", "each", "does",
    "been", "were", "have", "more", "most", "very", "just", "where", "while", "being",
];

/// How far a reference is supported by its evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct Support {
    /// The share of the reference's content words found in the evidence;
    /// absent when the reference has none.
    pub share: Option<f64>,
    /// Whether every number the reference states is in the evidence.
    pub numbers_traceable: bool,
}

impl Support {
    /// Whether the reference counts as supported at `min_share`.
    #[must_use]
    pub fn holds(&self, min_share: f64) -> bool {
        self.numbers_traceable && self.share.is_some_and(|share| share >= min_share)
    }
}

/// How far `evidence` supports `reference`.
#[must_use]
pub fn support(reference: &str, evidence: &str) -> Support {
    let wanted = content_words(reference);
    let found = content_words(evidence);
    let share = (!wanted.is_empty())
        .then(|| wanted.intersection(&found).count() as f64 / wanted.len() as f64);
    Support {
        share,
        numbers_traceable: answer_numbers_traceable(reference, evidence),
    }
}

/// The distinct content words of `text`; see the module documentation.
#[must_use]
pub fn content_words(text: &str) -> BTreeSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .map(str::to_lowercase)
        .filter(|word| {
            word.chars().any(|c| c.is_ascii_digit())
                || (word.chars().count() >= 3 && !STOPWORDS.contains(&word.as_str()))
        })
        .map(|word| match word.strip_suffix('s') {
            Some(stem) if word.chars().count() > 3 => stem.to_string(),
            _ => word,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn support_counts_content_words_and_insists_on_numbers() {
        let evidence = "The console UART runs at 115200 baud with eight data bits.";
        assert!(support("115200 baud", evidence).holds(1.0));
        assert!(support("It runs at 115200 Baud.", evidence).holds(0.8));
        let invented = support("9600 baud", evidence);
        assert!(!invented.numbers_traceable);
        assert!(!invented.holds(0.0));
        assert!(!support("parity is odd", evidence).holds(0.5));
        assert_eq!(support("it is", evidence).share, None, "no content words");
    }
}
