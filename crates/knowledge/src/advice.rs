// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements task generation that teaches a model what a
// person advised, grounded in what they wrote, for its clients. If your team
// needs expertise in finding the advice in a person's writings, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Finding the advice in a person's writings: the sections in which a writer
//! tells a correspondent what to do, what to avoid or what to value.
//!
//! Advice is rare in a corpus and a model shown three sections at a time
//! mostly finds none, so the generator is shown only the sections that read
//! as advice. The test is cheap and deterministic: running prose, with at
//! least one phrase writers use when they counsel. It decides what is
//! offered, not what is true; whether an offered section yields a task is the
//! generator model's and the admission gates' call.

use crate::tasks::SourceText;

/// Phrases a writer uses when counselling a correspondent.
const CUES: &[&str] = &[
    "i advise",
    "my advice",
    "i recommend",
    "you should",
    "i would have you",
    "i would recommend",
    "i would suggest",
    "i would counsel",
    "i think it best",
    "i deem it",
    "my counsel",
    "be careful",
    "be sure",
    "beware",
    "never",
    "always",
    "do not",
    "i would not",
    "let me advise",
    "let me urge",
    "i urge",
    "you will do well",
    "avoid",
    "take care",
    "i hope you will",
];

/// The fewest words a section needs to hold a passage of advice.
const MIN_WORDS: usize = 30;

/// How many of the advice phrases `text` contains.
#[must_use]
pub fn advice_cues(text: &str) -> usize {
    let lower = text.to_lowercase();
    CUES.iter().filter(|cue| lower.contains(*cue)).count()
}

/// Whether `text` is running prose and not an index entry, a table or a
/// list: few digits, no run of capitals, no `TITLE--` entries or footnote
/// brackets.
#[must_use]
pub fn is_prose(text: &str) -> bool {
    let chars = text.chars().count().max(1);
    let digits = text.chars().filter(char::is_ascii_digit).count();
    let capitals = text
        .split_whitespace()
        .filter(|w| {
            w.len() > 3
                && w.chars().all(|c| !c.is_lowercase())
                && w.chars().any(char::is_alphabetic)
        })
        .count();
    digits * 100 / chars < 2
        && text.matches("--").count() < 3
        && capitals < 3
        && !text.contains('[')
}

/// Whether the section `text` reads as advice.
#[must_use]
pub fn reads_as_advice(text: &str) -> bool {
    text.split_whitespace().count() >= MIN_WORDS && is_prose(text) && advice_cues(text) >= 1
}

/// The positions of the sections of `source` that read as advice, in order.
#[must_use]
pub fn advice_sections(source: &SourceText) -> Vec<usize> {
    (0..source.sections().len())
        .filter(|&at| source.section_text(at).is_some_and(reads_as_advice))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advice_is_prose_with_a_counselling_phrase() {
        let advice = "I advise you to fix a habit of study every morning before you do anything else, and never let a day pass without reading something of history or ethics, for what is not fixed by the pen is lost.";
        assert!(reads_as_advice(advice));
        let business = "I have received your favour of the tenth and enclose the bill of lading for the hogsheads of tobacco shipped on the brig Eliza within the month.";
        assert!(!reads_as_advice(business));
        assert!(
            !reads_as_advice("Never mind."),
            "too short to hold a passage"
        );
    }

    #[test]
    fn an_index_entry_is_not_advice_however_many_cues_it_has() {
        let index = "AMERICA, U. STATES OF--Imperfections of Articles of Confederation, 78. A New Constitution for, necessary, 78. Views of U. States prevalent in Europe, 407, 413. Never to be forgotten, always, 423. Do not, 427.";
        assert!(advice_cues(index) >= 2);
        assert!(!reads_as_advice(index));
    }
}
