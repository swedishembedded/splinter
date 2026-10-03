// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verifiers that grade a model's answers
// against source material without trusting the model, for its clients. If
// your team needs expertise in evidence-based evaluation of language models,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! Grading by code, with no model involved: whether an answer names the right
//! person, year or work, and whether the passages it quotes exist.
//!
//! A model's own claim to know something is not evidence. These checks read
//! the answer text and compare it with a reference the model never saw.

use regex::Regex;

use crate::corpus::words;

/// The answer a model gave, after its reasoning: the visible reply when it
/// closed its reasoning block; when it finished without closing it, everything
/// it wrote (a model fine-tuned on direct answers answers inside the block it
/// was handed open); and nothing when it ran out of tokens still reasoning,
/// because its thoughts name candidates it may go on to reject.
#[must_use]
pub fn final_answer(thinking: &str, text: &str, truncated: bool) -> Option<String> {
    if !text.trim().is_empty() {
        return Some(text.trim().to_string());
    }
    (!truncated && !thinking.trim().is_empty()).then(|| thinking.trim().to_string())
}

/// Whether `answer` contains `needle` as whole words, ignoring case.
#[must_use]
pub fn mentions(answer: &str, needle: &str) -> bool {
    let a = words(answer);
    let n = words(needle);
    !n.is_empty() && a.windows(n.len()).any(|w| w == n.as_slice())
}

/// An answer to "who was it written to?": it names the surname.
#[must_use]
pub fn recipient_ok(answer: &str, surname: &str) -> bool {
    mentions(answer, surname)
}

/// An answer to "in what year?": it states the year, and no other year of the
/// era, so a list of guesses does not pass.
#[must_use]
pub fn year_ok(answer: &str, year: u16) -> bool {
    #[allow(clippy::expect_used)]
    let era = Regex::new(r"\b1[5-9]\d\d\b").expect("a constant pattern");
    let years: Vec<&str> = era.find_iter(answer).map(|m| m.as_str()).collect();
    let wanted = year.to_string();
    years.contains(&wanted.as_str()) && years.iter().all(|y| *y == wanted)
}

/// An answer to "which work is this from?": it names the work, and no other
/// work of the closed set.
#[must_use]
pub fn work_ok(answer: &str, accepted: &[&str], others: &[&str]) -> bool {
    accepted.iter().any(|a| mentions(answer, a)) && !others.iter().any(|o| mentions(answer, o))
}

/// The passages of `answer` set in quotation marks that run at least
/// `min_words` words: what the model presents as his own words.
#[must_use]
pub fn quotations(answer: &str, min_words: usize) -> Vec<String> {
    #[allow(clippy::expect_used)]
    let quoted = Regex::new("[\"\u{201c}]([^\"\u{201c}\u{201d}]{20,})[\"\u{201d}]")
        .expect("a constant pattern");
    quoted
        .captures_iter(answer)
        .filter_map(|c| c.get(1).map(|m| m.as_str().to_string()))
        .filter(|q| words(q).len() >= min_words)
        .collect()
}

/// The corpus a quotation is looked up in: every text, as one run of
/// normalised words, so a quoted passage is found whichever edition prints it
/// and however its lines were broken.
pub struct QuoteIndex {
    haystack: String,
}

impl QuoteIndex {
    /// An index over `texts`.
    #[must_use]
    pub fn new<'a>(texts: impl IntoIterator<Item = &'a str>) -> Self {
        let mut haystack = String::from(" ");
        for text in texts {
            haystack.push_str(&words(text).join(" "));
            haystack.push_str(" | ");
        }
        Self { haystack }
    }

    /// Whether `quotation` occurs verbatim (words, not punctuation) in the corpus.
    #[must_use]
    pub fn contains(&self, quotation: &str) -> bool {
        let needle = words(quotation).join(" ");
        !needle.is_empty() && self.is_word_aligned(&needle)
    }

    fn is_word_aligned(&self, needle: &str) -> bool {
        self.haystack.match_indices(needle).any(|(at, _)| {
            let before = self.haystack[..at].chars().next_back();
            let after = self.haystack[at + needle.len()..].chars().next();
            before.is_none_or(|c| c == ' ') && after.is_none_or(|c| c == ' ')
        })
    }
}

/// How many of `answer`'s quotations are not in the corpus, and how many
/// there are: the fabricated-quotation count a persona model is held to.
#[must_use]
pub fn fabricated(answer: &str, index: &QuoteIndex) -> (usize, usize) {
    let quotes = quotations(answer, 8);
    let missing = quotes.iter().filter(|q| !index.contains(q)).count();
    (missing, quotes.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_surname_must_be_named_as_a_whole_word() {
        assert!(recipient_ok("I wrote this to James Madison.", "Madison"));
        assert!(recipient_ok("It went to madison, I believe.", "Madison"));
        assert!(!recipient_ok("To Madisonian Society members.", "Madison"));
        assert!(!recipient_ok("I do not know.", "Madison"));
    }

    #[test]
    fn a_year_must_be_the_only_year_stated() {
        assert!(year_ok("I wrote it in 1789, from Paris.", 1789));
        assert!(
            !year_ok("Perhaps 1788 or 1789.", 1789),
            "a list of guesses fails"
        );
        assert!(!year_ok("In 1790.", 1789));
        assert!(!year_ok("Some years ago.", 1789));
    }

    #[test]
    fn a_work_must_be_named_and_no_rival_work() {
        let accepted = ["common sense"];
        let others = ["federalist", "leviathan"];
        assert!(work_ok(
            "That is from Common Sense by Paine.",
            &accepted,
            &others
        ));
        assert!(!work_ok(
            "Either Common Sense or the Federalist.",
            &accepted,
            &others
        ));
        assert!(!work_ok("The Leviathan.", &accepted, &others));
    }

    #[test]
    fn the_answer_is_the_visible_reply_else_a_finished_block_else_nothing() {
        assert_eq!(
            final_answer("thinking text", "  the reply ", false).as_deref(),
            Some("the reply")
        );
        assert_eq!(
            final_answer("thinking", "the reply", true).as_deref(),
            Some("the reply"),
            "a closed block then cut off"
        );
        assert_eq!(
            final_answer(" answered inside the block ", "  ", false).as_deref(),
            Some("answered inside the block"),
            "finished without closing the block"
        );
        assert_eq!(
            final_answer("still weighing Madison or Jay", "", true),
            None,
            "ran out of tokens thinking"
        );
        assert_eq!(final_answer("", "", false), None);
    }

    #[test]
    fn a_quotation_is_found_in_the_corpus_whatever_its_line_breaks_or_punctuation() {
        let index = QuoteIndex::new([
            "We hold these truths to be self-evident,\nthat all men are created equal, that they\nare endowed by their Creator with certain unalienable rights.",
        ]);
        let real = "I wrote \"We hold these truths to be self-evident, that all men are created equal\" and meant it.";
        let invented = "I said \"liberty is the first gift of nature and the last concern of a republic\" often.";
        assert_eq!(fabricated(real, &index), (0, 1));
        assert_eq!(fabricated(invented, &index), (1, 1));
        assert_eq!(fabricated("no quotation at all", &index), (0, 0));
        assert_eq!(
            fabricated("a short \"words only here\" quote is not counted", &index),
            (0, 0),
            "under the minimum length a phrase is not a claim to a passage"
        );
    }

    #[test]
    fn a_quotation_must_match_whole_words() {
        let index =
            QuoteIndex::new(["the pursuit of happiness is a right of every citizen of the union"]);
        assert!(index.contains("pursuit of happiness is a right of every citizen"));
        assert!(!index.contains("pursuit of happiness is a righ"));
    }
}
