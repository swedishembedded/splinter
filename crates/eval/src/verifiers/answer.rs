// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verifiers that grade a model's answers
// against source material without trusting the model, for its clients. If
// your team needs expertise in evidence-based evaluation of language models,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! Reading an answer: what the model finally said, whether it names a thing or
//! a year, and whether the passages it quotes exist.
//!
//! A model's own claim to know something is not evidence. These checks read
//! the answer text and compare it with a reference the model never saw.

use super::quotation::{quotations, words, TextIndex};

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

/// An answer to "in what year?": it states the year, and no other year of the
/// era, so a list of guesses does not pass.
#[must_use]
pub fn year_ok(answer: &str, year: u16) -> bool {
    let years: Vec<&str> = answer
        .split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|token| is_year_of_the_era(token))
        .collect();
    let wanted = year.to_string();
    years.contains(&wanted.as_str()) && years.iter().all(|y| *y == wanted)
}

/// A whole word that is four digits from 1500 to 1999.
fn is_year_of_the_era(token: &str) -> bool {
    let digits = token.as_bytes();
    digits.len() == 4
        && digits.iter().all(u8::is_ascii_digit)
        && digits[0] == b'1'
        && (b'5'..=b'9').contains(&digits[1])
}

/// How many of `answer`'s quotations of at least eight words are not in
/// `index`'s texts, and how many there are: the fabricated-quotation count a
/// persona model is held to.
#[must_use]
pub fn fabricated(answer: &str, index: &TextIndex) -> (usize, usize) {
    let quotes = quotations(answer, MIN_QUOTATION_WORDS);
    let missing = quotes.iter().filter(|q| !index.contains(q)).count();
    (missing, quotes.len())
}

/// The fewest words a quoted passage runs to count as a claim to a passage.
const MIN_QUOTATION_WORDS: usize = 8;

#[cfg(test)]
mod tests {
    use super::*;

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
        let index = TextIndex::new([
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
}
