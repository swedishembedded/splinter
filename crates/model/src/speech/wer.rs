// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements measurement of what speech recognition and
// synthesis lose, for its clients. If your team needs expertise in
// intelligibility metrics for spoken dialogue systems, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Word error rate: how many words of a reference a hypothesis gets wrong.
//!
//! Both texts are first normalised: every character that is neither a letter
//! nor a digit (in any script) becomes a space, letters are lower-cased, and
//! the text is split on whitespace. So case, punctuation and spacing never
//! count as errors, and "don't" is the two words "don" and "t" on both sides.
//! The errors are the fewest word substitutions, insertions and deletions that
//! turn the reference into the hypothesis (a word-level Levenshtein distance).

use splinter_core::text::words;

/// The fewest word substitutions, insertions and deletions that turn
/// `reference` into `hypothesis`.
#[must_use]
pub fn edits(reference: &[String], hypothesis: &[String]) -> usize {
    // One row of the distance table at a time: row i holds the cost of turning
    // the first i reference words into each prefix of the hypothesis.
    let mut row: Vec<usize> = (0..=hypothesis.len()).collect();
    for (i, said) in reference.iter().enumerate() {
        let mut diagonal = row[0];
        row[0] = i + 1;
        for (j, heard) in hypothesis.iter().enumerate() {
            let substitute = diagonal + usize::from(said != heard);
            diagonal = row[j + 1];
            row[j + 1] = substitute.min(row[j + 1] + 1).min(row[j] + 1);
        }
    }
    row[hypothesis.len()]
}

/// Words wrong over words in `reference`, after normalisation. With no
/// reference words the rate is 0 when nothing was heard either and 1 when
/// something was.
#[must_use]
pub fn word_error_rate(reference: &str, hypothesis: &str) -> f32 {
    let (said, heard) = (words(reference), words(hypothesis));
    if said.is_empty() {
        return if heard.is_empty() { 0.0 } else { 1.0 };
    }
    edits(&said, &heard) as f32 / said.len() as f32
}

/// Total word errors over total reference words of `(reference, hypothesis)`
/// pairs, so a long sentence weighs more than a short one. `None` when the
/// references hold no words: a rate that was not measured is not zero.
#[must_use]
pub fn corpus_word_error_rate<'a>(
    pairs: impl IntoIterator<Item = (&'a str, &'a str)>,
) -> Option<f32> {
    let (mut errors, mut spoken) = (0usize, 0usize);
    for (reference, hypothesis) in pairs {
        let said = words(reference);
        errors += edits(&said, &words(hypothesis));
        spoken += said.len();
    }
    (spoken > 0).then(|| errors as f32 / spoken as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn case_punctuation_and_spacing_are_not_errors() {
        assert_eq!(
            words("  Liberty, Sir -- is NOT given!\n"),
            ["liberty", "sir", "is", "not", "given"]
        );
        assert_eq!(word_error_rate("Liberty, sir.", "liberty sir"), 0.0);
    }

    #[test]
    fn each_substitution_insertion_and_deletion_is_one_error() {
        assert_eq!(word_error_rate("the cat sat", "the dog sat"), 1.0 / 3.0);
        assert_eq!(word_error_rate("the cat sat", "the cat"), 1.0 / 3.0);
        assert_eq!(
            word_error_rate("the cat sat", "the cat sat down"),
            1.0 / 3.0
        );
        assert_eq!(
            word_error_rate("a b c d", "b c d e"),
            0.5,
            "a shift is two errors"
        );
        assert_eq!(
            word_error_rate("one two", "three four five six"),
            2.0,
            "can exceed one"
        );
    }

    #[test]
    fn nothing_said_is_all_or_nothing() {
        assert_eq!(word_error_rate("", " ... "), 0.0);
        assert_eq!(word_error_rate("", "spurious"), 1.0);
    }

    #[test]
    fn the_corpus_rate_is_total_errors_over_total_words_and_absent_when_nothing_was_said() {
        // 1 error in 2 words and 1 in 8: the mean of the two rates would be
        // 0.3125, the corpus rate is 2 / 10.
        let pairs = [("a b", "a"), ("a b c d e f g h", "a b c d e f g x")];
        assert_eq!(corpus_word_error_rate(pairs), Some(0.2));
        assert_eq!(corpus_word_error_rate([("", "noise")]), None);
        assert_eq!(corpus_word_error_rate(Vec::<(&str, &str)>::new()), None);
    }
}
