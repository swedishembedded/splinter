// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training data built from a writer's own
// text without a model in the loop, for its clients. If your team needs
// expertise in training a model on a person's voice from what they wrote,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! The admission of a description written for a passage: what a request may
//! say of a passage whose words are the answer to it.
//!
//! A request that quotes the passage hands the model its answer; one that
//! says what the passage is about, in other words, teaches it to write that
//! content in the writer's. A model writes the description, so code decides
//! whether it stands: it must be as long as a description and no longer, and
//! it must not carry the passage's own phrasing - no run of
//! [`SHARED_RUN_WORDS`] words in common with it, and no more than
//! [`MAX_COPIED_SHARE`] of its words inside shorter runs
//! ([`COPIED_RUN_WORDS`]) the passage also has. Single words are not held
//! against it: an honest description of a passage names what the passage
//! names.

use std::collections::HashSet;

/// The fewest words a description holds: about 30 tokens.
pub const MIN_WORDS: usize = 20;

/// The most words a description holds: about 80 tokens.
pub const MAX_WORDS: usize = 60;

/// A run of this many words in common with the passage is a quotation.
pub const SHARED_RUN_WORDS: usize = 8;

/// Words of a shorter run of the passage that count as copied phrasing.
pub const COPIED_RUN_WORDS: usize = 4;

/// The most of a description's words that may lie in copied runs.
pub const MAX_COPIED_SHARE: f64 = 0.35;

/// Why a description does not stand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rejection {
    /// Fewer than [`MIN_WORDS`] words.
    TooShort,
    /// More than [`MAX_WORDS`] words.
    TooLong,
    /// A run of [`SHARED_RUN_WORDS`] words is the passage's own.
    Quotes,
    /// More than [`MAX_COPIED_SHARE`] of it is phrasing the passage has.
    Copies,
}

impl std::fmt::Display for Rejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::TooShort => "too_short",
            Self::TooLong => "too_long",
            Self::Quotes => "quotes",
            Self::Copies => "copies",
        })
    }
}

/// The words of `text`, lower-cased, with everything that is not a letter
/// or a digit taken as the space between them.
fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Every run of `n` consecutive words of `words`.
fn runs(words: &[String], n: usize) -> impl Iterator<Item = &[String]> {
    words.windows(n)
}

/// Whether `description` may stand as the account of `passage`.
pub fn admit(description: &str, passage: &str) -> Result<(), Rejection> {
    let said = words(description);
    if said.len() < MIN_WORDS {
        return Err(Rejection::TooShort);
    }
    if said.len() > MAX_WORDS {
        return Err(Rejection::TooLong);
    }
    let target = words(passage);
    let quoted: HashSet<&[String]> = runs(&target, SHARED_RUN_WORDS).collect();
    if runs(&said, SHARED_RUN_WORDS).any(|run| quoted.contains(run)) {
        return Err(Rejection::Quotes);
    }
    let copied: HashSet<&[String]> = runs(&target, COPIED_RUN_WORDS).collect();
    let mut covered = vec![false; said.len()];
    for (start, run) in runs(&said, COPIED_RUN_WORDS).enumerate() {
        if copied.contains(run) {
            covered[start..start + COPIED_RUN_WORDS].fill(true);
        }
    }
    let share = covered.iter().filter(|&&c| c).count() as f64 / said.len() as f64;
    if share > MAX_COPIED_SHARE {
        return Err(Rejection::Copies);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const PASSAGE: &str = "I have to thank you for the transmission of the letters of the \
        committee, and I return them with my opinion that the militia of the several states \
        should be organised under the care of the states themselves, not of the union.";

    /// A description in other words stands, though it names what the
    /// passage names.
    #[test]
    fn a_description_in_other_words_is_admitted() {
        let description = "Thanks a correspondent for sending on the committee's papers and \
            returns them, adding a view that the state militias ought to be organised and \
            looked after by each state rather than by the federal government.";
        assert_eq!(admit(description, PASSAGE), Ok(()));
    }

    /// A description is neither a word or two nor an essay.
    #[test]
    fn a_description_has_the_length_of_one() {
        assert_eq!(
            admit("About the militia.", PASSAGE),
            Err(Rejection::TooShort)
        );
        let essay = "militia ".repeat(MAX_WORDS + 1);
        assert_eq!(admit(&essay, PASSAGE), Err(Rejection::TooLong));
    }

    /// Eight words of the passage, with other case and punctuation, are a
    /// quotation.
    #[test]
    fn eight_words_of_the_passage_are_a_quotation() {
        let description = "A letter returning papers, in which he gives his opinion that the \
            militia of the several States should be organised under the care of a general \
            government after all, a view he would later reverse entirely.";
        assert_eq!(admit(description, PASSAGE), Err(Rejection::Quotes));
    }

    /// A description stitched of the passage's shorter phrases is copying
    /// even when no eight words run on.
    #[test]
    fn a_description_made_of_the_passages_phrases_is_copied() {
        let description = "I have to thank you, in short, for the transmission of the letters, \
            and my opinion is that the militia of the several states, said he, should be \
            organised under the care, I say, of the states themselves, plainly, not of the union, \
            which he held firmly.";
        assert_eq!(admit(description, PASSAGE), Err(Rejection::Copies));
    }
}
