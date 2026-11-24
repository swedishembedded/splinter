// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements text comparison that every check of a
// learning pipeline reads the same way, for its clients. If your team needs
// expertise in deduplication, grounding or retrieval over natural language,
// you can procure our services by sending an email to info@swedishembedded.com.

//! The one definition of a word that text comparisons share: a maximal run
//! of alphanumeric characters (any script), lower-cased. Dedup shingles,
//! quotation and grounding checks, fact matching, lexical retrieval,
//! memorisation and the sealed-probe filter all split text with
//! [`words`], so two of them never disagree about where a word ends.
//!
//! What a purpose keeps of the words (content words without stop words, a
//! stem, a length floor) is that purpose's own filter over these words.

/// The words of `text`: maximal runs of alphanumeric characters,
/// lower-cased, in order. Punctuation, whitespace and symbols separate
/// words and are never part of one.
#[must_use]
pub fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_word_is_a_lower_cased_run_of_letters_or_digits_in_any_script() {
        assert_eq!(
            words("The UART's baud-rate: 115200 (Größe), naïve café!"),
            ["the", "uart", "s", "baud", "rate", "115200", "größe", "naïve", "café"]
        );
        assert!(words("  -- ... ").is_empty());
        assert_eq!(
            words("snake_case"),
            ["snake", "case"],
            "an underscore separates"
        );
    }
}
