// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements mechanical grading of answers against the
// specific names, numbers and terms a fact states, for its clients. If your
// team needs expertise in checking what a model says without trusting a
// model, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The keys of a fact: the specific tokens an answer must contain.
//!
//! A key is a name, a number (a date's day and year included) or a quoted
//! term the fact's statement holds and its question does not give away. They
//! are read from the statement by code alone, so the first of the two checks
//! an answer must pass (the second is the calibrated judge) involves no
//! model. Matching is by whole words, ignoring case; a number matches its
//! digits or its word (`40` and `forty`).

use serde::{Deserialize, Serialize};
use splinter_sdk::measure::verifiers::answer::mentions;

/// The most keys a statement may hold: more than this is not one fact.
pub const MAX_KEYS: usize = 5;

/// What kind of token a key is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KeyKind {
    /// A person, place or title, known by the last word of its capitalised run.
    Name,
    /// Digits (commas removed) or a number word.
    Number,
    /// A term the statement puts in quotation marks.
    Term,
}

/// One specific token an answer must contain.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Key {
    /// The token as the statement gives it.
    pub text: String,
    /// What it is.
    pub kind: KeyKind,
}

/// Words that open a sentence without naming anything.
const FUNCTION_WORDS: &[&str] = &[
    "a", "an", "and", "as", "at", "but", "by", "for", "from", "he", "her", "his", "i", "if", "in",
    "is", "it", "its", "of", "on", "or", "our", "she", "that", "the", "their", "there", "these",
    "they", "this", "those", "to", "was", "we", "when", "where", "which", "while", "with", "who",
];

/// Titles that are no name when they stand alone.
const TITLES: &[&str] = &[
    "mr",
    "mrs",
    "dr",
    "sir",
    "lord",
    "colonel",
    "general",
    "president",
    "governor",
    "secretary",
    "judge",
    "captain",
    "major",
    "king",
    "esq",
];

/// Number words and the value each stands for. `one` is left out: it is the
/// pronoun in too many sentences to be a figure.
const NUMBER_WORDS: &[(&str, u32)] = &[
    ("two", 2),
    ("three", 3),
    ("four", 4),
    ("five", 5),
    ("six", 6),
    ("seven", 7),
    ("eight", 8),
    ("nine", 9),
    ("ten", 10),
    ("eleven", 11),
    ("twelve", 12),
    ("thirteen", 13),
    ("fourteen", 14),
    ("fifteen", 15),
    ("sixteen", 16),
    ("seventeen", 17),
    ("eighteen", 18),
    ("nineteen", 19),
    ("twenty", 20),
    ("thirty", 30),
    ("forty", 40),
    ("fifty", 50),
    ("sixty", 60),
    ("seventy", 70),
    ("eighty", 80),
    ("ninety", 90),
    ("hundred", 100),
    ("thousand", 1000),
];

/// The keys of `statement` that `question` does not already give away.
#[must_use]
pub fn extract(statement: &str, question: &str) -> Vec<Key> {
    let (terms, rest) = split_quoted(statement);
    let mut keys: Vec<Key> = terms
        .into_iter()
        .map(|text| Key {
            text,
            kind: KeyKind::Term,
        })
        .collect();
    keys.extend(numbers(&rest).into_iter().map(|text| Key {
        text,
        kind: KeyKind::Number,
    }));
    keys.extend(names(&rest).into_iter().map(|text| Key {
        text,
        kind: KeyKind::Name,
    }));
    let mut seen = std::collections::HashSet::new();
    keys.retain(|k| !given_away(k, question) && seen.insert(k.text.to_lowercase()));
    keys
}

/// Whether `question` already holds `key`.
fn given_away(key: &Key, question: &str) -> bool {
    alternatives(key).iter().any(|a| mentions(question, a))
}

/// The terms in quotation marks in `text`, and `text` without them.
fn split_quoted(text: &str) -> (Vec<String>, String) {
    let mut terms = Vec::new();
    let mut rest = String::new();
    let mut inside: Option<String> = None;
    for c in text.chars() {
        match (&mut inside, c) {
            (None, '"' | '\u{201c}') => inside = Some(String::new()),
            (Some(term), '"' | '\u{201d}') => {
                let term = term.trim().to_string();
                if (1..=8).contains(&term.split_whitespace().count()) {
                    terms.push(term);
                }
                inside = None;
                rest.push(' ');
            }
            (Some(term), c) => term.push(c),
            (None, c) => rest.push(c),
        }
    }
    if let Some(unclosed) = inside {
        rest.push_str(&unclosed);
    }
    (terms, rest)
}

/// Digit figures (commas inside removed) and number words, in order.
fn numbers(text: &str) -> Vec<String> {
    let mut found = Vec::new();
    for token in text.split(|c: char| !(c.is_alphanumeric() || c == ',' || c == '.')) {
        let token = token.trim_matches(|c: char| c == ',' || c == '.');
        if !token.is_empty()
            && token
                .chars()
                .all(|c| c.is_ascii_digit() || c == ',' || c == '.')
        {
            found.push(token.replace(',', ""));
        } else if NUMBER_WORDS
            .iter()
            .any(|(word, _)| token.eq_ignore_ascii_case(word))
        {
            found.push(token.to_lowercase());
        }
    }
    found
}

/// The names `text` holds, each known by the last word of its run.
#[must_use]
pub fn names_of(text: &str) -> Vec<String> {
    names(text)
}

/// Each run of capitalised words, known by its last word; a sentence's first
/// word counts only when it is no function word.
fn names(text: &str) -> Vec<String> {
    let mut found = Vec::new();
    let mut run: Option<String> = None;
    let mut sentence_start = true;
    let flush = |run: &mut Option<String>, found: &mut Vec<String>| {
        if let Some(last) = run.take() {
            if !TITLES.contains(&last.to_lowercase().as_str()) && last.chars().count() > 1 {
                found.push(last);
            }
        }
    };
    for raw in text.split_whitespace() {
        let word = raw.trim_matches(|c: char| !c.is_alphanumeric());
        let word = word.strip_suffix("'s").unwrap_or(word);
        let capitalised = word.chars().next().is_some_and(char::is_uppercase)
            && word.chars().any(char::is_lowercase);
        let function_word = FUNCTION_WORDS.contains(&word.to_lowercase().as_str());
        if capitalised && !(sentence_start && function_word) {
            run = Some(word.to_string());
        } else {
            flush(&mut run, &mut found);
        }
        sentence_start = raw.ends_with(['.', '!', '?']);
        if raw.ends_with([',', ';', ':', '.', '!', '?']) {
            flush(&mut run, &mut found);
        }
    }
    flush(&mut run, &mut found);
    found
}

/// The spellings of `key` an answer may use.
fn alternatives(key: &Key) -> Vec<String> {
    let mut spellings = vec![key.text.clone()];
    if key.kind == KeyKind::Number {
        if let Ok(value) = key.text.parse::<u32>() {
            spellings.extend(
                NUMBER_WORDS
                    .iter()
                    .filter(|(_, v)| *v == value)
                    .map(|(word, _)| (*word).to_string()),
            );
        } else if let Some((_, value)) = NUMBER_WORDS
            .iter()
            .find(|(word, _)| key.text.eq_ignore_ascii_case(word))
        {
            spellings.push(value.to_string());
        }
    }
    spellings
}

/// Whether `answer` holds `key`.
#[must_use]
pub fn present(answer: &str, key: &Key) -> bool {
    alternatives(key).iter().any(|a| mentions(answer, a))
}

/// The keys `answer` does not hold.
#[must_use]
pub fn missing<'a>(answer: &str, keys: &'a [Key]) -> Vec<&'a Key> {
    keys.iter().filter(|k| !present(answer, k)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(keys: &[Key]) -> Vec<&str> {
        keys.iter().map(|k| k.text.as_str()).collect()
    }

    #[test]
    fn keys_are_the_names_numbers_and_terms_the_question_does_not_give_away() {
        let keys = extract(
            "In a letter to Madison on March 4, 1801, Jefferson said the \"Mammoth Cheese\" \
             weighed 1,235 pounds and cost forty dollars.",
            "What did Jefferson say about the cheese in his letter to Madison?",
        );
        let mut got = texts(&keys);
        got.sort_unstable();
        assert_eq!(
            got,
            vec!["1235", "1801", "4", "Mammoth Cheese", "March", "forty"]
        );
        let kind = |text: &str| keys.iter().find(|k| k.text == text).map(|k| k.kind);
        assert_eq!(kind("March"), Some(KeyKind::Name));
        assert_eq!(kind("1801"), Some(KeyKind::Number));
        assert_eq!(kind("Mammoth Cheese"), Some(KeyKind::Term));
    }

    #[test]
    fn a_name_is_its_last_word_and_a_sentence_opener_is_no_name() {
        let keys = extract(
            "The envoy James Monroe sailed to Paris. He met General Lafayette there.",
            "Who sailed?",
        );
        assert_eq!(texts(&keys), vec!["Monroe", "Paris", "Lafayette"]);
    }

    #[test]
    fn an_answer_holds_a_key_as_a_whole_word_in_any_case_and_numbers_in_words_or_digits() {
        let keys = extract("The post paid 40 dollars to Barnes.", "What was paid?");
        assert!(missing("He paid forty dollars to Mr. BARNES.", &keys).is_empty());
        assert_eq!(
            texts(
                &missing("He paid 400 dollars to Barnesville.", &keys)
                    .into_iter()
                    .cloned()
                    .collect::<Vec<_>>()
            ),
            vec!["40", "Barnes"]
        );
    }

    #[test]
    fn a_statement_with_nothing_specific_has_no_keys() {
        assert!(extract("He liked the garden.", "What did he like?").is_empty());
    }
}
