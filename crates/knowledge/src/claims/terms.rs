// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements admission gates that keep ungrounded
// claims out of training data, for its clients. If your team needs
// expertise in synthetic data quality or provenance, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The terms of a statement that must be found in the words it rests on.
//!
//! A model that restates what a person said may carry its own knowledge
//! into the restatement: a plausible port, a product name, a date. Three
//! kinds of term are held to the cited words, because they are where such a
//! statement turns false without turning odd:
//!
//! * a **number** (a date's year, month and day included) must be a
//!   standalone number there ([`crate::gates::first_untraceable_number`]);
//! * a **quoted term** (in double quotes, backticks or curly quotes) must
//!   occur there as written, case and spacing aside;
//! * a **name** must occur there as a whole word, case aside: a word with a
//!   capital that does not begin a sentence, a word with a capital inside
//!   it, an identifier (with `_`, `/`, `\`, a digit among letters or a dot
//!   between characters).
//!
//! The first word of a sentence is not taken for a name, since a capital
//! says nothing there.

use crate::gates::{first_untraceable_number, normalize};

/// A term of a statement that the words it rests on do not carry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unsupported {
    /// `number`, `quoted_term` or `name`.
    pub kind: &'static str,
    /// The term as the statement writes it.
    pub term: String,
}

/// The first term of `statement` that `evidence` does not carry; numbers
/// first, then quoted terms, then names.
#[must_use]
pub fn unsupported_term(statement: &str, evidence: &str) -> Option<Unsupported> {
    if let Some(term) = first_untraceable_number(statement, evidence, false) {
        return Some(Unsupported {
            kind: "number",
            term,
        });
    }
    let (quoted, rest) = split_quoted(statement);
    let normalized = normalize(evidence);
    if let Some(term) = quoted
        .into_iter()
        .find(|t| !normalized.contains(&normalize(t)))
    {
        return Some(Unsupported {
            kind: "quoted_term",
            term,
        });
    }
    let lowered = evidence.to_lowercase();
    names(&rest)
        .into_iter()
        .find(|name| !contains_word(&lowered, &name.to_lowercase()))
        .map(|term| Unsupported { kind: "name", term })
}

/// The quoted terms of `text`, and `text` with them blanked out.
fn split_quoted(text: &str) -> (Vec<String>, String) {
    let mut terms = Vec::new();
    let mut rest = String::with_capacity(text.len());
    let mut open: Option<(char, String)> = None;
    for c in text.chars() {
        match &mut open {
            Some((closer, term)) if c == *closer => {
                if !term.trim().is_empty() {
                    terms.push(term.trim().to_string());
                }
                open = None;
                rest.push(' ');
            }
            Some((_, term)) => term.push(c),
            None => match closer_of(c) {
                Some(closer) => open = Some((closer, String::new())),
                None => rest.push(c),
            },
        }
    }
    // A quote left open quotes nothing: its text stays in the statement.
    if let Some((_, term)) = open {
        rest.push(' ');
        rest.push_str(&term);
    }
    (terms, rest)
}

fn closer_of(opener: char) -> Option<char> {
    match opener {
        '"' => Some('"'),
        '`' => Some('`'),
        '\u{201c}' => Some('\u{201d}'),
        _ => None,
    }
}

/// The words of `text` that are names.
fn names(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut sentence_start = true;
    for raw in text.split_whitespace() {
        let at_start = sentence_start;
        sentence_start = raw.ends_with(['.', '!', '?', ':']);
        let token = raw
            .trim_start_matches(|c: char| !(c.is_alphanumeric() || matches!(c, '/' | '\\' | '.')))
            .trim_end_matches(|c: char| !c.is_alphanumeric());
        let token = token
            .strip_suffix("'s")
            .or_else(|| {
                token
                    .strip_suffix('\u{2019}')
                    .and_then(|t| t.strip_suffix('s'))
            })
            .unwrap_or(token);
        if is_name(token, at_start) && !out.iter().any(|n| n == token) {
            out.push(token.to_string());
        }
    }
    out
}

fn is_name(token: &str, at_start: bool) -> bool {
    let chars: Vec<char> = token.chars().collect();
    if chars.len() < 2 || chars.iter().all(|c| !c.is_alphabetic()) {
        // A number: the number rule's.
        return false;
    }
    let first_upper = chars[0].is_uppercase();
    let inner_upper = chars[1..].iter().any(|c| c.is_uppercase());
    let identifier = chars.iter().any(|c| matches!(c, '_' | '/' | '\\'))
        || chars.iter().any(char::is_ascii_digit)
        || chars
            .windows(3)
            .any(|w| w[1] == '.' && w[0].is_alphanumeric() && w[2].is_alphanumeric());
    identifier || inner_upper || (first_upper && !at_start)
}

/// Whether `needle` occurs in `haystack` with no letter or digit against it.
fn contains_word(haystack: &str, needle: &str) -> bool {
    haystack.match_indices(needle).any(|(at, _)| {
        let before = haystack[..at].chars().next_back();
        let after = haystack[at + needle.len()..].chars().next();
        before.is_none_or(|c| !c.is_alphanumeric()) && after.is_none_or(|c| !c.is_alphanumeric())
    })
}
