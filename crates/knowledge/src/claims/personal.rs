// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements admission gates that keep ungrounded
// claims out of training data, for its clients. If your team needs
// expertise in synthetic data quality or provenance, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Personal data about someone other than the person, found by code.
//!
//! What a person says about a colleague, a friend or a relative is not theirs
//! to teach a model that others may use, and once in weights it cannot be
//! taken out short of training again. A statement is refused as personal data
//! of a third party when it gives one of five kinds of personal attribute
//! (contact details, a home address, health, a relationship, finances) and is
//! about someone else: the extractor said so (`third_party`), or it did not
//! say the statement is about the person themselves and the statement names
//! someone. The rule is deliberately conservative about what it recognises -
//! a short list of words and shapes - and never decides on a statement the
//! extractor marked as about the person.

use splinter_core::claim::ClaimSubject;

/// Words, word beginnings (ending in `*`) and phrases (with a space) that
/// give an attribute, by kind. Words that also describe software (`health`,
/// `debt`, `address`, `mobile`, `moved`) are left out: the cost of a missed
/// attribute is a fact kept out of the weights by the next rule or by the
/// person's `forget`, the cost of a false refusal is a fact never learned.
const CUES: [(&str, &[&str]); 5] = [
    (
        "contact",
        &[
            "phone",
            "telephone",
            "whatsapp",
            "email address",
            "e-mail address",
        ],
    ),
    (
        "address",
        &[
            "lives",
            "resides",
            "residence",
            "home address",
            "street address",
        ],
    ),
    (
        "health",
        &[
            "diagnos*",
            "illness",
            "disease",
            "cancer",
            "diabetes",
            "medication",
            "surgery",
            "pregnan*",
            "disability",
            "medical condition",
            "health condition",
        ],
    ),
    (
        "relationship",
        &[
            "married",
            "divorce*",
            "spouse",
            "wife",
            "husband",
            "girlfriend",
            "boyfriend",
            "dating",
            "affair",
        ],
    ),
    (
        "finance",
        &[
            "salary",
            "owes",
            "mortgage",
            "iban",
            "bankrupt*",
            "paycheck",
        ],
    ),
];

/// Capitalised words that are not names.
const NOT_NAMES: [&str; 22] = [
    "the", "a", "an", "this", "that", "these", "those", "it", "its", "our", "their", "his", "her",
    "my", "your", "in", "on", "at", "for", "to", "if", "when",
];

/// A phone number's shape: seven to fifteen digits in two to four groups
/// divided by spaces, dashes or brackets, the groups not all of one length (a
/// bare run of digits is an identifier or a timestamp, and equal groups are a
/// list of ports or years).
fn has_phone_number(text: &str) -> bool {
    let mut groups: Vec<usize> = Vec::new();
    let flush = |groups: &mut Vec<usize>| {
        let total: usize = groups.iter().sum();
        let mixed = groups.windows(2).any(|w| w[0] != w[1]);
        let found = (2..=4).contains(&groups.len()) && (7..=15).contains(&total) && mixed;
        groups.clear();
        found
    };
    let mut digits = 0;
    for c in text.chars().chain([';']) {
        if c.is_ascii_digit() {
            digits += 1;
            continue;
        }
        if digits > 0 {
            groups.push(digits);
            digits = 0;
        }
        if !matches!(c, ' ' | '-' | '(' | ')' | '+') && flush(&mut groups) {
            return true;
        }
    }
    false
}

fn cue_kind(statement: &str) -> Option<&'static str> {
    let lowered = statement.to_lowercase();
    let words: Vec<&str> = lowered
        .split(|c: char| !(c.is_alphanumeric() || c == '-'))
        .filter(|w| !w.is_empty())
        .collect();
    CUES.iter()
        .find(|(_, cues)| {
            cues.iter().any(|cue| match cue.strip_suffix('*') {
                Some(stem) => words.iter().any(|w| w.starts_with(stem)),
                None if cue.contains(' ') => lowered.contains(cue),
                None => words.contains(cue),
            })
        })
        .map(|(kind, _)| *kind)
        .or_else(|| has_phone_number(statement).then_some("contact"))
}

/// Whether `statement` names someone: a capitalised word, wherever it stands
/// in a sentence, that is not a determiner or a preposition.
fn names_someone(statement: &str) -> bool {
    statement.split(|c: char| !c.is_alphanumeric()).any(|w| {
        w.chars().next().is_some_and(char::is_uppercase)
            && w.chars().count() > 1
            && !NOT_NAMES.contains(&w.to_lowercase().as_str())
    })
}

/// The kind of personal attribute `statement` gives of someone other than
/// the person, when it does; see the module documentation.
#[must_use]
pub fn third_party_personal(
    statement: &str,
    subject: Option<ClaimSubject>,
) -> Option<&'static str> {
    if subject == Some(ClaimSubject::Own) {
        return None;
    }
    let kind = cue_kind(statement)?;
    (subject == Some(ClaimSubject::ThirdParty) || names_someone(statement)).then_some(kind)
}
