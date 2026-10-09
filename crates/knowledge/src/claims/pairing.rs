// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements admission gates that keep ungrounded
// claims out of training data, for its clients. If your team needs
// expertise in synthetic data quality or provenance, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Which claims are about one thing, and whether two of them say the same,
//! decided by code.
//!
//! The question an extractor writes for a claim is its own wording and says
//! little about whether two claims are about one thing, so claims are paired
//! on what they name: two claims are *candidates* when they share a name (a
//! capitalised word that does not begin a sentence, an identifier, a quoted
//! term: [`super::terms::names_of`], over the statement and the question) and
//! at least one more content word between them. Two claims that name nothing
//! at all are candidates when their statements share most of their content
//! words. A judge ([`super::ClaimJudge`]) decides a candidate pair. Without
//! one, the statements decide: they *agree* when one contains the other or
//! when they state the same numbers, and a candidate pair that agrees
//! reinforces while one that does not supersedes (the later wins). Without a
//! judge, two different facts about one named subject are one pair, so the
//! later replaces the earlier; the ledger keeps both rulings.

use std::collections::BTreeSet;

use splinter_core::claim::Claim;

use super::terms::names_of;
use crate::tasks::dedup::references_agree;

/// Words that carry no subject.
const STOP_WORDS: [&str; 24] = [
    "about", "also", "does", "from", "have", "into", "than", "that", "their", "then", "there",
    "these", "this", "those", "was", "were", "what", "when", "where", "which", "will", "with",
    "would", "your",
];

/// The share of content words two claims with no names must have in common.
const NAMELESS_OVERLAP: f64 = 0.5;

/// The lower-cased words of `text` that say what it is about: at least four
/// letters or any digit, and no stop word.
fn content_words(text: &str) -> BTreeSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .map(str::to_lowercase)
        .filter(|w| {
            (w.chars().count() >= 4 || w.chars().any(|c| c.is_ascii_digit()))
                && !STOP_WORDS.contains(&w.as_str())
        })
        .collect()
}

/// The names `claim` carries, lower-cased, in its statement and question.
fn names(claim: &Claim) -> BTreeSet<String> {
    names_of(&claim.statement)
        .into_iter()
        .chain(names_of(&claim.question))
        .flat_map(|n| content_words(&n).into_iter().chain([n.to_lowercase()]))
        .collect()
}

fn words(claim: &Claim) -> BTreeSet<String> {
    content_words(&format!("{} {}", claim.statement, claim.question))
}

/// Whether `a` and `b` are about one thing; see the module documentation.
#[must_use]
pub fn same_subject(a: &Claim, b: &Claim) -> bool {
    let (names_a, names_b) = (names(a), names(b));
    if names_a.is_empty() && names_b.is_empty() {
        let (words_a, words_b) = (content_words(&a.statement), content_words(&b.statement));
        let shared = words_a.intersection(&words_b).count();
        let all = words_a.union(&words_b).count();
        return shared >= 2 && shared as f64 / all as f64 >= NAMELESS_OVERLAP;
    }
    names_a.intersection(&names_b).next().is_some() && words(a).intersection(&words(b)).count() >= 2
}

/// The numbers `text` states, as written.
fn numbers(text: &str) -> BTreeSet<String> {
    text.split(|c: char| !(c.is_ascii_digit() || c == '.' || c == ','))
        .map(|n| n.trim_matches(['.', ',']).replace(',', ""))
        .filter(|n| !n.is_empty())
        .collect()
}

/// Whether two statements about one subject say the same thing: one holds
/// the other, or both state numbers and the same ones.
#[must_use]
pub fn statements_agree(a: &str, b: &str) -> bool {
    if references_agree(a, b) {
        return true;
    }
    let (a, b) = (numbers(a), numbers(b));
    !a.is_empty() && a == b
}

/// Whether two statements, not necessarily about the same subject by
/// [`same_subject`], state the same thing: they [`statements_agree`], or
/// share most of their content words and state no numbers that differ.
#[must_use]
pub fn statements_alike(a: &str, b: &str) -> bool {
    if statements_agree(a, b) {
        return true;
    }
    let (numbers_a, numbers_b) = (numbers(a), numbers(b));
    if numbers_a != numbers_b {
        return false;
    }
    let (words_a, words_b) = (content_words(a), content_words(b));
    let all = words_a.union(&words_b).count();
    all > 0 && words_a.intersection(&words_b).count() as f64 / all as f64 >= NAMELESS_OVERLAP
}
