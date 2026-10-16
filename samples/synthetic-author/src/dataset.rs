// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Training records from decisions read out of the articles.

use serde_json::Value;
use splinter_sdk::vocabulary::rng::SplitMix64;

use crate::episode::feature_sentences;
use crate::gates::brief_phrase_copied;
use crate::mine::Verified;
use crate::records::chat_record;
use crate::scenario::Archetype;
use crate::tools::TRAINING;

/// The options a mined decision was made among: exactly the ways the reader
/// found open, shuffled the same way every time for the same decision.
fn options_of(seed: u64, group: &str, decision: &Verified) -> Vec<Archetype> {
    let mut options = decision.open.clone();
    let mix = group
        .bytes()
        .fold(seed, |h, b| h.wrapping_mul(0x100_0000_01B3) ^ u64::from(b));
    SplitMix64::new(mix).shuffle(&mut options);
    options
}

/// The reason a decision teaches: the first reason quoted from the essay that
/// is not blank and does not repeat a phrase of the brief the essay was
/// written from, which would teach the brief's wording as the author's.
#[must_use]
pub fn motivation_of(decision: &Verified) -> Option<&str> {
    decision
        .reasons
        .iter()
        .map(|r| r.trim())
        .find(|r| !r.is_empty() && brief_phrase_copied(r).is_none())
}

/// One record for `decision` through each tool a model is taught with, or
/// none when the decision has no reason to teach: a reason is part of what is
/// learned, so a record never carries an empty one.
///
/// The decision is one the reading gate or `verify` passed, so the option
/// chosen is among those listed.
#[must_use]
pub fn records_for(seed: u64, group: &str, decision: &Verified) -> Vec<Value> {
    let Some(motivation) = motivation_of(decision) else {
        return Vec::new();
    };
    let options = options_of(seed, group, decision);
    // The reader's briefing, and, when the essay stated every fact, the facts
    // in the words the examination puts them in.
    let mut briefings = vec![decision.briefing.clone()];
    if let Some(features) = decision.facts.features(decision.offer_made) {
        briefings.push(format!(
            "A matter arose over {}. {}",
            decision.case,
            feature_sentences(&features, "An outside house")
        ));
    }
    briefings
        .iter()
        .flat_map(|briefing| {
            TRAINING.iter().map(|schema| {
                chat_record(
                    group,
                    briefing,
                    &options,
                    schema,
                    decision.chose,
                    motivation,
                )
            })
        })
        .collect()
}
