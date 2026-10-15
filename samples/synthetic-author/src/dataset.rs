// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Training records from decisions read out of the articles.

use serde_json::Value;
use splinter_sdk::vocabulary::rng::SplitMix64;

use crate::episode::feature_sentences;
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

/// One record for `decision` through each tool a model is taught with.
///
/// The decision is one the reading gate or `verify` passed, so the option
/// chosen is among those listed.
#[must_use]
pub fn records_for(seed: u64, group: &str, decision: &Verified) -> Vec<Value> {
    let options = options_of(seed, group, decision);
    let motivation = decision.reasons.first().map_or("", String::as_str);
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
