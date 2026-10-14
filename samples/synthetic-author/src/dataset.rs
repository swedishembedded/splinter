// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Training records from decisions read out of the articles.

use serde_json::Value;
use splinter_sdk::vocabulary::rng::SplitMix64;

use crate::mine::Verified;
use crate::records::chat_record;
use crate::scenario::Archetype;
use crate::tools::TRAINING;

/// The options a mined decision was made among: the three ways open in every
/// case, and outside help when an offer was made. Their order is shuffled,
/// the same way every time for the same decision.
fn options_of(seed: u64, group: &str, decision: &Verified) -> Vec<Archetype> {
    let mut options = vec![
        Archetype::ActAlone,
        Archetype::ConsultThenAct,
        Archetype::HoldAndWait,
    ];
    if decision.offer_made {
        options.push(Archetype::DeferToOutsider);
    }
    let mix = group
        .bytes()
        .fold(seed, |h, b| h.wrapping_mul(0x100_0000_01B3) ^ u64::from(b));
    SplitMix64::new(mix).shuffle(&mut options);
    options
}

/// One record for `decision` through each tool a model is taught with.
///
/// # Panics
/// Never: the option chosen is one of those listed, since an offer was made
/// whenever outside help was chosen; if the reader said otherwise the choice is
/// listed anyway.
#[must_use]
pub fn records_for(seed: u64, group: &str, decision: &Verified) -> Vec<Value> {
    let mut options = options_of(seed, group, decision);
    if !options.contains(&decision.chose) {
        options.push(decision.chose);
    }
    let motivation = decision.reasons.first().map_or("", String::as_str);
    TRAINING
        .iter()
        .map(|schema| {
            chat_record(
                group,
                &decision.briefing,
                &options,
                schema,
                decision.chose,
                motivation,
            )
        })
        .collect()
}
