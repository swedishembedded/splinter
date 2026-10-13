// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Invented facts: the names, places and figures of a case. They are made of
//! syllables and drawn from the seed, so no model has met them before and
//! whatever a model says about them it learned from the corpus.

use serde::Serialize;
use splinter_sdk::vocabulary::rng::SplitMix64;

const PLACE_STEMS: [&str; 14] = [
    "Dun", "Hol", "Ker", "Mar", "Os", "Pel", "Quin", "Rav", "Sol", "Tarn", "Ul", "Vash", "Wend",
    "Yor",
];
const PLACE_ENDS: [&str; 10] = [
    "mere", "wick", "stead", "combe", "haven", "brook", "gate", "field", "ness", "ton",
];
const HOUSE_STEMS: [&str; 10] = [
    "Aldr", "Bren", "Cal", "Dor", "Esk", "Fen", "Gar", "Hol", "Ivar", "Kest",
];
const HOUSE_ENDS: [&str; 6] = ["ane", "ish", "ow", "ard", "ell", "un"];

/// The matter each topic is about, as one noun phrase a case is named by.
pub const MATTERS: [&str; 12] = [
    "granary",
    "river crossing",
    "apprenticeship",
    "market scales",
    "border fair",
    "harbour light",
    "common pasture",
    "boundary stone",
    "town archive",
    "trading licence",
    "winter relief",
    "mill dam",
];

/// One case: the facts that belong to a single episode.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Case {
    /// How the case is referred to: "the Dunmere granary".
    pub name: String,
    /// The place the case is set in.
    pub place: String,
    /// The year it happened.
    pub year: u32,
    /// How many households depend on the matter.
    pub households: u32,
    /// The outside house that may offer help, by name.
    pub house: String,
}

/// A case for a matter of `topic`, drawn from `rng`.
pub fn case(rng: &mut SplitMix64, topic: usize) -> Case {
    let pick = |rng: &mut SplitMix64, items: &[&str]| items[rng.below(items.len())].to_string();
    let place = format!("{}{}", pick(rng, &PLACE_STEMS), pick(rng, &PLACE_ENDS));
    let house = format!("{}{}", pick(rng, &HOUSE_STEMS), pick(rng, &HOUSE_ENDS));
    Case {
        name: format!("the {place} {}", MATTERS[topic % MATTERS.len()]),
        place,
        year: 1100 + rng.below(700) as u32,
        households: 12 + rng.below(380) as u32,
        house: format!("House {house}"),
    }
}
