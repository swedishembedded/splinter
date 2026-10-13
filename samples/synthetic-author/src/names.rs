// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Human-readable names for synthetic authors, made of syllables no real
//! person is likely to carry together.

use crate::rng::Rng;

const FIRST_STEMS: [&str; 12] = [
    "Mar", "Tol", "Isa", "Ren", "Kal", "Dev", "Ors", "Vel", "Jun", "Bra", "Hes", "Lun",
];
const FIRST_ENDS: [&str; 8] = ["ek", "ra", "ian", "a", "os", "en", "ith", "el"];
const FAMILY_STEMS: [&str; 10] = [
    "Tal", "Ven", "Cor", "Hal", "Wyn", "Ash", "Brin", "Sel", "Dar", "Fen",
];
const FAMILY_ENDS: [&str; 10] = [
    "ane", "wick", "dorn", "mere", "ley", "stad", "holm", "ford", "croft", "vale",
];

/// A given name and a family name, each capitalised.
pub fn name(rng: &mut Rng) -> String {
    format!(
        "{}{} {}{}",
        FIRST_STEMS[rng.below(FIRST_STEMS.len())],
        FIRST_ENDS[rng.below(FIRST_ENDS.len())],
        FAMILY_STEMS[rng.below(FAMILY_STEMS.len())],
        FAMILY_ENDS[rng.below(FAMILY_ENDS.len())],
    )
}
