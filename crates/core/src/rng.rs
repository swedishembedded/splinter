// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The one seeded generator everything built from code draws from: task
//! generators, partitions and synthetic authors. A seed means the same stream
//! on every platform and in every build, so what a seed produces can be pinned
//! by a digest and drawn again.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// SplitMix64: one 64-bit state, a full period, and no dependency.
#[derive(Clone, Debug)]
pub struct SplitMix64(u64);

impl SplitMix64 {
    /// A generator that starts from `seed`.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// The next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A value in `0..n`; an empty range counts as one choice.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n.max(1) as u64) as usize
    }

    /// A value in `low..=high`; `low` must not exceed `high`.
    pub fn between(&mut self, low: u64, high: u64) -> u64 {
        low + self.next_u64() % (high - low + 1)
    }

    /// A fair coin.
    pub fn coin(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }

    /// One of `items`, or nothing when there are none.
    pub fn pick<'a, T>(&mut self, items: &'a [T]) -> Option<&'a T> {
        items.get(self.below(items.len()))
    }

    /// Puts `items` in a random order (Fisher-Yates).
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i + 1);
            items.swap(i, j);
        }
    }
}

/// The seed of one random draw of a run, as it is printed, recorded and
/// passed back with `--seed`. Core only names the value; where an unset seed
/// comes from is the model crate's concern (`splinter_model::seed::draw`).
///
/// It prints as a decimal and reads decimal or `0x`-prefixed hexadecimal over
/// the whole `u64` range; a negative number, a value above `u64::MAX` and
/// anything else is refused rather than clamped or replaced by a default.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Seed(u64);

impl Seed {
    /// The seed with value `value`.
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// The seed's value, the one a generator starts from.
    #[must_use]
    pub const fn value(self) -> u64 {
        self.0
    }
}

impl From<u64> for Seed {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

impl fmt::Display for Seed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Text that is not a seed.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{text:?} is not a seed (a non-negative integer up to {max}, decimal or 0x-hex)", max = u64::MAX)]
pub struct ParseSeedError {
    text: String,
}

impl FromStr for Seed {
    type Err = ParseSeedError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let trimmed = text.trim();
        let parsed = match trimmed
            .strip_prefix("0x")
            .or_else(|| trimmed.strip_prefix("0X"))
        {
            Some(hex) => u64::from_str_radix(hex, 16),
            None => trimmed.parse::<u64>(),
        };
        parsed.map(Self).map_err(|_| ParseSeedError {
            text: text.to_string(),
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] // a panic is the failure report
mod tests {
    use super::*;

    #[test]
    fn a_seed_prints_as_decimal_and_reads_decimal_or_hex() {
        // seed: the spellings asserted are those of these particular values
        assert_eq!(Seed::new(42).to_string(), "42");
        assert_eq!("42".parse::<Seed>().unwrap(), Seed::new(42));
        assert_eq!("0xFF".parse::<Seed>().unwrap(), Seed::new(255));
        assert_eq!(
            u64::MAX.to_string().parse::<Seed>().unwrap(),
            Seed::new(u64::MAX)
        );
        assert_eq!(serde_json::to_string(&Seed::new(7)).unwrap(), "7");
    }

    #[test]
    fn text_that_is_not_a_seed_is_refused() {
        for text in ["-1", "18446744073709551616", "", "abc", "0x", "1.5"] {
            assert!(text.parse::<Seed>().is_err(), "{text:?}");
        }
    }
}
