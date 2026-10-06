// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The one seeded generator the task generators built from code draw
//! from, so a seed means the same task on every platform and every build.

/// SplitMix64: a small, fixed, well-mixed generator.
pub(crate) struct SplitMix64(pub(crate) u64);

impl SplitMix64 {
    /// The next value.
    pub(crate) fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A value in `0..n`; `n` is at least 1 at every call site.
    pub(crate) fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }

    /// A value in `low..=high`.
    pub(crate) fn between(&mut self, low: u64, high: u64) -> u64 {
        low + self.next() % (high - low + 1)
    }
}
