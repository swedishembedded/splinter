// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements release gates that adopt a model update
// only on paired, held-out evidence, for its clients. If your team needs
// expertise in evaluation statistics for model releases, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The statistic a release decision rests on: a one-sided paired sign test
//! over pass/fail outcomes of two models on the same items.
//!
//! Only discordant pairs carry evidence: an item both models got right, or
//! both got wrong, says nothing about which is better and is excluded from
//! the test. The p-value is `P(Binomial(n, 0.5) >= k)` for `n` discordant
//! pairs of which the candidate won `k`: the chance of at least that many
//! candidate wins if the candidate were no better.
//!
//! The test itself is arithmetic the model backend already owns, so a
//! [`Significance`] is handed in: the gate decides what the numbers mean,
//! and never computes them a second way.

use serde::{Deserialize, Serialize};

/// One sign test's result.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SignTest {
    /// Discordant pairs: one model right, the other wrong.
    pub discordant: usize,
    /// Of those, the pairs the candidate got right.
    pub candidate_wins: usize,
    /// The one-sided p-value; `1.0` when no pair is discordant.
    pub p_value: f64,
}

/// A paired sign test.
pub trait Significance {
    /// The sign test over `pairs` of `(candidate right, baseline right)`,
    /// one per item, both models on the same item. Taking pairs makes
    /// unequal arms unrepresentable.
    fn sign_test(&self, pairs: &[(bool, bool)]) -> SignTest;
}
