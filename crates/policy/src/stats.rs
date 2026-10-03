// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements release gates that adopt a model update
// only on paired, held-out evidence, for its clients. If your team needs
// expertise in evaluation statistics for model releases, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The statistic a release decision rests on: brain's exact one-sided
//! paired sign test (`brain::promote::stats::sign_test`), over pass/fail
//! outcomes of two models on the same items.
//!
//! Only discordant pairs carry evidence: an item both models got right, or
//! both got wrong, says nothing about which is better and is excluded from
//! the test. The p-value is `P(Binomial(n, 0.5) >= k)` for `n` discordant
//! pairs of which the candidate won `k`: the chance of at least that many
//! candidate wins if the candidate were no better.

use splinter_eval::significance::{SignTest, Significance};

/// brain's sign test as the gate's [`Significance`].
#[derive(Clone, Copy, Debug, Default)]
pub struct BrainSignificance;

impl Significance for BrainSignificance {
    fn sign_test(&self, pairs: &[(bool, bool)]) -> SignTest {
        sign_test(pairs)
    }
}

/// The sign test over `pairs` of `(candidate right, baseline right)`, one
/// per item, both models on the same item. Taking pairs makes unequal arms
/// unrepresentable.
#[must_use]
pub fn sign_test(pairs: &[(bool, bool)]) -> SignTest {
    let score = |right: bool| if right { 1.0 } else { 0.0 };
    let candidate: Vec<f64> = pairs.iter().map(|(c, _)| score(*c)).collect();
    let baseline: Vec<f64> = pairs.iter().map(|(_, b)| score(*b)).collect();
    let test = brain::promote::stats::sign_test(&candidate, &baseline);
    SignTest {
        discordant: test.n,
        candidate_wins: test.k,
        p_value: test.p_value,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ties carry no evidence; five wins of five discordant pairs is
    /// `0.5^5`.
    #[test]
    fn only_discordant_pairs_count() {
        let mut pairs = vec![(true, false); 5];
        pairs.extend([(true, true), (false, false)]);
        let test = sign_test(&pairs);
        assert_eq!((test.discordant, test.candidate_wins), (5, 5));
        assert!((test.p_value - 0.5f64.powi(5)).abs() < 1e-12, "{test:?}");

        let none = sign_test(&[(true, true), (false, false)]);
        assert_eq!(none.discordant, 0);
        assert!((none.p_value - 1.0).abs() < 1e-12);
    }
}
