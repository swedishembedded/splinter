// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements release gates that adopt a model update
// only on paired, held-out evidence, for its clients. If your team needs
// expertise in evaluation statistics for model releases, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The statistic a release decision rests on: the exact one-sided paired sign
//! test over pass/fail outcomes of two models on the same items, computed
//! here as plain arithmetic (it needs no model).
//!
//! Only discordant pairs carry evidence: an item both models got right, or
//! both got wrong, says nothing about which is better and is excluded from
//! the test. The p-value is `P(Binomial(n, 0.5) >= k)` for `n` discordant
//! pairs of which the candidate won `k`: the chance of at least that many
//! candidate wins if the candidate were no better.

use splinter_eval::significance::{SignTest, Significance};

/// The exact sign test as the gate's [`Significance`].
#[derive(Clone, Copy, Debug, Default)]
pub struct ExactSignTest;

impl Significance for ExactSignTest {
    fn sign_test(&self, pairs: &[(bool, bool)]) -> SignTest {
        sign_test(pairs)
    }
}

/// The sign test over `pairs` of `(candidate right, baseline right)`, one
/// per item, both models on the same item. Taking pairs makes unequal arms
/// unrepresentable.
#[must_use]
pub fn sign_test(pairs: &[(bool, bool)]) -> SignTest {
    let discordant = pairs.iter().filter(|(c, b)| c != b).count();
    let candidate_wins = pairs.iter().filter(|(c, b)| *c && !*b).count();
    SignTest {
        discordant,
        candidate_wins,
        p_value: binomial_upper_tail(discordant, candidate_wins),
    }
}

/// `P(Binomial(n, 1/2) >= k)`, summed in log space so a large `n` neither
/// overflows nor underflows a term that matters: `ln C(n, i)` is carried from
/// one `i` to the next by `ln((n - i) / (i + 1))`, and the terms are added
/// relative to the largest.
fn binomial_upper_tail(n: usize, k: usize) -> f64 {
    if k == 0 {
        return 1.0;
    }
    if k > n {
        return 0.0;
    }
    let mut log_choose = vec![0.0f64; n + 1];
    for i in 0..n {
        log_choose[i + 1] = log_choose[i] + ((n - i) as f64 / (i + 1) as f64).ln();
    }
    let half = n as f64 * std::f64::consts::LN_2;
    let tail = &log_choose[k..];
    let top = tail.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = tail.iter().map(|l| (l - top).exp()).sum();
    (top + sum.ln() - half).exp().min(1.0)
}

/// A mean and the interval a bootstrap puts around it.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct Interval {
    /// The mean of the observations.
    pub mean: f64,
    /// The lower end.
    pub low: f64,
    /// The upper end.
    pub high: f64,
}

impl Interval {
    /// Whether the interval is entirely above zero: the difference it
    /// describes is positive at the level it was made for.
    #[must_use]
    pub fn above_zero(&self) -> bool {
        self.low > 0.0
    }
}

/// The least observations a bootstrap interval is made from.
const MIN_BOOTSTRAP_OBSERVATIONS: usize = 2;

/// A percentile bootstrap interval of the mean of `observations` at `level`
/// (a fraction: the interval that holds that share of the resampled means), over `iterations` resamples with replacement.
/// Pass the per-item difference of two arms on the same items to get a paired
/// interval: an item is the unit, so a gain that rests on a few items shows
/// as a wide one. `seed` makes it repeatable.
///
/// `None` when there are too few observations to resample or the arguments
/// are unusable.
#[must_use]
pub fn bootstrap_interval(
    observations: &[f64],
    iterations: usize,
    level: f64,
    seed: u64,
) -> Option<Interval> {
    if observations.len() < MIN_BOOTSTRAP_OBSERVATIONS
        || iterations == 0
        || !(level > 0.0 && level < 1.0)
        || observations.iter().any(|o| !o.is_finite())
    {
        return None;
    }
    let n = observations.len();
    let mut rng = splinter_core::rng::SplitMix64::new(seed);
    let mut means: Vec<f64> = (0..iterations)
        .map(|_| {
            let total: f64 = (0..n).map(|_| observations[rng.below(n)]).sum();
            total / n as f64
        })
        .collect();
    means.sort_by(f64::total_cmp);
    let at = |q: f64| means[((q * (iterations - 1) as f64).round() as usize).min(iterations - 1)];
    let tail = (1.0 - level) / 2.0;
    Some(Interval {
        mean: observations.iter().sum::<f64>() / n as f64,
        low: at(tail),
        high: at(1.0 - tail),
    })
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

    /// The tail against a direct sum for small `n`, and its symmetry at the
    /// middle for a large one.
    #[test]
    fn the_tail_is_the_binomial_sum() {
        let direct = |n: u32, k: u32| -> f64 {
            let choose = |n: u32, i: u32| -> f64 {
                (0..i)
                    .map(|j| f64::from(n - j) / f64::from(j + 1))
                    .product()
            };
            (k..=n).map(|i| choose(n, i)).sum::<f64>() / 2f64.powi(n as i32)
        };
        for n in 0..=20u32 {
            for k in 0..=n {
                let got = binomial_upper_tail(n as usize, k as usize);
                assert!((got - direct(n, k)).abs() < 1e-12, "n {n} k {k}: {got}");
            }
        }
        // P(X >= 501) is half of 1 - P(X = 500): just under a half.
        let upper = binomial_upper_tail(1000, 501);
        assert!(upper > 0.48 && upper < 0.5, "{upper}");
    }

    #[test]
    fn a_bootstrap_interval_of_a_consistent_gain_excludes_zero_and_of_noise_includes_it() {
        let gain = vec![1.0; 20];
        let interval = bootstrap_interval(&gain, 500, 0.95, 7).unwrap();
        assert_eq!(
            (interval.mean, interval.low, interval.high),
            (1.0, 1.0, 1.0)
        );
        assert!(interval.above_zero());
        let noise: Vec<f64> = (0..20)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let interval = bootstrap_interval(&noise, 2000, 0.95, 7).unwrap();
        assert!(
            interval.mean.abs() < 1e-12 && interval.low < 0.0 && interval.high > 0.0,
            "{interval:?}"
        );
        assert!(!interval.above_zero());
    }

    #[test]
    fn the_interval_is_repeatable_and_narrows_with_more_items() {
        let few: Vec<f64> = (0..10).map(|i| if i < 6 { 1.0 } else { 0.0 }).collect();
        let many: Vec<f64> = (0..200).map(|i| if i < 120 { 1.0 } else { 0.0 }).collect();
        let a = bootstrap_interval(&few, 1000, 0.95, 3).unwrap();
        assert_eq!(a, bootstrap_interval(&few, 1000, 0.95, 3).unwrap());
        let b = bootstrap_interval(&many, 1000, 0.95, 3).unwrap();
        assert!(b.high - b.low < a.high - a.low, "{a:?} {b:?}");
    }

    #[test]
    fn there_is_no_interval_from_too_little_or_unusable_input() {
        assert!(bootstrap_interval(&[1.0], 100, 0.95, 1).is_none());
        assert!(bootstrap_interval(&[1.0, 0.0], 0, 0.95, 1).is_none());
        assert!(bootstrap_interval(&[1.0, 0.0], 100, 1.5, 1).is_none());
        assert!(bootstrap_interval(&[1.0, f64::NAN], 100, 0.95, 1).is_none());
    }
}
