// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements judges of synthetic speech calibrated against
// the people who listen to it, for its clients. If your team needs expertise
// in learning a listener's preference from a few hundred comparisons, you can
// procure our services by sending an email to info@swedishembedded.com.

//! A judge of how natural a take sounds, learned from a listener's pairwise
//! choices.
//!
//! A person who hears two takes of the same sentence says which sounds more
//! natural. The judge is a linear score on features of a take, fitted so that the
//! probability the first beats the second is the logistic function of the
//! difference of their scores (the Bradley-Terry model), with an L2 penalty
//! because there are far fewer comparisons than features. It is fitted on the
//! listener's choices, on control pairs whose answer is known (a take against
//! the same take with noise added), or both, weighted. The listener's own held-out
//! choices are the test of it; its agreement with them, not its fit, is the claim.

use serde::{Deserialize, Serialize};

/// One comparison: the features of two takes, and how strongly to count it.
#[derive(Clone, Debug)]
pub struct Comparison {
    /// Features of the take that won.
    pub winner: Vec<f32>,
    /// Features of the take that lost.
    pub loser: Vec<f32>,
    /// How much this comparison counts.
    pub weight: f32,
}

/// A fitted judge.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Judge {
    mean: Vec<f32>,
    scale: Vec<f32>,
    weights: Vec<f32>,
}

/// How a judge is fitted.
#[derive(Clone, Copy, Debug)]
pub struct FitOptions {
    /// L2 penalty on the weights.
    pub l2: f32,
    /// Passes of gradient descent.
    pub epochs: usize,
    /// Step size.
    pub learning_rate: f32,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            l2: 1.0,
            epochs: 400,
            learning_rate: 0.1,
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl Judge {
    /// Fit a judge to `comparisons`. Features are standardised over every take
    /// that appears in them.
    ///
    /// # Panics
    /// With no comparisons, or comparisons whose features differ in length.
    #[must_use]
    pub fn fit(comparisons: &[Comparison], options: &FitOptions) -> Self {
        assert!(
            !comparisons.is_empty(),
            "a judge needs at least one comparison"
        );
        let dims = comparisons[0].winner.len();
        assert!(
            comparisons
                .iter()
                .all(|c| c.winner.len() == dims && c.loser.len() == dims),
            "every take has the same number of features"
        );
        let takes: Vec<&Vec<f32>> = comparisons
            .iter()
            .flat_map(|c| [&c.winner, &c.loser])
            .collect();
        let n = takes.len() as f32;
        let mean: Vec<f32> = (0..dims)
            .map(|d| takes.iter().map(|t| t[d]).sum::<f32>() / n)
            .collect();
        let scale: Vec<f32> = (0..dims)
            .map(|d| {
                let var = takes.iter().map(|t| (t[d] - mean[d]).powi(2)).sum::<f32>() / n;
                var.sqrt().max(1e-6)
            })
            .collect();
        let mut judge = Self {
            mean,
            scale,
            weights: vec![0.0; dims],
        };
        let diffs: Vec<(Vec<f32>, f32)> = comparisons
            .iter()
            .map(|c| {
                let a = judge.standardise(&c.winner);
                let b = judge.standardise(&c.loser);
                (a.iter().zip(&b).map(|(x, y)| x - y).collect(), c.weight)
            })
            .collect();
        let total_weight: f32 = diffs.iter().map(|(_, w)| w).sum();
        for _ in 0..options.epochs {
            let mut gradient = vec![0.0f32; dims];
            for (diff, weight) in &diffs {
                let z: f32 = diff.iter().zip(&judge.weights).map(|(x, w)| x * w).sum();
                // d(-log sigmoid(z))/dw = -(1 - sigmoid(z)) * diff
                let pull = weight * (1.0 - sigmoid(z));
                for (g, x) in gradient.iter_mut().zip(diff) {
                    *g -= pull * x / total_weight;
                }
            }
            for (w, g) in judge.weights.iter_mut().zip(&gradient) {
                *w -= options.learning_rate * (g + options.l2 * *w / total_weight);
            }
        }
        judge
    }

    fn standardise(&self, features: &[f32]) -> Vec<f32> {
        features
            .iter()
            .zip(&self.mean)
            .zip(&self.scale)
            .map(|((x, m), s)| (x - m) / s)
            .collect()
    }

    /// The weight of each feature, on standardised features.
    #[must_use]
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// The judge's score of a take. Only differences between takes of one
    /// sentence mean anything.
    #[must_use]
    pub fn score(&self, features: &[f32]) -> f32 {
        self.standardise(features)
            .iter()
            .zip(&self.weights)
            .map(|(x, w)| x * w)
            .sum()
    }

    /// The probability the judge gives to `first` sounding better than `second`.
    #[must_use]
    pub fn probability(&self, first: &[f32], second: &[f32]) -> f32 {
        sigmoid(self.score(first) - self.score(second))
    }

    /// The share of `comparisons` whose winner the judge scores above the loser.
    #[must_use]
    pub fn agreement(&self, comparisons: &[Comparison]) -> f32 {
        let right = comparisons
            .iter()
            .filter(|c| self.score(&c.winner) > self.score(&c.loser))
            .count();
        right as f32 / comparisons.len().max(1) as f32
    }
}

/// The share of `comparisons` a judge predicts when it is fitted on all the
/// others and asked about the one left out, over every choice of the one left
/// out. With the few hundred comparisons a listener can give, this is the
/// honest estimate of agreement with that listener on comparisons not yet made.
#[must_use]
pub fn leave_one_out(comparisons: &[Comparison], options: &FitOptions) -> f32 {
    leave_one_out_with(comparisons, &[], options)
}

/// [`leave_one_out`] where every fit also learns from `always`, comparisons that
/// are never left out and never counted (degraded copies of takes, for example).
#[must_use]
pub fn leave_one_out_with(
    comparisons: &[Comparison],
    always: &[Comparison],
    options: &FitOptions,
) -> f32 {
    let right = (0..comparisons.len())
        .filter(|&held| {
            let mut rest: Vec<Comparison> = comparisons
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != held)
                .map(|(_, c)| c.clone())
                .collect();
            rest.extend_from_slice(always);
            Judge::fit(&rest, options).agreement(&comparisons[held..=held]) > 0.5
        })
        .count();
    right as f32 / comparisons.len().max(1) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic pseudo-random sequence in [0, 1).
    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32) / (1u64 << 31) as f32
    }

    /// Takes with 6 features of which only the first two carry quality; the
    /// listener prefers the take with the higher `quality = f0 - f1`, with some
    /// noise in the listener's choice.
    fn comparisons(count: usize, seed: u64, noise: f32) -> Vec<Comparison> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                let take = |s: &mut u64| (0..6).map(|_| lcg(s) * 2.0 - 1.0).collect::<Vec<f32>>();
                let (a, b) = (take(&mut state), take(&mut state));
                let mut quality = |t: &Vec<f32>| t[0] - t[1] + noise * (lcg(&mut state) - 0.5);
                let (qa, qb) = (quality(&a), quality(&b));
                if qa >= qb {
                    Comparison {
                        winner: a,
                        loser: b,
                        weight: 1.0,
                    }
                } else {
                    Comparison {
                        winner: b,
                        loser: a,
                        weight: 1.0,
                    }
                }
            })
            .collect()
    }

    #[test]
    fn a_judge_learns_which_features_make_a_take_better_and_agrees_with_comparisons_it_never_saw() {
        let train = comparisons(300, 1, 0.3);
        let test = comparisons(300, 2, 0.3);
        let judge = Judge::fit(&train, &FitOptions::default());
        let agreement = judge.agreement(&test);
        assert!(agreement > 0.85, "held-out agreement {agreement}");
        // The quality features carry the weight, the others little.
        let weights = &judge.weights;
        assert!(weights[0] > 0.5 && weights[1] < -0.5, "{weights:?}");
        assert!(
            weights[2..]
                .iter()
                .all(|w| w.abs() < weights[0].abs() / 2.0),
            "{weights:?}"
        );
    }

    #[test]
    fn with_no_signal_a_judge_does_no_better_than_chance_on_new_comparisons() {
        // Winners and losers are the same distribution: nothing to learn.
        let mut state = 9u64;
        let random = |count: usize, state: &mut u64| -> Vec<Comparison> {
            (0..count)
                .map(|_| {
                    let take = |s: &mut u64| (0..6).map(|_| lcg(s)).collect::<Vec<f32>>();
                    Comparison {
                        winner: take(state),
                        loser: take(state),
                        weight: 1.0,
                    }
                })
                .collect()
        };
        let judge = Judge::fit(&random(200, &mut state), &FitOptions::default());
        let agreement = judge.agreement(&random(2000, &mut state));
        assert!(
            (agreement - 0.5).abs() < 0.07,
            "chance expected, got {agreement}"
        );
    }

    #[test]
    fn a_heavier_comparison_pulls_the_judge_harder() {
        let a: Vec<f32> = vec![1.0, -1.0, 0.0, 0.0, 0.0, 0.0];
        let b: Vec<f32> = vec![-1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let fitted = |weight: f32| {
            let mut set = comparisons(60, 3, 0.0);
            // One comparison against the other sixty, counted lightly or heavily.
            set.push(Comparison {
                winner: b.clone(),
                loser: a.clone(),
                weight,
            });
            Judge::fit(&set, &FitOptions::default()).probability(&a, &b)
        };
        let (light, heavy) = (fitted(0.01), fitted(50.0));
        assert!(
            light > 0.5,
            "sixty agreeing comparisons outvote one light dissent: {light}"
        );
        assert!(
            heavy < light,
            "a heavy dissent moves the judge: {heavy} vs {light}"
        );
    }

    #[test]
    fn leave_one_out_finds_the_signal_when_there_is_one_and_chance_when_there_is_none() {
        let signal = leave_one_out(&comparisons(80, 4, 0.3), &FitOptions::default());
        assert!(signal > 0.8, "{signal}");
        let mut state = 5u64;
        let none: Vec<Comparison> = (0..80)
            .map(|_| {
                let take = |s: &mut u64| (0..6).map(|_| lcg(s)).collect::<Vec<f32>>();
                Comparison {
                    winner: take(&mut state),
                    loser: take(&mut state),
                    weight: 1.0,
                }
            })
            .collect();
        let chance = leave_one_out(&none, &FitOptions::default());
        assert!((chance - 0.5).abs() < 0.15, "{chance}");
    }
}
