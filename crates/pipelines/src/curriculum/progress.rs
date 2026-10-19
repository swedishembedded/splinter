// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements curricula that spend a learner's GPU time on
// the task families it is still improving on, for its clients. If your team
// needs expertise in automatic curricula for fine-tuning agents, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Learning progress: which task families deserve solver and teacher time.
//!
//! The bandit framing of a self-evolving curriculum (Self-Evolving Curriculum
//! for LLM Reasoning, 2025): each family is an arm, and its reward is how
//! fast the policy's verified pass rate on it is moving, per unit of GPU
//! time it costs, not how hard it is. A family whose rate rises quickly is
//! being learned; one that stays flat at a low rate is out of reach for now
//! and one at a high rate is done. Neither of those pays back time spent on
//! it. This is the competence signal only, not a learned predictor of it.
//!
//! * A [`History`] is a family's verified outcomes, one [`Round`] per policy
//!   that attempted it, oldest first.
//! * [`progress`] is the absolute difference between the mean pass rate of
//!   the newest window of rounds and the window before it. Two windows make
//!   it a smoothed slope, and the absolute value counts a family that is
//!   being forgotten as one the policy still has something to learn about.
//! * [`weights`] divides it by the family's GPU seconds per attempt, relative
//!   to the mean over families (a family whose attempts cost nothing known is
//!   priced at the mean, never at zero), adds a novelty bonus that shrinks
//!   with attempts so an unmeasured family is not ignored, and mixes the
//!   result with the uniform distribution: every family keeps at least
//!   `epsilon / n` of the draw, so none starves and a changed policy can
//!   revive a family that looked flat.
//! * [`order`] draws every family once, without replacement, proportionally
//!   to its weight, from a seed and the family names alone (never the order
//!   they were met in). [`ranked`] is the deterministic variant, heaviest
//!   first, for a budget that will not reach everything.

use std::collections::BTreeMap;

use serde::Serialize;
use splinter_core::digest::Digest;
use splinter_core::rng::Seed;

use splinter_orchestrator::error::OrchestratorError;

/// The smallest cost ratio a family is divided by: a family whose attempts
/// are very cheap must not take the whole draw on a rounding difference.
pub const MIN_COST_RATIO: f64 = 0.05;

/// One policy's attempts at a family.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct Round {
    /// Graded attempts.
    pub attempts: u32,
    /// Of those, verified passes.
    pub passes: u32,
    /// Device-busy seconds the attempts took, when every one recorded it.
    pub gpu_seconds: Option<f64>,
}

impl Round {
    fn rate(&self) -> Option<f64> {
        (self.attempts > 0).then(|| f64::from(self.passes) / f64::from(self.attempts))
    }
}

/// A family's rounds, oldest first.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct History {
    /// The rounds.
    pub rounds: Vec<Round>,
}

impl History {
    /// Graded attempts over every round.
    #[must_use]
    pub fn attempts(&self) -> u32 {
        self.rounds.iter().map(|r| r.attempts).sum()
    }
}

/// The knobs of the strategy.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Params {
    /// Rounds in one of the two windows [`progress`] compares.
    pub window: usize,
    /// The novelty bonus of a family with no attempts, in pass-rate units.
    pub novelty: f64,
    /// The share of the draw that is uniform, in `[0, 1]`.
    pub epsilon: f64,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            window: 2,
            novelty: 0.1,
            epsilon: 0.1,
        }
    }
}

impl Params {
    /// Refuses a parameter outside its range.
    pub fn validate(&self) -> Result<(), OrchestratorError> {
        let refuse = |what: String| Err(OrchestratorError::Refused(what));
        if self.window == 0 {
            return refuse("the learning-progress window is at least one round".into());
        }
        if !(self.novelty.is_finite() && self.novelty >= 0.0) {
            return refuse(format!(
                "the novelty bonus {} is not a number >= 0",
                self.novelty
            ));
        }
        if !(self.epsilon.is_finite() && (0.0..=1.0).contains(&self.epsilon)) {
            return refuse(format!(
                "the exploration share {} is not in [0, 1]",
                self.epsilon
            ));
        }
        Ok(())
    }
}

/// What a family's draw weight was made of.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Score {
    /// Absolute change in pass rate between the two windows; 0 with fewer
    /// than two measured rounds.
    pub progress: f64,
    /// GPU seconds per attempt over the recent rounds; `None` when none
    /// recorded it.
    pub gpu_seconds_per_attempt: Option<f64>,
    /// The novelty bonus.
    pub novelty: f64,
    /// The share of the draw, over all families summing to 1.
    pub weight: f64,
}

/// The absolute change in verified pass rate between the newest window of
/// rounds and the one before it; see the module documentation. A window is
/// `window` rounds, fewer when the history is short (half of it, at least
/// one round), and a round with no graded attempt is not a round.
#[must_use]
pub fn progress(history: &History, window: usize) -> f64 {
    let rates: Vec<f64> = history.rounds.iter().filter_map(Round::rate).collect();
    if rates.len() < 2 {
        return 0.0;
    }
    let width = window.min(rates.len() / 2).max(1);
    let mean = |rates: &[f64]| rates.iter().sum::<f64>() / rates.len() as f64;
    let (earlier, recent) = rates.split_at(rates.len() - width);
    let earlier = &earlier[earlier.len().saturating_sub(width)..];
    (mean(recent) - mean(earlier)).abs()
}

/// GPU seconds per attempt over the newest `2 * window` rounds that recorded
/// them; `None` when none did.
fn gpu_per_attempt(history: &History, window: usize) -> Option<f64> {
    let (seconds, attempts) = history
        .rounds
        .iter()
        .rev()
        .take(2 * window)
        .filter_map(|r| r.gpu_seconds.map(|s| (s, f64::from(r.attempts))))
        .fold((0.0, 0.0), |(s, a), (rs, ra)| (s + rs, a + ra));
    (attempts > 0.0).then(|| seconds / attempts)
}

/// Each family's [`Score`]; the weights sum to 1. An empty map has none.
#[must_use]
pub fn weights(histories: &BTreeMap<String, History>, params: &Params) -> BTreeMap<String, Score> {
    let costs: BTreeMap<&String, Option<f64>> = histories
        .iter()
        .map(|(family, h)| (family, gpu_per_attempt(h, params.window)))
        .collect();
    let measured: Vec<f64> = costs.values().flatten().copied().collect();
    let mean_cost =
        (!measured.is_empty()).then(|| measured.iter().sum::<f64>() / measured.len() as f64);
    let raw: BTreeMap<&String, (f64, f64)> = histories
        .iter()
        .map(|(family, h)| {
            let ratio = match (costs[family], mean_cost) {
                (Some(cost), Some(mean)) if mean > 0.0 => (cost / mean).max(MIN_COST_RATIO),
                _ => 1.0,
            };
            let novelty = params.novelty / f64::from(1 + h.attempts()).sqrt();
            (
                family,
                (progress(h, params.window) / ratio + novelty, novelty),
            )
        })
        .collect();
    let total: f64 = raw.values().map(|(score, _)| score).sum();
    let uniform = 1.0 / histories.len().max(1) as f64;
    histories
        .iter()
        .map(|(family, h)| {
            let (score, novelty) = raw[family];
            let learned = if total > 0.0 { score / total } else { uniform };
            (
                family.clone(),
                Score {
                    progress: progress(h, params.window),
                    gpu_seconds_per_attempt: costs[family],
                    novelty,
                    weight: (1.0 - params.epsilon) * learned + params.epsilon * uniform,
                },
            )
        })
        .collect()
}

/// A uniform number in `(0, 1)` that depends on `seed` and `family` alone.
fn unit(seed: Seed, family: &str) -> f64 {
    let mut input = seed.value().to_le_bytes().to_vec();
    input.extend_from_slice(family.as_bytes());
    let digest = Digest::of(&input);
    // A digest is hex; 16 digits are 64 bits, of which a float keeps 53.
    let bits = u64::from_str_radix(&digest.hex()[..16], 16).unwrap_or(0);
    ((bits >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

/// Every family once, drawn without replacement in proportion to its weight
/// (each is given the key `ln(u) / weight`, the largest first), by a draw
/// that depends on `seed` and the family names alone.
#[must_use]
pub fn order(scores: &BTreeMap<String, Score>, seed: Seed) -> Vec<String> {
    let mut keyed: Vec<(f64, &String)> = scores
        .iter()
        .map(|(family, score)| (unit(seed, family).ln() / score.weight, family))
        .collect();
    keyed.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(b.1)));
    keyed
        .into_iter()
        .map(|(_, family)| family.clone())
        .collect()
}

/// Every family, heaviest weight first, ties by name.
#[must_use]
pub fn ranked(scores: &BTreeMap<String, Score>) -> Vec<String> {
    let mut all: Vec<(&String, f64)> = scores.iter().map(|(f, s)| (f, s.weight)).collect();
    all.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    all.into_iter().map(|(family, _)| family.clone()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rounds(rates: &[(u32, u32)], gpu_per_round: Option<f64>) -> History {
        History {
            rounds: rates
                .iter()
                .map(|&(passes, attempts)| Round {
                    attempts,
                    passes,
                    gpu_seconds: gpu_per_round,
                })
                .collect(),
        }
    }

    fn histories(gpu: [Option<f64>; 3]) -> BTreeMap<String, History> {
        BTreeMap::from([
            (
                "improving".to_string(),
                rounds(&[(1, 10), (3, 10), (6, 10), (8, 10)], gpu[0]),
            ),
            (
                "flat-low".to_string(),
                rounds(&[(1, 10), (1, 10), (1, 10), (1, 10)], gpu[1]),
            ),
            (
                "saturated".to_string(),
                rounds(&[(10, 10), (10, 10), (10, 10), (10, 10)], gpu[2]),
            ),
        ])
    }

    /// How often each family is drawn first over many seeds.
    fn first_shares(scores: &BTreeMap<String, Score>, draws: u64) -> BTreeMap<String, f64> {
        let mut firsts: BTreeMap<String, f64> = BTreeMap::new();
        for seed in 0..draws {
            let first = order(scores, Seed::new(seed)).remove(0);
            *firsts.entry(first).or_default() += 1.0 / draws as f64;
        }
        firsts
    }

    #[test]
    fn a_fast_improving_family_is_drawn_more_than_flat_and_saturated_ones() {
        let scores = weights(&histories([None; 3]), &Params::default());
        let shares = first_shares(&scores, 4000);
        assert!(shares["improving"] > 0.7, "{shares:?}");
        assert!(shares["improving"] > 3.0 * shares["flat-low"], "{shares:?}");
        assert!(
            shares["improving"] > 3.0 * shares["saturated"],
            "{shares:?}"
        );
    }

    #[test]
    fn a_family_that_is_being_forgotten_still_counts_as_moving() {
        let falling = rounds(&[(9, 10), (6, 10), (3, 10), (1, 10)], None);
        assert!(progress(&falling, 2) > 0.5);
    }

    #[test]
    fn progress_needs_two_measured_rounds() {
        assert_eq!(progress(&History::default(), 2), 0.0);
        assert_eq!(progress(&rounds(&[(5, 10)], None), 2), 0.0);
        assert_eq!(progress(&rounds(&[(5, 10), (0, 0)], None), 2), 0.0);
    }

    #[test]
    fn the_exploration_floor_holds_for_every_family() {
        let params = Params {
            epsilon: 0.3,
            ..Params::default()
        };
        let scores = weights(&histories([None; 3]), &params);
        for (family, score) in &scores {
            assert!(
                score.weight >= 0.3 / 3.0 - 1e-12,
                "{family}: {}",
                score.weight
            );
        }
        let total: f64 = scores.values().map(|s| s.weight).sum();
        assert!((total - 1.0).abs() < 1e-9);
    }

    #[test]
    fn no_family_starves_when_nothing_is_moving() {
        let flat = BTreeMap::from([
            ("a".to_string(), rounds(&[(2, 10), (2, 10)], None)),
            ("b".to_string(), rounds(&[(10, 10), (10, 10)], None)),
        ]);
        let params = Params {
            novelty: 0.0,
            ..Params::default()
        };
        let shares = first_shares(&weights(&flat, &params), 2000);
        assert!(shares["a"] > 0.4 && shares["b"] > 0.4, "{shares:?}");
    }

    #[test]
    fn a_family_with_no_attempts_is_not_ignored() {
        let mut all = histories([None; 3]);
        all.insert("unseen".into(), History::default());
        let scores = weights(&all, &Params::default());
        assert!(scores["unseen"].weight > scores["saturated"].weight);
        assert!(scores["unseen"].novelty > scores["improving"].novelty);
    }

    #[test]
    fn the_same_seed_gives_the_same_order_whatever_order_families_were_met_in() {
        // seed: the property under test is that a seed fixes the order.
        let scores = weights(&histories([None; 3]), &Params::default());
        let reversed: BTreeMap<String, Score> = scores.clone().into_iter().rev().collect();
        for seed in 0..50 {
            assert_eq!(
                order(&scores, Seed::new(seed)),
                order(&reversed, Seed::new(seed))
            );
        }
        assert_ne!(
            (0..50)
                .map(|s| order(&scores, Seed::new(s)))
                .collect::<Vec<_>>(),
            vec![order(&scores, Seed::new(0)); 50],
            "different seeds must be able to draw different orders"
        );
    }

    #[test]
    fn a_costlier_family_is_drawn_less_for_the_same_progress() {
        let both = BTreeMap::from([
            ("cheap".to_string(), rounds(&[(2, 10), (6, 10)], Some(10.0))),
            ("dear".to_string(), rounds(&[(2, 10), (6, 10)], Some(100.0))),
        ]);
        let scores = weights(&both, &Params::default());
        assert!(
            scores["cheap"].weight > 2.0 * scores["dear"].weight,
            "{scores:?}"
        );
    }

    #[test]
    fn an_unmeasured_cost_is_priced_at_the_mean_not_at_zero() {
        let mixed = BTreeMap::from([
            (
                "measured".to_string(),
                rounds(&[(2, 10), (6, 10)], Some(40.0)),
            ),
            ("unmeasured".to_string(), rounds(&[(2, 10), (6, 10)], None)),
        ]);
        let scores = weights(&mixed, &Params::default());
        assert!(scores["unmeasured"].gpu_seconds_per_attempt.is_none());
        let (a, b) = (scores["measured"].weight, scores["unmeasured"].weight);
        assert!((a - b).abs() < 1e-9, "{a} vs {b}");
    }

    #[test]
    fn ranking_is_heaviest_first_with_ties_by_name() {
        let scores = weights(&histories([None; 3]), &Params::default());
        let ranked = ranked(&scores);
        assert_eq!(ranked[0], "improving");
        assert_eq!(ranked[1..], ["flat-low", "saturated"]);
    }

    #[test]
    fn parameters_out_of_range_are_refused() {
        for bad in [
            Params {
                window: 0,
                ..Params::default()
            },
            Params {
                epsilon: 1.5,
                ..Params::default()
            },
            Params {
                epsilon: f64::NAN,
                ..Params::default()
            },
            Params {
                novelty: -0.1,
                ..Params::default()
            },
        ] {
            assert!(bad.validate().is_err(), "{bad:?}");
        }
        assert!(Params::default().validate().is_ok());
    }
}
