// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training loops that roll a model back only on
// evidence, for its clients. If your team needs expertise in sequential
// testing for continuously trained models, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The paired, anytime-valid comparison of a candidate policy with the
//! anchor (the last promoted version) on the same monitor tasks.
//!
//! Each task is attempted once by each policy and graded pass or fail. Only
//! discordant pairs (one passed, the other failed) carry evidence: under no
//! difference the anchor wins a discordant pair with probability one half.
//! Two test martingales run over the discordant pairs, each a likelihood
//! ratio of a fixed alternative to its null, so by Ville's inequality each
//! may be read after every pair and stopped whenever it crosses, with its
//! error rate held:
//!
//! * **regression** - null: no difference (`q = 1/2`); alternative: the
//!   anchor wins `q = 1/2 + delta` of discordant pairs. Crossing `1/alpha`
//!   is evidence the candidate regressed: roll back.
//! * **non-inferiority** - null: the candidate is worse by `delta`
//!   (`q = 1/2 + delta`); alternative: no difference. Crossing `1/beta` is
//!   evidence it is not worse: it may become the anchor.
//!
//! Successive comparisons of one run spend the error rates: the `k`-th (from
//! 0) is held to `alpha 2^-(k+1)` and `beta 2^-(k+1)`, so over every
//! comparison the run makes the chance of a false rollback stays below
//! `alpha` and of a false promotion below `beta`. Neither crossing within the
//! comparison's pair budget decides nothing: training continues.

use serde::{Deserialize, Serialize};

/// What a comparison decided.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Decision {
    /// Not yet decided.
    Undecided,
    /// The candidate regressed by more than `delta`: roll back to the anchor.
    Regressed,
    /// The candidate is not worse than the anchor by `delta`: promote it.
    NonInferior,
    /// The pair budget is spent with neither crossing: continue.
    Inconclusive,
}

/// The registered parameters of the comparisons.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TestSettings {
    /// The least regression worth acting on, as the excess of the anchor's
    /// share of discordant pairs over one half (in `(0, 1/2)`).
    pub delta: f64,
    /// The run's bound on the chance of a false rollback.
    pub alpha: f64,
    /// The run's bound on the chance of a false promotion.
    pub beta: f64,
    /// Discordant pairs a comparison may read before it is inconclusive.
    pub max_discordant: u32,
}

impl TestSettings {
    /// Checks the settings.
    ///
    /// # Errors
    /// A setting out of range.
    pub fn validate(&self) -> Result<(), String> {
        if !(self.delta > 0.0 && self.delta < 0.5) {
            return Err(format!("delta {} is not in (0, 1/2)", self.delta));
        }
        for (name, value) in [("alpha", self.alpha), ("beta", self.beta)] {
            if !(value > 0.0 && value < 1.0) {
                return Err(format!("{name} {value} is not in (0, 1)"));
            }
        }
        if self.max_discordant == 0 {
            return Err("max_discordant is at least 1".into());
        }
        Ok(())
    }
}

/// One comparison's evidence.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Comparison {
    settings: TestSettings,
    /// The comparison's place among the run's, from 0.
    pub index: u32,
    /// Pairs read.
    pub pairs: u32,
    /// Discordant pairs the candidate won.
    pub candidate_wins: u32,
    /// Discordant pairs the anchor won.
    pub anchor_wins: u32,
    /// The regression martingale's value.
    pub regression_e: f64,
    /// The non-inferiority martingale's value.
    pub non_inferiority_e: f64,
    /// What it decided.
    pub decision: Decision,
}

impl Comparison {
    /// The `index`-th comparison of a run held to `settings`.
    #[must_use]
    pub fn new(settings: TestSettings, index: u32) -> Self {
        Self {
            settings,
            index,
            pairs: 0,
            candidate_wins: 0,
            anchor_wins: 0,
            regression_e: 1.0,
            non_inferiority_e: 1.0,
            decision: Decision::Undecided,
        }
    }

    /// The share of the run's error rates this comparison is held to.
    fn spent(&self) -> f64 {
        0.5_f64.powi(
            i32::try_from(self.index)
                .unwrap_or(i32::MAX)
                .saturating_add(1),
        )
    }

    /// The level the regression martingale must reach.
    #[must_use]
    pub fn regression_threshold(&self) -> f64 {
        1.0 / (self.settings.alpha * self.spent())
    }

    /// The level the non-inferiority martingale must reach.
    #[must_use]
    pub fn non_inferiority_threshold(&self) -> f64 {
        1.0 / (self.settings.beta * self.spent())
    }

    /// Reads one pair: whether the candidate and the anchor passed the same
    /// task. Returns the decision so far; a decided comparison reads no
    /// more.
    pub fn read(&mut self, candidate: bool, anchor: bool) -> Decision {
        if self.decision != Decision::Undecided {
            return self.decision;
        }
        self.pairs += 1;
        if candidate == anchor {
            return self.decision;
        }
        let q = 0.5 + self.settings.delta;
        if anchor {
            self.anchor_wins += 1;
            self.regression_e *= q / 0.5;
            self.non_inferiority_e *= 0.5 / q;
        } else {
            self.candidate_wins += 1;
            self.regression_e *= (1.0 - q) / 0.5;
            self.non_inferiority_e *= 0.5 / (1.0 - q);
        }
        self.decision = if self.regression_e >= self.regression_threshold() {
            Decision::Regressed
        } else if self.non_inferiority_e >= self.non_inferiority_threshold() {
            Decision::NonInferior
        } else if self.candidate_wins + self.anchor_wins >= self.settings.max_discordant {
            Decision::Inconclusive
        } else {
            Decision::Undecided
        };
        self.decision
    }

    /// Ends a comparison whose pairs ran out: undecided becomes
    /// inconclusive.
    pub fn close(&mut self) -> Decision {
        if self.decision == Decision::Undecided {
            self.decision = Decision::Inconclusive;
        }
        self.decision
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings() -> TestSettings {
        TestSettings {
            delta: 0.25,
            alpha: 0.05,
            beta: 0.1,
            max_discordant: 60,
        }
    }

    /// A deterministic stream of uniform draws (splitmix).
    fn uniforms(mut state: u64) -> impl FnMut() -> f64 {
        move || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            #[allow(clippy::cast_precision_loss)]
            let u = (z >> 11) as f64 / (1u64 << 53) as f64;
            u
        }
    }

    /// The decision of a comparison where the candidate passes with
    /// probability `c` and the anchor with `a`, independently.
    fn decide(seed: u64, c: f64, a: f64, index: u32) -> Decision {
        let mut draw = uniforms(seed);
        let mut comparison = Comparison::new(settings(), index);
        for _ in 0..400 {
            let decision = comparison.read(draw() < c, draw() < a);
            if decision != Decision::Undecided {
                return decision;
            }
        }
        comparison.close()
    }

    #[test]
    fn a_clear_regression_rolls_back_an_equal_candidate_is_promoted_and_concordant_pairs_say_nothing(
    ) {
        // seed: each assertion is a frequency over 200 seeded comparisons; the seeds are the trials.
        let regressed = (0..200)
            .filter(|s| decide(*s, 0.2, 0.8, 0) == Decision::Regressed)
            .count();
        assert!(regressed >= 190, "{regressed} of 200");
        let promoted = (0..200)
            .filter(|s| decide(1_000 + *s, 0.6, 0.6, 0) == Decision::NonInferior)
            .count();
        assert!(promoted >= 150, "{promoted} of 200");
        let mut comparison = Comparison::new(settings(), 0);
        for _ in 0..1_000 {
            comparison.read(true, true);
            comparison.read(false, false);
        }
        assert_eq!(comparison.decision, Decision::Undecided);
        assert_eq!(comparison.regression_e, 1.0);
    }

    #[test]
    fn false_rollbacks_stay_below_alpha_and_later_comparisons_are_held_to_less() {
        // seed: the false-rollback rate is a frequency over 2000 seeded null comparisons.
        let false_rollbacks = (0..2_000)
            .filter(|s| decide(5_000 + *s, 0.5, 0.5, 0) == Decision::Regressed)
            .count();
        assert!(
            false_rollbacks <= 2_000 * 5 / 100 / 2,
            "{false_rollbacks} of 2000 under no difference, alpha spent 1/2"
        );
        let first = Comparison::new(settings(), 0);
        let third = Comparison::new(settings(), 2);
        assert!((first.regression_threshold() - 40.0).abs() < 1e-9);
        assert!((third.regression_threshold() - 160.0).abs() < 1e-9);
        assert!(TestSettings {
            delta: 0.5,
            ..settings()
        }
        .validate()
        .is_err());
        assert!(TestSettings {
            alpha: 0.0,
            ..settings()
        }
        .validate()
        .is_err());
        assert!(settings().validate().is_ok());
    }
}
