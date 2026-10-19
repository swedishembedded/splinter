// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the choice between retraining, a better
// harness, more search or no change at all, on measured cost-adjusted gain,
// for its clients. If your team needs expertise in deciding how an agent
// should be improved before spending a training budget, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Choosing how to improve an agent from what each way was measured to do.
//! Pure: no model, no files.
//!
//! An intervention (more samples, another harness, a teacher's
//! demonstrations and training, ...) is measured against the incumbent on the
//! same target tasks. Its evidence is a paired result per **unit** - a family
//! of tasks, the unit of independent evidence - the arm's success rate in the
//! unit minus the incumbent's ([`ArmResult::deltas`]), and what it costs: a
//! one-time cost (training) and a per-use cost (extra inference on every
//! task it is deployed on), both in the same unit, seconds.
//!
//! The decision is a rule over three measured things, written down so a
//! reader can check it:
//!
//! 1. **Utility** - the mean of the unit deltas and its standard error over
//!    units. An arm is judged by its lower confidence bound,
//!    `mean - z * se`: an arm that looks good because few units were
//!    measured does not win.
//! 2. **Cost** - `one_time + expected_uses * per_use`.
//! 3. **Net value** - `expected_uses * value_per_success * lcb - cost`. The
//!    pricing says what a verified success is worth in seconds and how many
//!    uses the choice is amortised over.
//!
//! The arm with the greatest net value is chosen when that value is above
//! zero; otherwise nothing changes, which is itself a measured, priced
//! outcome. An arm measured on fewer than `min_units` units, or over the
//! budget, is not eligible; it is reported with its reason, never scored as
//! zero.

use serde::{Deserialize, Serialize};

/// What an intervention measured against the incumbent.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArmResult {
    /// Which intervention.
    pub label: String,
    /// Per unit: the arm's success rate minus the incumbent's, each in
    /// `[-1, 1]`. A unit nobody measured is absent, not `0`.
    pub deltas: Vec<f64>,
    /// Seconds spent once to have the arm (training); `0` for an arm that
    /// needs none.
    pub one_time_cost: f64,
    /// Seconds the arm adds to every use over the incumbent's.
    pub per_use_cost: f64,
}

/// What a success is worth, how long the choice is amortised over, and how
/// sure an arm must be.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Pricing {
    /// What one more verified success is worth, in seconds.
    pub value_per_success: f64,
    /// The uses the choice is amortised over.
    pub expected_uses: f64,
    /// Standard errors subtracted from the mean gain.
    pub confidence_z: f64,
    /// The fewest units an arm must be measured on to be eligible.
    pub min_units: usize,
    /// The most an arm's total cost may be, in seconds; `None` sets none.
    pub budget: Option<f64>,
}

/// One-sided 95 percent: the z of the lower confidence bound.
pub const DEFAULT_CONFIDENCE_Z: f64 = 1.645;

/// Why an arm was not scored.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "why", rename_all = "snake_case")]
pub enum Ineligible {
    /// Measured on too few units.
    TooFewUnits {
        /// Units measured.
        units: usize,
        /// Units required.
        required: usize,
    },
    /// Costs more than the budget.
    OverBudget,
    /// A number that is not finite was given.
    NotFinite,
}

/// How one arm scored.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Score {
    /// Which intervention.
    pub label: String,
    /// Units measured.
    pub units: usize,
    /// The mean gain over the incumbent; `None` when nothing was measured.
    pub mean_gain: Option<f64>,
    /// Its standard error over units; `None` with fewer than two units.
    pub standard_error: Option<f64>,
    /// The lower confidence bound of the gain.
    pub lower_bound: Option<f64>,
    /// Total cost, in seconds.
    pub cost: f64,
    /// Net value, in seconds; `None` for an ineligible arm.
    pub net_value: Option<f64>,
    /// Why it was not scored, when it was not.
    pub ineligible: Option<Ineligible>,
}

/// What was decided.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "decision", rename_all = "snake_case")]
pub enum Decision {
    /// Apply this intervention.
    Apply {
        /// Which.
        label: String,
        /// Its net value, in seconds.
        net_value: f64,
    },
    /// Change nothing, and why.
    NoChange {
        /// Why.
        reason: String,
    },
}

/// The decision and every arm's score.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Verdict {
    /// What was decided.
    pub decision: Decision,
    /// Each arm, in the order given.
    pub scores: Vec<Score>,
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// The standard error of the mean of `values`; `None` with fewer than two.
fn standard_error(values: &[f64]) -> Option<f64> {
    let n = values.len();
    if n < 2 {
        return None;
    }
    let m = mean(values);
    let variance = values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n - 1) as f64;
    Some((variance / n as f64).sqrt())
}

/// How `arm` scores under `pricing`.
#[must_use]
pub fn score(arm: &ArmResult, pricing: &Pricing) -> Score {
    let units = arm.deltas.len();
    let cost = arm.one_time_cost + pricing.expected_uses * arm.per_use_cost;
    let mut score = Score {
        label: arm.label.clone(),
        units,
        mean_gain: None,
        standard_error: None,
        lower_bound: None,
        cost,
        net_value: None,
        ineligible: None,
    };
    let finite = arm.deltas.iter().all(|d| d.is_finite()) && cost.is_finite();
    if !finite {
        score.ineligible = Some(Ineligible::NotFinite);
        return score;
    }
    if units > 0 {
        score.mean_gain = Some(mean(&arm.deltas));
        score.standard_error = standard_error(&arm.deltas);
    }
    if units < pricing.min_units.max(1) {
        score.ineligible = Some(Ineligible::TooFewUnits {
            units,
            required: pricing.min_units.max(1),
        });
        return score;
    }
    if pricing.budget.is_some_and(|budget| cost > budget) {
        score.ineligible = Some(Ineligible::OverBudget);
        return score;
    }
    // One unit has no spread to speak of: its bound is its mean, which
    // `min_units` exists to refuse.
    let gain = score.mean_gain.unwrap_or_default();
    let bound = gain - pricing.confidence_z * score.standard_error.unwrap_or_default();
    score.lower_bound = Some(bound);
    score.net_value = Some(pricing.expected_uses * pricing.value_per_success * bound - cost);
    score
}

/// Chooses among `arms` under `pricing`; see the module documentation.
#[must_use]
pub fn decide(arms: &[ArmResult], pricing: &Pricing) -> Verdict {
    let scores: Vec<Score> = arms.iter().map(|arm| score(arm, pricing)).collect();
    let best = scores
        .iter()
        .filter_map(|s| s.net_value.map(|v| (s, v)))
        .max_by(|a, b| a.1.total_cmp(&b.1));
    let decision = match best {
        Some((s, net_value)) if net_value > 0.0 => Decision::Apply {
            label: s.label.clone(),
            net_value,
        },
        Some(_) => Decision::NoChange {
            reason: "no measured intervention is worth its cost at the stated pricing".into(),
        },
        None => Decision::NoChange {
            reason: "no intervention was measured on enough units to be eligible".into(),
        },
    };
    Verdict { decision, scores }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PRICING: Pricing = Pricing {
        value_per_success: 100.0,
        expected_uses: 1000.0,
        confidence_z: DEFAULT_CONFIDENCE_Z,
        min_units: 4,
        budget: None,
    };

    fn arm(label: &str, deltas: &[f64], one_time: f64, per_use: f64) -> ArmResult {
        ArmResult {
            label: label.into(),
            deltas: deltas.to_vec(),
            one_time_cost: one_time,
            per_use_cost: per_use,
        }
    }

    fn applied(verdict: &Verdict) -> Option<&str> {
        match &verdict.decision {
            Decision::Apply { label, .. } => Some(label),
            Decision::NoChange { .. } => None,
        }
    }

    #[test]
    fn the_arm_with_the_greatest_net_value_is_chosen() {
        let verdict = decide(
            &[
                arm("cheap-small", &[0.1, 0.1, 0.1, 0.1], 0.0, 0.01),
                arm("dear-large", &[0.5, 0.5, 0.5, 0.5], 20_000.0, 0.0),
            ],
            &PRICING,
        );
        // 1000*100*0.1 - 10 = 9990 against 1000*100*0.5 - 20000 = 30000.
        assert_eq!(applied(&verdict), Some("dear-large"));
    }

    #[test]
    fn an_arm_is_judged_by_its_lower_bound_not_its_mean() {
        // Same mean 0.3; the second is all over the place.
        let steady = arm("steady", &[0.3, 0.3, 0.3, 0.3], 0.0, 0.0);
        let noisy = arm("noisy", &[0.9, -0.3, 0.9, -0.3], 0.0, 0.0);
        let verdict = decide(&[noisy, steady], &PRICING);
        assert_eq!(applied(&verdict), Some("steady"));
        let noisy = &verdict.scores[0];
        assert!(noisy.lower_bound.unwrap() < noisy.mean_gain.unwrap());
    }

    #[test]
    fn nothing_changes_when_no_arm_pays_for_itself() {
        let verdict = decide(&[arm("meagre", &[0.01; 6], 5_000.0, 0.0)], &PRICING);
        assert_eq!(applied(&verdict), None);
        assert!(verdict.scores[0].net_value.unwrap() < 0.0);
    }

    #[test]
    fn an_arm_measured_on_too_few_units_or_over_budget_is_not_scored_as_zero() {
        let pricing = Pricing {
            budget: Some(1_000.0),
            ..PRICING
        };
        let verdict = decide(
            &[
                arm("few", &[0.9, 0.9], 0.0, 0.0),
                arm("dear", &[0.9; 6], 2_000.0, 0.0),
                arm("none", &[], 0.0, 0.0),
            ],
            &pricing,
        );
        assert_eq!(applied(&verdict), None);
        assert_eq!(
            verdict.scores[0].ineligible,
            Some(Ineligible::TooFewUnits {
                units: 2,
                required: 4
            })
        );
        assert_eq!(verdict.scores[1].ineligible, Some(Ineligible::OverBudget));
        assert!(verdict.scores[2].mean_gain.is_none());
        assert!(verdict.scores.iter().all(|s| s.net_value.is_none()));
    }

    #[test]
    fn a_number_that_is_not_finite_makes_an_arm_ineligible() {
        let verdict = decide(&[arm("nan", &[f64::NAN; 5], 0.0, 0.0)], &PRICING);
        assert_eq!(verdict.scores[0].ineligible, Some(Ineligible::NotFinite));
    }
}
