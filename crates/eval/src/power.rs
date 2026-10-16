// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements power planning for paired model
// examinations, for its clients. If your team needs expertise in sizing an
// exam so that a real gain is seen and luck is not, you can procure our
// services by sending an email to info@swedishembedded.com.

//! How much an exam of independent units can see: the exact power of the
//! one-sided sign test, the smallest effect it detects, the level each of
//! a sequence of generations may spend, and the verdict that keeps a miss
//! on an exam too small to see the effect from being read as a failure.
//!
//! The unit is the family (a cluster of tasks counts once), not the task.
//! Each unit is a candidate win with probability `(d + e) / 2`, a baseline
//! win with `(d - e) / 2` and a tie otherwise, for discordance `d` and
//! effect `e` (the lead in the share of units won). The test is the
//! handed-in [`Significance`], so the critical number of wins is whatever
//! the gate's own test says, not a second copy of its arithmetic; only the
//! binomial arithmetic of the alternative is local.

use serde::{Deserialize, Serialize};

use crate::gate::{self, Check, Improvement, SuiteSummary};
use crate::paired::{by_cluster, PairedOutcome};
use crate::significance::{SignTest, Significance};

/// The power an exam is asked to have: four chances in five.
pub const DEFAULT_POWER: f64 = 0.8;

/// What a power calculation is about.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SignPlan {
    /// Independent units (families).
    pub units: usize,
    /// The share of units on which the arms differ.
    pub discordance: f64,
    /// The candidate's lead in the share of units won; at most `discordance`.
    pub effect: f64,
    /// The level the test is held to.
    pub alpha: f64,
}

/// The fewest wins out of `discordant` discordant units that the test calls
/// significant at `alpha`; `None` when not even all of them do (a test of
/// four units cannot reach 0.05).
#[must_use]
pub fn critical_wins(discordant: usize, alpha: f64, test: &dyn Significance) -> Option<usize> {
    let significant = |wins: usize| {
        let pairs: Vec<(bool, bool)> = (0..discordant).map(|i| (i < wins, i >= wins)).collect();
        test.sign_test(&pairs).p_value <= alpha
    };
    if discordant == 0 || !significant(discordant) {
        return None;
    }
    // The p-value falls as wins rise, so the boundary is found by bisection.
    let (mut low, mut high) = (0, discordant);
    while low < high {
        let mid = (low + high) / 2;
        if significant(mid) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    Some(low)
}

/// `ln(n!)` for `n` up to `max`.
fn ln_factorials(max: usize) -> Vec<f64> {
    let mut table = vec![0.0; max + 1];
    for n in 2..=max {
        table[n] = table[n - 1] + (n as f64).ln();
    }
    table
}

/// `P(Binomial(n, p) = k)` from a table of log factorials.
fn binomial_pmf(table: &[f64], n: usize, k: usize, p: f64) -> f64 {
    if p <= 0.0 {
        return f64::from(k == 0);
    }
    if p >= 1.0 {
        return f64::from(k == n);
    }
    let ln =
        table[n] - table[k] - table[n - k] + k as f64 * p.ln() + (n - k) as f64 * (1.0 - p).ln();
    ln.exp()
}

/// The exact chance that the test calls `plan` significant: over the number
/// of discordant units (binomial in the units and the discordance), the
/// chance that at least the critical number of them are candidate wins.
#[must_use]
pub fn power(plan: &SignPlan, test: &dyn Significance) -> f64 {
    let d = plan.discordance.clamp(0.0, 1.0);
    if plan.units == 0 || d == 0.0 {
        return 0.0;
    }
    let share_won = ((d + plan.effect.min(d)) / (2.0 * d)).clamp(0.0, 1.0);
    let table = ln_factorials(plan.units);
    (1..=plan.units)
        .filter_map(|m| {
            let critical = critical_wins(m, plan.alpha, test)?;
            let reject: f64 = (critical..=m)
                .map(|k| binomial_pmf(&table, m, k, share_won))
                .sum();
            Some(binomial_pmf(&table, plan.units, m, d) * reject)
        })
        .sum::<f64>()
        .min(1.0)
}

/// The smallest lead, in the share of units won, that `units` units at
/// `discordance` show with probability `wanted_power`; `None` when even a
/// lead as large as the discordance (every discordant unit won by the
/// candidate) falls short.
#[must_use]
pub fn minimum_detectable_effect(
    units: usize,
    discordance: f64,
    alpha: f64,
    wanted_power: f64,
    test: &dyn Significance,
) -> Option<f64> {
    let at = |effect: f64| {
        power(
            &SignPlan {
                units,
                discordance,
                effect,
                alpha,
            },
            test,
        )
    };
    let ceiling = discordance.clamp(0.0, 1.0);
    if at(ceiling) < wanted_power {
        return None;
    }
    // Power rises with the effect: bisect for the smallest that has enough.
    let (mut low, mut high) = (0.0, ceiling);
    for _ in 0..50 {
        let mid = (low + high) / 2.0;
        if at(mid) >= wanted_power {
            high = mid;
        } else {
            low = mid;
        }
    }
    Some(high)
}

/// The level generation `generation` (counted from 1) of a sequence of
/// decisions may spend, out of a total `alpha` for the whole sequence:
/// `alpha / (g (g + 1))`. The levels sum to `alpha * G / (G + 1)` over `G`
/// generations, so however long the sequence runs the chance that any
/// generation releases a candidate no better than its champion stays below
/// `alpha`, where ten tests at `alpha` each release one with probability up
/// to `1 - (1 - alpha)^10`. The schedule is fixed in advance and needs no
/// horizon. Generation 0 is treated as the first.
#[must_use]
pub fn spent_alpha(alpha: f64, generation: u32) -> f64 {
    let g = f64::from(generation.max(1));
    alpha / (g * (g + 1.0))
}

/// What a sign test on an exam of `units` independent units concluded.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub enum Verdict {
    /// The test rejected equality at its level.
    Significant,
    /// The test did not reject, and the exam had the power to see the
    /// effect it was asked about: no evidence of the effect.
    NotSignificant,
    /// The test did not reject, but the exam could not have seen the effect
    /// asked about with the wanted power: neither a pass nor a fail.
    Inconclusive {
        /// The power at the effect asked about and the observed discordance.
        power: f64,
        /// The smallest effect the exam could have seen; `None` when none
        /// is reachable with this many units.
        minimum_detectable_effect: Option<f64>,
    },
}

impl Verdict {
    /// The verdict on `test` over `units` units (families) at `alpha`, given
    /// the effect the exam is asked to detect. Without one the exam is not
    /// held to a power, and a miss is a plain non-rejection.
    #[must_use]
    pub fn of(
        test: &SignTest,
        units: usize,
        alpha: f64,
        target_effect: Option<f64>,
        significance: &dyn Significance,
    ) -> Self {
        if test.p_value <= alpha {
            return Self::Significant;
        }
        let Some(effect) = target_effect else {
            return Self::NotSignificant;
        };
        let discordance = if units == 0 {
            0.0
        } else {
            test.discordant as f64 / units as f64
        };
        let reached = power(
            &SignPlan {
                units,
                discordance,
                effect,
                alpha,
            },
            significance,
        );
        // An effect beyond the discordance cannot be produced by these
        // units at all; `power` caps it, so compare the asked effect too.
        if reached >= DEFAULT_POWER && effect <= discordance {
            Self::NotSignificant
        } else {
            Self::Inconclusive {
                power: reached,
                minimum_detectable_effect: minimum_detectable_effect(
                    units,
                    discordance,
                    alpha,
                    DEFAULT_POWER,
                    significance,
                ),
            }
        }
    }
}

/// How the improvement check is held to a level and to a power.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rule {
    /// The level of the whole sequence of releases.
    pub alpha: f64,
    /// The 1-based place of this decision in the lineage's sequence, when
    /// the level is spent across it.
    pub generation: Option<u32>,
    /// The lead the exam is asked to be able to see.
    pub target_effect: Option<f64>,
}

/// What the improvement check recorded of its level and power.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PowerAssessment {
    /// Independent units (families) the test rests on, ties included.
    pub units: usize,
    /// The level of the whole sequence.
    pub nominal_alpha: f64,
    /// The generation whose share was spent, when it was spent.
    pub generation: Option<u32>,
    /// The lead asked for.
    pub target_effect: Option<f64>,
    /// What the test concluded.
    pub verdict: Verdict,
}

/// [`gate::improvement`] held to `rule`: at the level this generation may
/// spend, and, when a target lead is named, with a miss on an exam too small
/// to see it called inconclusive and not a failure. A rejection is a pass
/// whatever the power: the p-value is valid at its level however few units
/// there are.
#[must_use]
pub fn judged_improvement(
    suite: SuiteSummary,
    variants: Option<SuiteSummary>,
    outcomes: &[PairedOutcome],
    rule: Rule,
    significance: &dyn Significance,
) -> Check<Improvement> {
    let level = rule
        .generation
        .map_or(rule.alpha, |g| spent_alpha(rule.alpha, g));
    let mut check = gate::improvement(suite, variants, outcomes, level, significance);
    let Some(measured) = check.measured.as_mut() else {
        return check;
    };
    let units = by_cluster(outcomes).len();
    let verdict = Verdict::of(
        &measured.sign_test,
        units,
        level,
        rule.target_effect,
        significance,
    );
    if let Verdict::Inconclusive {
        power,
        minimum_detectable_effect,
    } = verdict
    {
        check.inconclusive = true;
        check.reason = Some(format!(
            "inconclusive: {units} independent unit(s) give {:.0}% power to see a lead of {:.2} \
             at level {level:.4}, and the test did not reject (smallest lead it could see: {})",
            power * 100.0,
            rule.target_effect.unwrap_or_default(),
            minimum_detectable_effect
                .map_or_else(|| "none with these units".into(), |e| format!("{e:.2}")),
        ));
    }
    measured.power = Some(PowerAssessment {
        units,
        nominal_alpha: rule.alpha,
        generation: rule.generation,
        target_effect: rule.target_effect,
        verdict,
    });
    check
}

#[cfg(test)]
mod tests;
