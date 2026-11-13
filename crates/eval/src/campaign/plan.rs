// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for running model-improvement
// experiments whose exam can see the effect they aim for, for its clients. If
// your team needs expertise in power analysis and sequential testing for
// model gates you can procure our services by sending an email to
// info@swedishembedded.com.

//! Power-first planning: an arm is planned against the exam that will judge
//! it, and refused when that exam could not detect the effect it targets.
//!
//! A null result from an exam that could not have seen the effect says
//! nothing, and a GPU-hour spent to obtain one is lost. Before an arm runs,
//! the campaign states the effect it hopes for (the candidate's lead, in the
//! share of exam units it wins), asks what the exam can detect at the level
//! the arm may spend, and refuses the arm when the smallest detectable lead
//! exceeds the target ([`plan_arm`]). The level is spent by the generation's
//! place in the campaign ([`spent_alpha`]), so ten gates in a row release a
//! null candidate at most [`DEFAULT_TOTAL_ALPHA`] of the time instead of 40
//! percent: sequential spending is the default for a campaign, not an opt-in.

use serde::{Deserialize, Serialize};

use crate::power::{minimum_detectable_effect, power, SignPlan};
pub use crate::power::{spent_alpha, DEFAULT_POWER};
use crate::significance::Significance;

/// The level a whole sequence of gates may spend in total.
pub const DEFAULT_TOTAL_ALPHA: f64 = 0.05;

/// The exam an arm will be judged on.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExamShape {
    /// Independent units (families, not tasks).
    pub units: usize,
    /// The share of units on which a candidate and its champion are expected
    /// to differ: measured from an earlier pair, or a stated pilot guess.
    pub discordance: f64,
}

/// What an arm is asked to do.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArmGoal {
    /// The lead, in the share of exam units won, the arm is meant to produce.
    pub target_effect: f64,
    /// Where this arm falls in the campaign's sequence of gates, from 1.
    pub generation: u32,
    /// The level the whole sequence spends.
    pub total_alpha: f64,
    /// The chance the exam is required to have of seeing the target effect.
    pub wanted_power: f64,
}

impl ArmGoal {
    /// A goal of `target_effect` for gate `generation`, at the default total
    /// level and power.
    #[must_use]
    pub fn new(target_effect: f64, generation: u32) -> Self {
        Self {
            target_effect,
            generation,
            total_alpha: DEFAULT_TOTAL_ALPHA,
            wanted_power: DEFAULT_POWER,
        }
    }
}

/// An arm that may run.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlannedArm {
    /// The level this arm's gate may spend.
    pub alpha: f64,
    /// The smallest lead its exam detects at the wanted power.
    pub minimum_detectable_effect: f64,
    /// The exam's power against the target effect.
    pub power_at_target: f64,
}

/// An arm that must not run.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum ArmRefused {
    /// The plan is not a plan: a target outside `(0, 1]`, no units, a
    /// discordance outside `(0, 1]`, a level or power outside `(0, 1)`, or a
    /// target larger than the discordance (a candidate cannot win more units
    /// than differ).
    #[error("the plan is invalid: {0}")]
    Invalid(String),
    /// The exam cannot detect the target effect.
    #[error(
        "the exam cannot detect the target: {units} units at discordance {discordance} detect a \
         lead of {detectable} at power {wanted_power} and level {alpha:.4}, and the arm targets {target}; \
         widen the exam or lower the claim"
    )]
    Underpowered {
        /// Units in the exam.
        units: usize,
        /// The discordance assumed.
        discordance: f64,
        /// The level of this gate.
        alpha: f64,
        /// The power asked for.
        wanted_power: f64,
        /// The target effect.
        target: f64,
        /// What the exam detects; `none: even a clean sweep falls short`
        /// when no lead is detectable.
        detectable: String,
    },
}

/// Plans an arm judged on `exam`, or refuses it.
///
/// # Errors
/// [`ArmRefused::Invalid`] for an impossible plan; [`ArmRefused::Underpowered`]
/// when the exam's smallest detectable lead at this generation's spent level
/// exceeds the target.
pub fn plan_arm(
    exam: &ExamShape,
    goal: &ArmGoal,
    test: &dyn Significance,
) -> Result<PlannedArm, ArmRefused> {
    let invalid = |why: &str| Err(ArmRefused::Invalid(why.to_string()));
    let open_unit = |x: f64| x.is_finite() && x > 0.0 && x <= 1.0;
    if exam.units == 0 {
        return invalid("the exam has no units");
    }
    if !open_unit(exam.discordance) {
        return invalid("the discordance must lie in (0, 1]");
    }
    if !open_unit(goal.target_effect) {
        return invalid("the target effect must lie in (0, 1]");
    }
    if goal.target_effect > exam.discordance {
        return invalid("the target effect exceeds the discordance: a candidate cannot win more units than differ");
    }
    if !(goal.total_alpha > 0.0
        && goal.total_alpha < 1.0
        && goal.wanted_power > 0.0
        && goal.wanted_power < 1.0)
    {
        return invalid("the level and the power must lie in (0, 1)");
    }
    if goal.generation == 0 {
        return invalid("gates are counted from 1");
    }
    let alpha = spent_alpha(goal.total_alpha, goal.generation);
    let detectable =
        minimum_detectable_effect(exam.units, exam.discordance, alpha, goal.wanted_power, test);
    match detectable {
        Some(mde) if mde <= goal.target_effect => Ok(PlannedArm {
            alpha,
            minimum_detectable_effect: mde,
            power_at_target: power(
                &SignPlan {
                    units: exam.units,
                    discordance: exam.discordance,
                    effect: goal.target_effect,
                    alpha,
                },
                test,
            ),
        }),
        other => Err(ArmRefused::Underpowered {
            units: exam.units,
            discordance: exam.discordance,
            alpha,
            wanted_power: goal.wanted_power,
            target: goal.target_effect,
            detectable: other.map_or_else(
                || "none: even a clean sweep falls short".to_string(),
                |mde| format!("{mde:.3}"),
            ),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::significance::SignTest;

    /// An exact one-sided binomial sign test, as the model backend supplies.
    struct Exact;

    impl Significance for Exact {
        fn sign_test(&self, pairs: &[(bool, bool)]) -> SignTest {
            let wins = pairs.iter().filter(|(c, b)| *c && !*b).count();
            let losses = pairs.iter().filter(|(c, b)| !*c && *b).count();
            let n = wins + losses;
            let mut p = 0.0;
            for k in wins..=n {
                let mut c = 1.0_f64;
                for i in 0..k {
                    c = c * (n - i) as f64 / (i + 1) as f64;
                }
                p += c * 0.5_f64.powi(n as i32);
            }
            SignTest {
                discordant: n,
                candidate_wins: wins,
                p_value: if n == 0 { 1.0 } else { p },
            }
        }
    }

    fn exam(units: usize) -> ExamShape {
        ExamShape {
            units,
            discordance: 0.5,
        }
    }

    #[test]
    fn an_exam_that_cannot_see_the_target_refuses_the_arm_before_it_runs() {
        let refused = plan_arm(&exam(20), &ArmGoal::new(0.1, 1), &Exact).unwrap_err();
        assert!(
            matches!(refused, ArmRefused::Underpowered { units: 20, .. }),
            "{refused:?}"
        );
        assert!(refused.to_string().contains("widen the exam"));
    }

    #[test]
    fn an_exam_that_can_see_it_plans_the_arm_at_the_spent_level() {
        let goal = ArmGoal {
            wanted_power: 0.8,
            ..ArmGoal::new(0.3, 1)
        };
        let planned = plan_arm(&exam(400), &goal, &Exact).unwrap();
        assert_eq!(planned.alpha, spent_alpha(DEFAULT_TOTAL_ALPHA, 1));
        assert!(planned.minimum_detectable_effect <= 0.3);
        assert!(planned.power_at_target >= 0.8, "{planned:?}");
    }

    #[test]
    fn later_gates_spend_less_alpha_so_the_same_exam_may_stop_sufficing() {
        let goal = |g| ArmGoal::new(0.12, g);
        let first = plan_arm(&exam(300), &goal(1), &Exact);
        let tenth = plan_arm(&exam(300), &goal(10), &Exact);
        assert!(first.is_ok(), "{first:?}");
        let (first, tenth_planned) = (first.unwrap(), tenth.as_ref().ok());
        if let Some(tenth) = tenth_planned {
            assert!(tenth.alpha < first.alpha);
            assert!(tenth.minimum_detectable_effect >= first.minimum_detectable_effect);
        } else {
            assert!(matches!(tenth, Err(ArmRefused::Underpowered { .. })));
        }
        assert!(spent_alpha(0.05, 10) < spent_alpha(0.05, 1));
    }

    #[test]
    fn an_impossible_plan_is_invalid_not_underpowered() {
        for (shape, goal) in [
            (exam(0), ArmGoal::new(0.1, 1)),
            (
                ExamShape {
                    units: 100,
                    discordance: 0.0,
                },
                ArmGoal::new(0.1, 1),
            ),
            (exam(100), ArmGoal::new(0.0, 1)),
            (exam(100), ArmGoal::new(0.9, 1)),
            (exam(100), ArmGoal::new(0.1, 0)),
            (
                exam(100),
                ArmGoal {
                    total_alpha: 1.5,
                    ..ArmGoal::new(0.1, 1)
                },
            ),
        ] {
            assert!(
                matches!(plan_arm(&shape, &goal, &Exact), Err(ArmRefused::Invalid(_))),
                "{shape:?} {goal:?}"
            );
        }
    }
}
