// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pre-registered release requirements for
// time-to-event risk models, written down before a candidate is scored. If
// your team needs expertise in validating risk models against a champion on
// held-out participants, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The names a timeline model's measurements travel under, and the
//! requirements written against them.
//!
//! The scorer (the model adapter, which alone can compute survival metrics)
//! files every number under a name from this module; the requirements built
//! here refer to the same names, so a number and the requirement it is judged
//! by cannot drift apart. A [`TimelinePlan`] is the pre-registration: data,
//! written before the candidate is scored, turned by [`TimelinePlan::spec`]
//! into the [`PredictiveSpec`] the predictive gate decides. A horizon that was
//! not measured (too few events) has no name in the evidence, so a requirement
//! on it fails as unmeasured and is never read as zero.

use serde::{Deserialize, Serialize};

use crate::metric_gate::Requirement;
use crate::predictive_gate::PredictiveSpec;

/// The name of the serving-identity value: the largest absolute difference
/// between what the shipped file predicts and what the model that was scored
/// predicts (and, for the model as trained, on its probe subjects).
pub const SERVE_MAX_ABS_DIFF: &str = "serve_max_abs_diff";
/// The largest absolute difference between a batched forecast and the same
/// history forecast alone.
pub const SERVE_BATCH_MAX_ABS_DIFF: &str = "serve_batch_max_abs_diff";
/// The share of held-out units the model's support would withhold an answer
/// for.
pub const SERVE_ABSTENTION_RATE: &str = "serve_abstention_rate";
/// How many served probabilities are not finite or not inside [0, 1].
pub const SERVE_INVALID_PROBABILITIES: &str = "serve_invalid_probabilities";
/// How many served curves step the wrong way.
pub const SERVE_NON_MONOTONE_CURVES: &str = "serve_non_monotone_curves";
/// The name of the held-out event negative log-likelihood difference.
pub const NLL_DIFF: &str = "diff:nll";

/// Which model a number belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Arm {
    /// The model in place.
    Champion,
    /// The model that would replace it.
    Candidate,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Arm::Champion => "champion",
            Arm::Candidate => "candidate",
        }
    }
}

fn at(horizon: Option<f64>) -> String {
    horizon.map_or_else(String::new, |h| format!(":h{h}"))
}

/// An arm's own metric (`uno_c`, `auc`, `brier`, `slope`, `intercept`, `oe`,
/// `ece`, `ibs`) for `code`, at `horizon` or over the whole grid (`None`).
#[must_use]
pub fn arm_metric(arm: Arm, code: &str, horizon: Option<f64>, metric: &str) -> String {
    format!("{}:{code}{}:{metric}", arm.name(), at(horizon))
}

/// The held-out event negative log-likelihood of an arm.
#[must_use]
pub fn arm_nll(arm: Arm) -> String {
    format!("{}:nll", arm.name())
}

/// The candidate-minus-champion difference interval of `metric` for `code`.
#[must_use]
pub fn diff_metric(code: &str, horizon: Option<f64>, metric: &str) -> String {
    format!("diff:{code}{}:{metric}", at(horizon))
}

/// The candidate-minus-champion integrated Brier difference on one subgroup
/// (a point value: the retention check reads the worst of them).
#[must_use]
pub fn subgroup_diff(subgroup: &str, code: &str) -> String {
    format!("subgroup:{subgroup}:{code}")
}

/// The number of events of `code` in a subgroup within the grid, which a
/// subgroup comparison needs before it counts.
#[must_use]
pub fn subgroup_events(subgroup: &str, code: &str) -> String {
    format!("subgroup_events:{subgroup}:{code}")
}

/// Bands the candidate's own calibration must lie in at every judged horizon.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CalibrationBands {
    /// The recalibration slope (1 when calibrated).
    pub slope: (f64, f64),
    /// The recalibration intercept (0 when calibrated).
    pub intercept: (f64, f64),
    /// Observed over expected (1 when calibrated).
    pub oe: (f64, f64),
    /// The most the expected calibration error may be.
    pub max_ece: f64,
}

/// What is judged for one outcome code.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CodePlan {
    /// The outcome code, or the all-cause name the scorer was given.
    pub code: String,
    /// The horizons its calibration is judged at.
    pub horizons: Vec<f64>,
    /// Whether the candidate must beat the champion on this code's
    /// integrated Brier score (the interval of the difference must exclude
    /// zero on the better side).
    pub must_improve: bool,
}

/// The pre-registered requirements of a timeline release.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TimelinePlan {
    /// Each judged code.
    pub codes: Vec<CodePlan>,
    /// Whether the held-out event likelihood must improve too.
    pub improve_nll: bool,
    /// The calibration bands.
    pub calibration: CalibrationBands,
    /// The named subgroups no code may regress on.
    pub subgroups: Vec<String>,
    /// How much worse a subgroup's integrated Brier score may get.
    pub subgroup_margin: f64,
    /// The fewest events a subgroup needs for its comparison to count.
    pub min_subgroup_events: usize,
    /// The largest tolerated difference between the shipped file's predictions
    /// and the scored model's.
    pub serve_tolerance: f64,
    /// The largest tolerated difference between a batched forecast and a
    /// single one.
    pub batch_tolerance: f64,
    /// The largest share of held-out units the support may withhold an answer
    /// for: a model whose own test units mostly lie outside what it was
    /// trained on is not served correctly, however well it scores on the
    /// rest. Registered before scoring, like every band.
    pub max_abstention_rate: f64,
}

/// A value that must lie in `[0, hi]`; absent or NaN fails.
fn within(value: &str, hi: f64) -> Requirement {
    Requirement::Within {
        value: value.into(),
        lo: 0.0,
        hi,
    }
}

impl TimelinePlan {
    /// The requirements this plan registers, by check.
    #[must_use]
    pub fn spec(&self) -> PredictiveSpec {
        let mut performance = Vec::new();
        for code in self.codes.iter().filter(|c| c.must_improve) {
            performance.push(Requirement::Improves {
                interval: diff_metric(&code.code, None, "ibs"),
                lower_is_better: true,
            });
        }
        if self.improve_nll {
            performance.push(Requirement::Improves {
                interval: NLL_DIFF.into(),
                lower_is_better: true,
            });
        }
        let bands = &self.calibration;
        let mut calibration = Vec::new();
        for code in &self.codes {
            for h in &code.horizons {
                let within = |metric: &str, (lo, hi): (f64, f64)| Requirement::Within {
                    value: arm_metric(Arm::Candidate, &code.code, Some(*h), metric),
                    lo,
                    hi,
                };
                calibration.push(within("slope", bands.slope));
                calibration.push(within("intercept", bands.intercept));
                calibration.push(within("oe", bands.oe));
                calibration.push(within("ece", (0.0, bands.max_ece)));
            }
        }
        let mut retention = Vec::new();
        for subgroup in &self.subgroups {
            for code in &self.codes {
                retention.push(Requirement::Within {
                    value: subgroup_events(subgroup, &code.code),
                    lo: self.min_subgroup_events as f64,
                    hi: f64::MAX,
                });
            }
        }
        if !self.subgroups.is_empty() {
            retention.push(Requirement::NotWorseBy {
                prefix: "subgroup:".into(),
                bound: self.subgroup_margin,
                lower_is_better: true,
            });
        }
        PredictiveSpec {
            performance,
            calibration,
            retention,
            serving: vec![
                within(SERVE_MAX_ABS_DIFF, self.serve_tolerance),
                within(SERVE_BATCH_MAX_ABS_DIFF, self.batch_tolerance),
                within(SERVE_ABSTENTION_RATE, self.max_abstention_rate),
                within(SERVE_INVALID_PROBABILITIES, 0.0),
                within(SERVE_NON_MONOTONE_CURVES, 0.0),
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan() -> TimelinePlan {
        TimelinePlan {
            codes: vec![CodePlan {
                code: "death:cvd".into(),
                horizons: vec![5.0, 10.0],
                must_improve: true,
            }],
            improve_nll: true,
            calibration: CalibrationBands {
                slope: (0.8, 1.25),
                intercept: (-0.4, 0.4),
                oe: (0.8, 1.25),
                max_ece: 0.05,
            },
            subgroups: vec!["female".into()],
            subgroup_margin: 0.002,
            min_subgroup_events: 10,
            serve_tolerance: 1e-6,
            batch_tolerance: 1e-6,
            max_abstention_rate: 0.1,
        }
    }

    #[test]
    fn a_plan_registers_a_requirement_for_every_judged_number() {
        let spec = plan().spec();
        assert_eq!(
            spec.performance.len(),
            2,
            "ibs of the code, and the likelihood"
        );
        assert_eq!(spec.calibration.len(), 8, "four bands at two horizons");
        assert_eq!(
            spec.retention.len(),
            2,
            "events of the subgroup, and its worst regression"
        );
        assert_eq!(
            spec.serving.len(),
            5,
            "identity, batching, abstention, validity, monotonicity"
        );
        assert!(spec.calibration.iter().all(|r| matches!(
            r,
            Requirement::Within { value, .. } if value.starts_with("candidate:death:cvd:h")
        )));
    }

    #[test]
    fn names_say_arm_code_horizon_and_metric() {
        assert_eq!(
            arm_metric(Arm::Candidate, "death:cvd", Some(5.0), "slope"),
            "candidate:death:cvd:h5:slope"
        );
        assert_eq!(
            arm_metric(Arm::Champion, "death:any", None, "ibs"),
            "champion:death:any:ibs"
        );
        assert_eq!(
            diff_metric("death:cvd", Some(2.5), "brier"),
            "diff:death:cvd:h2.5:brier"
        );
        assert_eq!(
            subgroup_diff("female", "death:cvd"),
            "subgroup:female:death:cvd"
        );
    }
}
