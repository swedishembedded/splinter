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
//!
//! # The calibration rule
//!
//! A calibration band judged on a few dozen events is mostly noise: the
//! recalibration slope of a perfectly calibrated model spreads by 0.1 to 0.2
//! at a few hundred events, so a point-estimate band would reject good models
//! at random and pass bad ones by luck. The plan therefore judges a (code,
//! horizon) only where the test part holds at least
//! [`TimelinePlan::min_test_events`] events of the code by the horizon
//! ([`DEFAULT_MIN_TEST_EVENTS`]). Elsewhere the horizon is
//! [`Judgement::Uncertified`]: not failed, not zero, and not offered as a
//! calibrated output of the release. The pairs in [`TimelinePlan::required`]
//! must be judged for a release: a required pair the test part cannot
//! certify fails closed, so a cohort too small to certify cannot release.

use serde::{Deserialize, Serialize};

use crate::metric_gate::{Evidence, Requirement};
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
/// The fewest events of a code by a horizon the test part must hold for the
/// horizon's calibration to be judged (see the module documentation).
pub const DEFAULT_MIN_TEST_EVENTS: usize = 200;

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

/// An arm's own metric (`uno_c`, `auc`, `brier`, `slope`, `intercept`,
/// `intercept_in_the_large`, `oe`,
/// `ece`, `ibs`) for `code`, at `horizon` or over the whole grid (`None`).
#[must_use]
pub fn arm_metric(arm: Arm, code: &str, horizon: Option<f64>, metric: &str) -> String {
    format!("{}:{code}{}:{metric}", arm.name(), at(horizon))
}

/// The events of `code` by `horizon` in the scored units, measured for every
/// horizon the plan names, however few: the count a judgement is made on.
#[must_use]
pub fn arm_events(arm: Arm, code: &str, horizon: f64) -> String {
    arm_metric(arm, code, Some(horizon), "events")
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
    /// Calibration in the large: the log-odds shift that makes the average
    /// predicted risk the observed one (0 when calibrated). It replaces the
    /// two-parameter recalibration intercept, which is read at a predicted risk
    /// of one half and so measures the slope error as much as the average risk;
    /// a plan written with `intercept` is refused when read, not reinterpreted.
    pub intercept_in_the_large: (f64, f64),
    /// Observed over expected (1 when calibrated).
    pub oe: (f64, f64),
    /// The most the expected calibration error may be.
    pub max_ece: f64,
}

/// Which risk a horizon's calibration is judged on. Discrimination and error
/// are always the model's raw risk, so two models are compared on what they
/// predict; calibration is judged on the risk the model is served as.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationBasis {
    /// The model's raw cumulative incidence: it has no calibration at this
    /// code and horizon (none at all, or the validation data could not
    /// support one), so it is judged as it is.
    #[default]
    Raw,
    /// The calibrated risk brain serves at this code and horizon.
    Calibrated,
    /// Read from records written before an uncalibrated horizon was judged
    /// raw: brain declared it uncalibrated and nothing was judged. Never
    /// produced now.
    Uncalibrated,
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

/// A (code, horizon) that must be certified for a release.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RequiredHorizon {
    /// The outcome code.
    pub code: String,
    /// The horizon, in the data's unit.
    pub horizon: f64,
}

/// Whether the plan judged a horizon's calibration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Judgement {
    /// The test part held enough events: the bands apply to the point
    /// estimates, and a release certifies the horizon.
    Judged,
    /// The test part held too few events (or none were measured): nothing is
    /// judged, and the horizon carries no certified risk.
    Uncertified,
}

/// One (code, horizon) as the plan judged it, recorded with the evaluation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HorizonJudgement {
    /// The outcome code, or the all-cause name.
    pub code: String,
    /// The horizon.
    pub horizon: f64,
    /// Events by the horizon in the test part; absent when not measured.
    pub test_events: Option<usize>,
    /// Whether a release requires this pair to be certified.
    pub required: bool,
    /// Judged or uncertified.
    pub status: Judgement,
    /// The risk the calibration was judged on; absent when it was not
    /// measured.
    pub basis: Option<CalibrationBasis>,
}

/// The pre-registered requirements of a timeline release.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TimelinePlan {
    /// Each judged code.
    pub codes: Vec<CodePlan>,
    /// The fewest test events of a code by a horizon for its calibration to
    /// be judged ([`DEFAULT_MIN_TEST_EVENTS`]).
    pub min_test_events: usize,
    /// The pairs a release needs certified; one that cannot be judged for
    /// lack of events fails the release.
    pub required: Vec<RequiredHorizon>,
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
    /// How the plan judges each of its (code, horizon) pairs on `evidence`,
    /// in plan order, the required pairs a code does not list last. `basis`
    /// says which risk the candidate's calibration was measured on.
    #[must_use]
    pub fn judgements(
        &self,
        evidence: &Evidence,
        basis: impl Fn(&str, f64) -> Option<CalibrationBasis>,
    ) -> Vec<HorizonJudgement> {
        let mut pairs: Vec<(&str, f64)> = self
            .codes
            .iter()
            .flat_map(|c| c.horizons.iter().map(|h| (c.code.as_str(), *h)))
            .collect();
        for r in &self.required {
            if !pairs.iter().any(|(c, h)| *c == r.code && *h == r.horizon) {
                pairs.push((r.code.as_str(), r.horizon));
            }
        }
        pairs
            .into_iter()
            .map(|(code, horizon)| {
                let events = evidence
                    .values
                    .get(&arm_events(Arm::Candidate, code, horizon))
                    .map(|e| *e as usize);
                let judged = events.is_some_and(|e| e >= self.min_test_events);
                HorizonJudgement {
                    code: code.to_owned(),
                    horizon,
                    test_events: events,
                    required: self
                        .required
                        .iter()
                        .any(|r| r.code == code && r.horizon == horizon),
                    status: if judged {
                        Judgement::Judged
                    } else {
                        Judgement::Uncertified
                    },
                    basis: basis(code, horizon),
                }
            })
            .collect()
    }

    /// The requirements this plan registers, by check, given what was
    /// measured: the calibration bands are registered for the pairs the test
    /// part can judge, and a required pair it cannot judge registers the
    /// event minimum it fails.
    #[must_use]
    pub fn spec(&self, evidence: &Evidence) -> PredictiveSpec {
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
        for j in self.judgements(evidence, |_, _| None) {
            match (j.status, j.required) {
                (Judgement::Judged, _) => {
                    let within = |metric: &str, (lo, hi): (f64, f64)| Requirement::Within {
                        value: arm_metric(Arm::Candidate, &j.code, Some(j.horizon), metric),
                        lo,
                        hi,
                    };
                    calibration.push(within("slope", bands.slope));
                    calibration.push(within(
                        "intercept_in_the_large",
                        bands.intercept_in_the_large,
                    ));
                    calibration.push(within("oe", bands.oe));
                    calibration.push(within("ece", (0.0, bands.max_ece)));
                }
                (Judgement::Uncertified, true) => calibration.push(Requirement::AtLeast {
                    value: arm_events(Arm::Candidate, &j.code, j.horizon),
                    min: self.min_test_events as f64,
                }),
                (Judgement::Uncertified, false) => {}
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
    use crate::metric_gate::decide;

    fn plan() -> TimelinePlan {
        TimelinePlan {
            codes: vec![CodePlan {
                code: "death:cvd".into(),
                horizons: vec![5.0, 10.0],
                must_improve: true,
            }],
            min_test_events: 200,
            required: vec![RequiredHorizon {
                code: "death:cvd".into(),
                horizon: 5.0,
            }],
            improve_nll: true,
            calibration: CalibrationBands {
                slope: (0.8, 1.25),
                intercept_in_the_large: (-0.4, 0.4),
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

    /// Evidence of a candidate calibrated at every band's centre, with these
    /// events by the 5 and the 10 year horizon.
    fn evidence(events5: f64, events10: f64) -> Evidence {
        let mut e = Evidence::default();
        for (h, events) in [(5.0, events5), (10.0, events10)] {
            let put = |e: &mut Evidence, metric: &str, v: f64| {
                e.values
                    .insert(arm_metric(Arm::Candidate, "death:cvd", Some(h), metric), v);
            };
            put(&mut e, "events", events);
            put(&mut e, "slope", 1.0);
            put(&mut e, "intercept_in_the_large", 0.0);
            put(&mut e, "oe", 1.0);
            put(&mut e, "ece", 0.01);
        }
        e
    }

    fn decided(plan: &TimelinePlan, e: &Evidence) -> crate::metric_gate::MetricGate {
        decide(&plan.spec(e).calibration, e)
    }

    #[test]
    fn a_plan_registers_a_requirement_for_every_judged_number() {
        let spec = plan().spec(&evidence(400.0, 300.0));
        assert_eq!(
            spec.performance.len(),
            2,
            "ibs of the code, and the likelihood"
        );
        assert_eq!(
            spec.calibration.len(),
            8,
            "four bands at two judged horizons"
        );
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
    fn a_horizon_with_too_few_events_is_uncertified_not_failed() {
        let e = evidence(400.0, 150.0);
        let spec = plan().spec(&e);
        assert_eq!(
            spec.calibration.len(),
            4,
            "only the 5 year horizon is judged"
        );
        assert!(
            decided(&plan(), &e).passed(),
            "the 10 year horizon is not a failure"
        );
        let j = plan().judgements(&e, |_, _| Some(CalibrationBasis::Calibrated));
        let status = |h: f64| j.iter().find(|j| j.horizon == h).unwrap();
        assert_eq!(status(5.0).status, Judgement::Judged);
        assert_eq!(status(10.0).status, Judgement::Uncertified);
        assert_eq!(status(10.0).test_events, Some(150));
        assert!(status(5.0).required && !status(10.0).required);
    }

    #[test]
    fn the_event_minimum_itself_is_judged() {
        let e = evidence(200.0, 199.0);
        let j = plan().judgements(&e, |_, _| None);
        assert_eq!(j[0].status, Judgement::Judged);
        assert_eq!(j[1].status, Judgement::Uncertified);
    }

    #[test]
    fn a_required_pair_the_test_part_cannot_judge_fails_closed() {
        let e = evidence(150.0, 400.0);
        let gate = decided(&plan(), &e);
        assert!(!gate.passed());
        let failed: Vec<_> = gate.checks.iter().filter(|(_, c)| !c.passed).collect();
        assert_eq!(failed.len(), 1, "{failed:?}");
        let why = failed[0].1.reason.as_deref().unwrap();
        assert!(
            why.contains("death:cvd:h5:events") && why.contains("200"),
            "{why}"
        );

        // A required pair nobody measured fails as unmeasured, never as zero.
        let mut unmeasured = evidence(400.0, 400.0);
        unmeasured
            .values
            .remove(&arm_events(Arm::Candidate, "death:cvd", 5.0));
        let gate = decided(&plan(), &unmeasured);
        assert!(!gate.passed());
        assert!(gate.checks.iter().any(|(_, c)| c.measured.is_none()));
    }

    #[test]
    fn a_judged_horizon_outside_the_bands_fails() {
        let mut e = evidence(400.0, 400.0);
        e.values.insert(
            arm_metric(Arm::Candidate, "death:cvd", Some(10.0), "slope"),
            1.3,
        );
        let gate = decided(&plan(), &e);
        assert!(!gate.passed());
        let why = gate
            .checks
            .iter()
            .find_map(|(_, c)| c.reason.as_deref())
            .unwrap();
        assert!(why.contains("h10:slope"), "{why}");
    }

    #[test]
    fn a_required_pair_outside_the_listed_horizons_is_still_required() {
        let mut p = plan();
        p.codes[0].horizons = vec![10.0];
        let e = evidence(100.0, 400.0);
        assert!(
            !decided(&p, &e).passed(),
            "5 years is required, listed or not"
        );
        assert_eq!(p.judgements(&e, |_, _| None).len(), 2);
    }

    #[test]
    fn names_say_arm_code_horizon_and_metric() {
        assert_eq!(
            arm_metric(Arm::Candidate, "death:cvd", Some(5.0), "slope"),
            "candidate:death:cvd:h5:slope"
        );
        assert_eq!(
            arm_events(Arm::Candidate, "death:cvd", 5.0),
            "candidate:death:cvd:h5:events"
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
