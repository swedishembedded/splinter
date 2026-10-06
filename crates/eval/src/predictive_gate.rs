// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pre-registered release gates for
// predictive models, where a candidate replaces the champion only on
// paired held-out evidence. If your team needs expertise in validating
// risk models or forecasters before they ship, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The release gate for a predictive model: one judged by metrics and their
//! intervals on held-out units, not by graded answers.
//!
//! Five checks decide a candidate against the champion it would replace:
//!
//! 1. **Predictive performance**: the candidate beats the champion by the
//!    pre-registered requirements.
//! 2. **Calibration**: the candidate's predictions are calibrated as
//!    required.
//! 3. **Retention**: no material regression on any named subgroup or slice.
//! 4. **Serving correctness**: the served model reproduces the evaluated one.
//! 5. **Data-policy compliance**: the terms of everything the candidate was
//!    made from permit the release asked for ([`policy_check`]).
//!
//! The first four are lists of [`Requirement`]s (a [`PredictiveSpec`]),
//! written as data before the candidate is scored; the gate only decides them
//! over the [`Evidence`] it is handed, by name ([`crate::metric_gate`]). The
//! caller computes every number, with whichever method it trusts; this module
//! performs no survival or prediction arithmetic. A requirement whose
//! evidence is absent fails, and a check with no requirement registered
//! fails: unmeasured is never a pass.
//!
//! Champion and candidate are compared on the same units. Both are named, and
//! a gate over two different unit sets, a repeated unit or an empty one is
//! refused outright ([`PairingError`]): numbers over different units are not
//! a comparison.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use splinter_core::digest::{canonical_json, Digest};
use splinter_core::terms::{Distribution, Terms, Use};

use crate::gate::Check;
use crate::metric_gate::{decide, Evidence, Requirement};

/// The most unit ids a [`PairingError::Mismatch`] names; the counts are
/// always whole.
const MISMATCH_SAMPLE: usize = 5;

/// The pre-registered requirements of each check.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PredictiveSpec {
    /// Predictive performance of the candidate against the champion.
    pub performance: Vec<Requirement>,
    /// Calibration of the candidate.
    pub calibration: Vec<Requirement>,
    /// No material regression on named subgroups or slices.
    pub retention: Vec<Requirement>,
    /// The served model matches the evaluated one.
    pub serving: Vec<Requirement>,
}

/// What was measured, on which units.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Measurements {
    /// The held-out units the champion was scored on.
    pub champion_units: Vec<String>,
    /// The held-out units the candidate was scored on: the same ones.
    pub candidate_units: Vec<String>,
    /// Every number, by name, as the requirements name them.
    pub evidence: Evidence,
}

/// Why two arms cannot be compared.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PairingError {
    /// An arm was scored on no unit.
    #[error("the {arm} was scored on no held-out unit")]
    Empty {
        /// `champion` or `candidate`.
        arm: &'static str,
    },
    /// An arm names a unit twice.
    #[error("the {arm} names held-out unit {unit:?} more than once")]
    Duplicate {
        /// `champion` or `candidate`.
        arm: &'static str,
        /// The repeated unit.
        unit: String,
    },
    /// The arms were scored on different units.
    #[error(
        "the champion and candidate were scored on different held-out units: {only_champion} \
         only the champion's (e.g. {champion_sample:?}), {only_candidate} only the candidate's \
         (e.g. {candidate_sample:?})"
    )]
    Mismatch {
        /// How many units only the champion has.
        only_champion: usize,
        /// Some of them.
        champion_sample: Vec<String>,
        /// How many units only the candidate has.
        only_candidate: usize,
        /// Some of them.
        candidate_sample: Vec<String>,
    },
}

/// One requirement and how it was decided.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Decided {
    /// The requirement.
    pub requirement: Requirement,
    /// Its check.
    pub check: Check<String>,
}

/// One of the four requirement-based checks.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Section {
    /// Each requirement, decided, in order.
    pub decided: Vec<Decided>,
    /// Whether every requirement passed; false with none registered.
    pub passed: bool,
}

impl Section {
    fn of(requirements: &[Requirement], evidence: &Evidence) -> Self {
        let gate = decide(requirements, evidence);
        Self {
            passed: gate.passed(),
            decided: gate
                .checks
                .into_iter()
                .map(|(requirement, check)| Decided { requirement, check })
                .collect(),
        }
    }
}

/// The gate's decision with every number it rests on.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PredictiveReport {
    /// Held-out units both arms were scored on.
    pub units: usize,
    /// The digest of the sorted unit ids: the evaluation split's identity.
    pub units_digest: Digest,
    /// Predictive performance.
    pub performance: Section,
    /// Calibration.
    pub calibration: Section,
    /// Retention on subgroups and slices.
    pub retention: Section,
    /// Serving correctness.
    pub serving: Section,
    /// Data-policy compliance.
    pub policy: Check<String>,
    /// Whether all five checks passed.
    pub passed: bool,
}

/// The data-policy check: whether the terms of everything the candidate was
/// made from permit a release of the kind asked for. Training must be allowed
/// for any release; an unrestricted one needs every axis allowed. Terms that
/// were never recorded are unmeasured, so they fail.
#[must_use]
pub fn policy_check(terms: Option<&Terms>, distribution: Distribution) -> Check<String> {
    let Some(terms) = terms else {
        return Check::unmeasured(
            "no terms were recorded for the data the candidate was made from",
        );
    };
    let verdict = match distribution {
        Distribution::Restricted => terms.permits(Use::Training),
        Distribution::Unrestricted => terms.permits_unrestricted_release(),
    };
    Check {
        passed: verdict.is_ok(),
        measured: Some(format!(
            "{}: training {:?}, commercial use {:?}, redistribution {:?}; {distribution:?} release",
            terms.name, terms.training, terms.commercial_use, terms.redistribution
        )),
        reason: verdict.err(),
    }
}

/// The distinct units of one arm; refused when it has none or repeats one.
fn unit_set<'a>(arm: &'static str, units: &'a [String]) -> Result<BTreeSet<&'a str>, PairingError> {
    if units.is_empty() {
        return Err(PairingError::Empty { arm });
    }
    let mut set = BTreeSet::new();
    for unit in units {
        if !set.insert(unit.as_str()) {
            return Err(PairingError::Duplicate {
                arm,
                unit: unit.clone(),
            });
        }
    }
    Ok(set)
}

/// The units both arms were scored on, sorted, or why the arms are not
/// paired.
fn paired_units(m: &Measurements) -> Result<Vec<&str>, PairingError> {
    let champion = unit_set("champion", &m.champion_units)?;
    let candidate = unit_set("candidate", &m.candidate_units)?;
    if champion != candidate {
        let sample = |a: &BTreeSet<&str>, b: &BTreeSet<&str>| -> (usize, Vec<String>) {
            let only: Vec<&&str> = a.difference(b).collect();
            (
                only.len(),
                only.into_iter()
                    .take(MISMATCH_SAMPLE)
                    .map(|u| (*u).to_string())
                    .collect(),
            )
        };
        let (only_champion, champion_sample) = sample(&champion, &candidate);
        let (only_candidate, candidate_sample) = sample(&candidate, &champion);
        return Err(PairingError::Mismatch {
            only_champion,
            champion_sample,
            only_candidate,
            candidate_sample,
        });
    }
    Ok(champion.into_iter().collect())
}

/// Decides a candidate against its champion: refused when the two were not
/// scored on the same units, otherwise every check, with every number.
/// `terms` are those of everything the candidate was made from, `None` when
/// none were recorded.
pub fn decide_predictive(
    spec: &PredictiveSpec,
    measurements: &Measurements,
    terms: Option<&Terms>,
    distribution: Distribution,
) -> Result<PredictiveReport, PairingError> {
    let units = paired_units(measurements)?;
    // Canonical JSON of a list of strings cannot fail; a failure would be a
    // fault in the serializer, and the digest of nothing would then be wrong
    // rather than absent, so it is not papered over.
    let encoded = canonical_json(&units).unwrap_or_default();
    let performance = Section::of(&spec.performance, &measurements.evidence);
    let calibration = Section::of(&spec.calibration, &measurements.evidence);
    let retention = Section::of(&spec.retention, &measurements.evidence);
    let serving = Section::of(&spec.serving, &measurements.evidence);
    let policy = policy_check(terms, distribution);
    let passed = performance.passed
        && calibration.passed
        && retention.passed
        && serving.passed
        && policy.passed;
    Ok(PredictiveReport {
        units: units.len(),
        units_digest: Digest::of(&encoded),
        performance,
        calibration,
        retention,
        serving,
        policy,
        passed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use splinter_core::terms::UsagePolicy;

    fn spec() -> PredictiveSpec {
        PredictiveSpec {
            performance: vec![Requirement::Improves {
                interval: "ibs_diff".into(),
                lower_is_better: true,
            }],
            calibration: vec![
                Requirement::Within {
                    value: "slope".into(),
                    lo: 0.9,
                    hi: 1.1,
                },
                Requirement::Covers {
                    interval: "intercept".into(),
                    target: 0.0,
                },
            ],
            retention: vec![Requirement::NotWorseBy {
                prefix: "subgroup:".into(),
                bound: 0.002,
                lower_is_better: true,
            }],
            serving: vec![Requirement::Within {
                value: "serve_max_abs_diff".into(),
                lo: 0.0,
                hi: 1e-6,
            }],
        }
    }

    fn units() -> Vec<String> {
        ["u3", "u1", "u2"].map(String::from).to_vec()
    }

    fn evidence() -> Evidence {
        let mut e = Evidence::default();
        e.intervals.insert("ibs_diff".into(), (-0.004, -0.001));
        e.intervals.insert("intercept".into(), (-0.2, 0.1));
        e.values.insert("slope".into(), 0.97);
        e.values.insert("subgroup:sex=male".into(), -0.002);
        e.values.insert("subgroup:sex=female".into(), 0.001);
        e.values.insert("serve_max_abs_diff".into(), 0.0);
        e
    }

    fn measurements(evidence: Evidence) -> Measurements {
        Measurements {
            champion_units: units(),
            candidate_units: units().into_iter().rev().collect(),
            evidence,
        }
    }

    fn open() -> Terms {
        UsagePolicy::Redistributable.terms("open")
    }

    fn report(evidence: Evidence, terms: Option<&Terms>) -> PredictiveReport {
        decide_predictive(
            &spec(),
            &measurements(evidence),
            terms,
            Distribution::Unrestricted,
        )
        .unwrap()
    }

    #[test]
    fn a_candidate_that_meets_every_check_passes_in_any_unit_order() {
        let r = report(evidence(), Some(&open()));
        assert!(r.passed, "{r:#?}");
        assert_eq!(r.units, 3);
        let shuffled = decide_predictive(
            &spec(),
            &Measurements {
                champion_units: vec!["u2".into(), "u3".into(), "u1".into()],
                ..measurements(evidence())
            },
            Some(&open()),
            Distribution::Unrestricted,
        )
        .unwrap();
        assert_eq!(
            shuffled.units_digest, r.units_digest,
            "the split's identity is its set"
        );
    }

    #[test]
    fn each_check_fails_alone() {
        let mut worse = evidence();
        worse.intervals.insert("ibs_diff".into(), (-0.004, 0.0005));
        let r = report(worse, Some(&open()));
        assert!(!r.passed && !r.performance.passed);
        assert!(r.calibration.passed && r.retention.passed && r.serving.passed && r.policy.passed);

        let mut miscalibrated = evidence();
        miscalibrated.values.insert("slope".into(), 1.4);
        let r = report(miscalibrated, Some(&open()));
        assert!(!r.passed && !r.calibration.passed && r.performance.passed);

        let mut regressed = evidence();
        regressed.values.insert("subgroup:age=80+".into(), 0.02);
        let r = report(regressed, Some(&open()));
        assert!(!r.passed && !r.retention.passed && r.calibration.passed);
        let why = r.retention.decided[0].check.reason.as_deref().unwrap();
        assert!(why.contains("subgroup:age=80+"), "{why}");

        let mut drifting = evidence();
        drifting.values.insert("serve_max_abs_diff".into(), 0.01);
        let r = report(drifting, Some(&open()));
        assert!(!r.passed && !r.serving.passed && r.retention.passed);
    }

    #[test]
    fn an_unmeasured_quantity_fails_and_is_never_zero() {
        let mut missing = evidence();
        missing.values.remove("serve_max_abs_diff");
        let r = report(missing, Some(&open()));
        assert!(!r.passed && !r.serving.passed);
        let check = &r.serving.decided[0].check;
        assert_eq!(check.measured, None);
        assert!(check.reason.as_deref().unwrap().starts_with("not measured"));

        let mut no_subgroups = evidence();
        no_subgroups
            .values
            .retain(|k, _| !k.starts_with("subgroup:"));
        assert!(!report(no_subgroups, Some(&open())).retention.passed);

        // A check nobody registered a requirement for is not a pass either.
        let unregistered = decide_predictive(
            &PredictiveSpec {
                serving: Vec::new(),
                ..spec()
            },
            &measurements(evidence()),
            Some(&open()),
            Distribution::Unrestricted,
        )
        .unwrap();
        assert!(!unregistered.serving.passed && !unregistered.passed);
    }

    #[test]
    fn arms_scored_on_different_units_are_refused() {
        let mut m = measurements(evidence());
        m.candidate_units = vec!["u1".into(), "u2".into(), "u9".into()];
        let err =
            decide_predictive(&spec(), &m, Some(&open()), Distribution::Restricted).unwrap_err();
        assert_eq!(
            err,
            PairingError::Mismatch {
                only_champion: 1,
                champion_sample: vec!["u3".into()],
                only_candidate: 1,
                candidate_sample: vec!["u9".into()],
            }
        );
        assert!(err.to_string().contains("u9"));

        let mut empty = measurements(evidence());
        empty.champion_units.clear();
        assert_eq!(
            decide_predictive(&spec(), &empty, None, Distribution::Restricted).unwrap_err(),
            PairingError::Empty { arm: "champion" }
        );
        let mut repeated = measurements(evidence());
        repeated.candidate_units.push("u1".into());
        assert_eq!(
            decide_predictive(&spec(), &repeated, None, Distribution::Restricted).unwrap_err(),
            PairingError::Duplicate {
                arm: "candidate",
                unit: "u1".into()
            }
        );
    }

    #[test]
    fn a_policy_failure_blocks_a_candidate_that_passes_every_measurement() {
        for label in [
            UsagePolicy::ResearchOnly,
            UsagePolicy::Noncommercial,
            UsagePolicy::RestrictedDua,
            UsagePolicy::Unknown,
        ] {
            let terms = label.terms(label.as_str());
            let r = report(evidence(), Some(&terms));
            assert!(
                r.performance.passed
                    && r.calibration.passed
                    && r.retention.passed
                    && r.serving.passed,
                "{label:?}"
            );
            assert!(
                !r.policy.passed && !r.passed,
                "{label:?} must not be released as unrestricted"
            );
        }
        // Terms never recorded are unmeasured: fail.
        let r = report(evidence(), None);
        assert!(!r.passed && r.policy.measured.is_none());
    }

    #[test]
    fn a_restricted_release_needs_training_allowed_and_nothing_more() {
        let research = UsagePolicy::ResearchOnly.terms("cohort");
        let r = decide_predictive(
            &spec(),
            &measurements(evidence()),
            Some(&research),
            Distribution::Restricted,
        )
        .unwrap();
        assert!(r.passed, "{r:#?}");
        let unknown = UsagePolicy::Unknown.terms("unread");
        assert!(!policy_check(Some(&unknown), Distribution::Restricted).passed);
    }
}
