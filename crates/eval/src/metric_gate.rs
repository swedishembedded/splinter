// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A release decision over continuous metrics: the gate for a model that is
//! not judged item by item (a risk model, a forecaster) but by a metric and
//! its interval.
//!
//! The requirements are data, written before the candidate is scored and
//! pinned with the evaluation set; the evidence is what was measured, by
//! name. A requirement whose evidence is missing fails: unmeasured is never
//! a pass. How the numbers were computed (which bootstrap, which test) is the
//! caller's; this module only decides.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::gate::Check;

/// One pre-registered requirement on named evidence.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Requirement {
    /// The interval of a difference (candidate minus baseline) lies entirely
    /// on the better side of zero.
    Improves {
        /// The interval's name in the evidence.
        interval: String,
        /// Whether lower values of the metric are better.
        lower_is_better: bool,
    },
    /// A value lies within `[lo, hi]`.
    Within {
        /// The value's name.
        value: String,
        /// Lower bound.
        lo: f64,
        /// Upper bound.
        hi: f64,
    },
    /// An interval contains `target`.
    Covers {
        /// The interval's name.
        interval: String,
        /// What it must contain.
        target: f64,
    },
    /// Every value named with `prefix` (a difference, candidate minus
    /// baseline, per subgroup) is no worse than `bound`.
    NotWorseBy {
        /// Name prefix of the per-subgroup differences.
        prefix: String,
        /// How much worse is tolerated.
        bound: f64,
        /// Whether lower values of the metric are better.
        lower_is_better: bool,
    },
    /// A test's p-value is at least `alpha` (the hypothesis it tests is not rejected).
    NotRejected {
        /// The p-value's name.
        p_value: String,
        /// Significance level.
        alpha: f64,
    },
}

/// What was measured, by name.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Evidence {
    /// Point values.
    pub values: BTreeMap<String, f64>,
    /// Intervals `(lo, hi)`.
    pub intervals: BTreeMap<String, (f64, f64)>,
}

/// The decision: one check per requirement, in order.
#[derive(Clone, Debug, PartialEq)]
pub struct MetricGate {
    /// Each requirement's check.
    pub checks: Vec<(Requirement, Check<String>)>,
}

impl MetricGate {
    /// Whether every requirement passed.
    #[must_use]
    pub fn passed(&self) -> bool {
        !self.checks.is_empty() && self.checks.iter().all(|(_, c)| c.passed)
    }
}

fn check(measured: String, failure: Option<String>) -> Check<String> {
    Check {
        passed: failure.is_none(),
        measured: Some(measured),
        reason: failure,
    }
}

fn decide_one(r: &Requirement, e: &Evidence) -> Check<String> {
    match r {
        Requirement::Improves {
            interval,
            lower_is_better,
        } => match e.intervals.get(interval) {
            None => Check::unmeasured(format!("no interval {interval}")),
            Some(&(lo, hi)) => {
                let ok = if *lower_is_better { hi < 0.0 } else { lo > 0.0 };
                check(format!("[{lo:+.6}, {hi:+.6}]"), (!ok).then(|| format!("{interval} [{lo:+.6}, {hi:+.6}] does not exclude zero on the better side")))
            }
        },
        Requirement::Within { value, lo, hi } => match e.values.get(value) {
            None => Check::unmeasured(format!("no value {value}")),
            Some(&v) => check(
                format!("{v:.6}"),
                (!(*lo..=*hi).contains(&v))
                    .then(|| format!("{value} {v:.6} is outside [{lo}, {hi}]")),
            ),
        },
        Requirement::Covers { interval, target } => match e.intervals.get(interval) {
            None => Check::unmeasured(format!("no interval {interval}")),
            Some(&(lo, hi)) => check(
                format!("[{lo:+.6}, {hi:+.6}]"),
                (!(lo <= *target && *target <= hi))
                    .then(|| format!("{interval} [{lo:+.6}, {hi:+.6}] does not contain {target}")),
            ),
        },
        Requirement::NotWorseBy {
            prefix,
            bound,
            lower_is_better,
        } => {
            let worse = |d: f64| if *lower_is_better { d } else { -d };
            let worst = e
                .values
                .iter()
                .filter(|(k, _)| k.starts_with(prefix.as_str()))
                .max_by(|a, b| worse(*a.1).total_cmp(&worse(*b.1)));
            match worst {
                None => Check::unmeasured(format!("no value named {prefix}*")),
                Some((name, &d)) => check(
                    format!("worst {name} {d:+.6}"),
                    (worse(d) > *bound)
                        .then(|| format!("{name} is worse by {:.6}, beyond {bound}", worse(d))),
                ),
            }
        }
        Requirement::NotRejected { p_value, alpha } => match e.values.get(p_value) {
            None => Check::unmeasured(format!("no p-value {p_value}")),
            Some(&p) => check(
                format!("p = {p:.4}"),
                (p < *alpha).then(|| format!("{p_value} = {p:.4} < {alpha}")),
            ),
        },
    }
}

/// Decide every requirement on the evidence.
#[must_use]
pub fn decide(requirements: &[Requirement], evidence: &Evidence) -> MetricGate {
    MetricGate {
        checks: requirements
            .iter()
            .map(|r| (r.clone(), decide_one(r, evidence)))
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence() -> Evidence {
        let mut e = Evidence::default();
        e.intervals.insert("ibs_diff".into(), (-0.004, -0.001));
        e.intervals.insert("intercept".into(), (-0.2, 0.1));
        e.values.insert("slope".into(), 0.97);
        e.values.insert("dcal_p".into(), 0.3);
        e.values.insert("subgroup:sex=male".into(), -0.002);
        e.values.insert("subgroup:sex=female".into(), 0.001);
        e
    }

    #[test]
    fn every_requirement_kind_decides() {
        let reqs = vec![
            Requirement::Improves {
                interval: "ibs_diff".into(),
                lower_is_better: true,
            },
            Requirement::Within {
                value: "slope".into(),
                lo: 0.9,
                hi: 1.1,
            },
            Requirement::Covers {
                interval: "intercept".into(),
                target: 0.0,
            },
            Requirement::NotWorseBy {
                prefix: "subgroup:".into(),
                bound: 0.002,
                lower_is_better: true,
            },
            Requirement::NotRejected {
                p_value: "dcal_p".into(),
                alpha: 0.05,
            },
        ];
        let g = decide(&reqs, &evidence());
        assert!(g.passed(), "{g:?}");
        let mut e = evidence();
        e.intervals.insert("ibs_diff".into(), (-0.004, 0.0005));
        e.values.insert("subgroup:sex=female".into(), 0.003);
        let g = decide(&reqs, &e);
        assert!(!g.checks[0].1.passed && !g.checks[3].1.passed && g.checks[1].1.passed);
    }

    #[test]
    fn unmeasured_fails_and_an_empty_gate_never_passes() {
        let reqs = vec![Requirement::Within {
            value: "missing".into(),
            lo: 0.0,
            hi: 1.0,
        }];
        let g = decide(&reqs, &Evidence::default());
        assert!(!g.passed());
        assert!(g.checks[0].1.measured.is_none());
        assert!(!decide(&[], &evidence()).passed());
    }

    #[test]
    fn requirements_round_trip_as_data() {
        let r = Requirement::NotWorseBy {
            prefix: "s:".into(),
            bound: 0.002,
            lower_is_better: true,
        };
        let text = serde_json::to_string(&r).unwrap();
        assert!(text.contains("\"kind\":\"not_worse_by\""));
        assert_eq!(serde_json::from_str::<Requirement>(&text).unwrap(), r);
    }
}
