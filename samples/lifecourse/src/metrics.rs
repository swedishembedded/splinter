// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The pre-registered metrics of one model on one set of subjects.
//!
//! Follow-up ends administratively (31 December 2019), so a cycle examined in
//! 2015 is followed for about four years at most. A metric at horizon `h` is
//! therefore computed on the cycles whose surviving participants were
//! followed at least `h` years ([`Horizons`], from the built data): inside
//! that set practically nobody is censored before `h` except by death, and
//! the inverse probability of censoring weights are close to one. Weighting long-horizon outcomes
//! up from short-follow-up cycles where nobody was followed that far is not
//! done - it is not estimable.
//!
//! All metrics are survey-weighted (the pooled examination weight).
//! All-cause death is the primary outcome: its risk by `t` is one minus the
//! predicted survival.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use splinter_sdk::model::timeline::survival::brier::{brier, brier_terms};
use splinter_sdk::model::timeline::survival::calibration::{at_horizon, d_calibration};
use splinter_sdk::model::timeline::survival::concordance::uno;
use splinter_sdk::model::timeline::survival::estimate::censoring;
use splinter_sdk::model::timeline::survival::Obs;
use splinter_sdk::model::timeline::{observed, Prediction, Subject};

use crate::build::CODES;

/// The horizon over which the primary metric integrates, and its grid.
pub const PRIMARY_HORIZON: u32 = 15;

/// Per cycle, the follow-up every survivor reached: the 0.1st percentile of
/// the survivors' follow-up, so the one or two records in a cycle with a
/// follow-up of zero (the public file's disclosure perturbation) do not set
/// it. Those few are then ordinary censorings inside the set, which the
/// censoring weights account for.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Horizons(pub BTreeMap<String, f64>);

impl Horizons {
    /// From the built subjects.
    pub fn of(subjects: &[Subject]) -> Horizons {
        let mut by: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        for s in subjects {
            let died = s.events.iter().any(|e| e.t > s.entry);
            if !died {
                by.entry(s.source.clone())
                    .or_default()
                    .push(s.at_risk.first().map_or(0.0, |w| w.to - s.entry));
            }
        }
        Horizons(
            by.into_iter()
                .map(|(k, mut v)| {
                    v.sort_by(f64::total_cmp);
                    (k, v[v.len() / 1000])
                })
                .collect(),
        )
    }

    /// Whether a subject's cycle supports horizon `h`.
    pub fn supports(&self, s: &Subject, h: f64) -> bool {
        self.0.get(&s.source).is_some_and(|fu| *fu >= h)
    }
}

/// A metric with the size of the set it was computed on.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Measured {
    /// The value.
    pub value: f64,
    /// Subjects in the evaluation set.
    pub n: usize,
    /// Events of the outcome among them, by the horizon.
    pub events: usize,
}

/// Everything measured on one test set.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Metrics {
    /// All-cause death: integrated Brier score over 1..=15 years (primary; lower is better).
    pub ibs_0_15: Option<Measured>,
    /// All-cause Brier score at 5, 10 and 15 years.
    pub brier: BTreeMap<u32, Measured>,
    /// All-cause Uno concordance truncated at 5 and 10 years.
    pub uno_c: BTreeMap<u32, Measured>,
    /// All-cause calibration at 10 years: slope, intercept, observed/expected, mean group gap.
    pub calibration_10: Option<Calibration>,
    /// D-calibration of predicted all-cause survival over every test subject.
    pub d_calibration_p: Option<f64>,
    /// Cause-specific Brier and Uno C at 10 years, per death code.
    pub cause_10: BTreeMap<String, (Measured, Measured)>,
}

/// Calibration at a horizon.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Calibration {
    /// Recalibration slope (1 when calibrated).
    pub slope: f64,
    /// Recalibration intercept (0 when calibrated).
    pub intercept: f64,
    /// Observed over expected.
    pub oe_ratio: f64,
    /// Weighted mean gap between observed and expected over risk deciles.
    pub mean_abs_gap: f64,
    /// Subjects.
    pub n: usize,
}

pub(crate) fn all_cause(obs: Vec<Obs>) -> Vec<Obs> {
    obs.into_iter()
        .map(|o| Obs {
            cause: o.cause.map(|_| 0),
            ..o
        })
        .collect()
}

pub(crate) fn subset<'a>(
    subjects: &'a [Subject],
    preds: &'a [Prediction],
    h: &Horizons,
    t: f64,
) -> (Vec<Subject>, Vec<&'a Prediction>) {
    let idx: Vec<usize> = (0..subjects.len())
        .filter(|&i| h.supports(&subjects[i], t))
        .collect();
    (
        idx.iter().map(|&i| subjects[i].clone()).collect(),
        idx.iter().map(|&i| &preds[i]).collect(),
    )
}

fn events_by(obs: &[Obs], cause: usize, t: f64) -> usize {
    obs.iter()
        .filter(|o| o.cause == Some(cause) && o.time <= t)
        .count()
}

/// Each subject's integrated all-cause Brier term over 1..=15 years (the
/// trapezoid rule over yearly terms, divided by the window), for subjects
/// whose cycle supports 15 years; `(subject_id, term)`.
pub fn ibs_terms(subjects: &[Subject], preds: &[Prediction], h: &Horizons) -> Vec<(String, f64)> {
    let (subs, ps) = subset(subjects, preds, h, PRIMARY_HORIZON as f64);
    if subs.is_empty() {
        return vec![];
    }
    let obs = all_cause(observed(&subs, &CODES));
    let g = censoring(&obs);
    let grid: Vec<f64> = (1..=PRIMARY_HORIZON).map(f64::from).collect();
    let per_t: Vec<Vec<f64>> = grid
        .iter()
        .map(|&t| {
            let risk: Vec<f64> = ps.iter().map(|p| 1.0 - p.survival(t)).collect();
            brier_terms(&risk, &obs, 0, t, &g).unwrap_or_else(|| vec![f64::NAN; risk.len()])
        })
        .collect();
    let width = grid[grid.len() - 1] - grid[0];
    (0..subs.len())
        .map(|i| {
            let area: f64 = grid
                .windows(2)
                .enumerate()
                .map(|(k, w)| 0.5 * (per_t[k][i] + per_t[k + 1][i]) * (w[1] - w[0]))
                .sum();
            (subs[i].subject_id.clone(), area / width)
        })
        .collect()
}

/// The pre-registered metrics of `preds` on `subjects`.
pub fn evaluate(subjects: &[Subject], preds: &[Prediction], h: &Horizons) -> Metrics {
    let mut m = Metrics::default();
    let terms = ibs_terms(subjects, preds, h);
    if !terms.is_empty() {
        let weight: BTreeMap<&str, f64> = subjects
            .iter()
            .map(|s| (s.subject_id.as_str(), s.weight))
            .collect();
        let (num, den) = terms.iter().fold((0.0, 0.0), |(a, b), (id, x)| {
            (a + x * weight[id.as_str()], b + weight[id.as_str()])
        });
        let (subs, _) = subset(subjects, preds, h, PRIMARY_HORIZON as f64);
        let obs = all_cause(observed(&subs, &CODES));
        m.ibs_0_15 = Some(Measured {
            value: num / den,
            n: terms.len(),
            events: events_by(&obs, 0, PRIMARY_HORIZON as f64),
        });
    }
    for t in [5u32, 10, 15] {
        let (subs, ps) = subset(subjects, preds, h, t as f64);
        if subs.len() < 2 {
            continue;
        }
        let obs = all_cause(observed(&subs, &CODES));
        let g = censoring(&obs);
        let risk: Vec<f64> = ps.iter().map(|p| 1.0 - p.survival(t as f64)).collect();
        let events = events_by(&obs, 0, t as f64);
        if let Some(b) = brier(&risk, &obs, 0, t as f64, &g) {
            m.brier.insert(
                t,
                Measured {
                    value: b,
                    n: subs.len(),
                    events,
                },
            );
        }
        if t <= 10 {
            if let Some(c) = uno(&risk, &obs, 0, t as f64, &g) {
                m.uno_c.insert(
                    t,
                    Measured {
                        value: c,
                        n: subs.len(),
                        events,
                    },
                );
            }
        }
        if t == 10 {
            let cal = at_horizon(&risk, &obs, 0, 10.0, &g, 10);
            m.calibration_10 = Some(Calibration {
                slope: cal.slope,
                intercept: cal.intercept,
                oe_ratio: cal.oe_ratio,
                mean_abs_gap: cal.mean_abs_gap,
                n: subs.len(),
            });
            for (k, code) in CODES.iter().enumerate() {
                // The cause of interest first, the others competing.
                let mut order: Vec<&str> = vec![code];
                order.extend(CODES.iter().filter(|c| *c != code));
                let cobs = observed(&subs, &order);
                let cif: Vec<f64> = ps
                    .iter()
                    .map(|p| p.cif(code, 10.0).unwrap_or(f64::NAN))
                    .collect();
                let events = events_by(&cobs, 0, 10.0);
                let gb = brier(&cif, &cobs, 0, 10.0, &g);
                let gc = uno(&cif, &cobs, 0, 10.0, &g);
                if let (Some(b), Some(c)) = (gb, gc) {
                    m.cause_10.insert(
                        CODES[k].to_string(),
                        (
                            Measured {
                                value: b,
                                n: subs.len(),
                                events,
                            },
                            Measured {
                                value: c,
                                n: subs.len(),
                                events,
                            },
                        ),
                    );
                }
            }
        }
    }
    let obs = all_cause(observed(subjects, &CODES));
    let surv: Vec<f64> = preds
        .iter()
        .zip(&obs)
        .map(|(p, o)| p.survival(o.time))
        .collect();
    m.d_calibration_p = Some(d_calibration(&surv, &obs, 10).p_value);
    m
}
