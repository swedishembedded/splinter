// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements honest head-to-head evaluation of
// time-to-event risk models on held-out participants, for its clients. If
// your team needs expertise in discrimination, calibration and paired
// comparison of competing-risk predictions, you can procure our services by
// sending an email to info@swedishembedded.com.

//! One model's predictions on the scored subjects, and its metrics.

use std::collections::BTreeMap;

use brain::survival::auc;
use brain::survival::brier::{brier, brier_terms};
use brain::survival::calibration::at_horizon;
use brain::survival::concordance::uno;
use brain::survival::estimate::{censoring, Step};
use brain::survival::Obs;

use super::{ArmScores, Calibration, HorizonScores, ScoreSpec, ViewScores};
use crate::timeline::training::TimelineError;
use crate::timeline::{observed, Subject, TimelineModel};

/// What a view of the outcomes is.
#[derive(Clone, Debug, PartialEq)]
pub enum ViewKind {
    /// One outcome code, the other absorbing codes competing with it.
    Cause(String),
    /// Any absorbing code: the union.
    AllCause,
}

/// A view of the outcomes metrics are reported on.
#[derive(Clone, Debug, PartialEq)]
pub struct View {
    /// The name its numbers travel under: the code, or the all-cause name.
    pub name: String,
    /// What it is.
    pub kind: ViewKind,
}

/// The views a spec scores: each outcome code, then the all-cause union.
pub(super) fn views(spec: &ScoreSpec) -> Vec<View> {
    let mut views: Vec<View> = spec
        .codes
        .iter()
        .map(|c| View {
            name: c.clone(),
            kind: ViewKind::Cause(c.clone()),
        })
        .collect();
    if let Some(name) = &spec.all_cause {
        views.push(View {
            name: name.clone(),
            kind: ViewKind::AllCause,
        });
    }
    views
}

/// The times after entry the integrated Brier score is taken over.
pub(super) fn grid(spec: &ScoreSpec) -> Vec<f64> {
    let last = spec.last_horizon();
    (1..=spec.ibs_points)
        .map(|k| last * k as f64 / spec.ibs_points as f64)
        .collect()
}

/// The outcomes of `subjects` as `view` sees them, cause 0 being the outcome
/// of interest.
pub(super) fn view_obs(view: &View, subjects: &[Subject], spec: &ScoreSpec) -> Vec<Obs> {
    match &view.kind {
        ViewKind::Cause(code) => {
            let mut order: Vec<&str> = vec![code];
            order.extend(
                spec.absorbing
                    .iter()
                    .map(String::as_str)
                    .filter(|a| a != code),
            );
            observed(subjects, &order)
        }
        ViewKind::AllCause => {
            let order: Vec<&str> = spec.absorbing.iter().map(String::as_str).collect();
            observed(subjects, &order)
                .into_iter()
                .map(|o| Obs {
                    cause: o.cause.map(|_| 0),
                    ..o
                })
                .collect()
        }
    }
}

/// A stable 64-bit cluster id of a group (FNV-1a): the same group is the
/// same cluster in every run.
fn cluster_of(subject: &Subject) -> u64 {
    // A record in no group is its own cluster.
    let key = subject
        .group_id
        .as_deref()
        .unwrap_or(subject.subject_id.as_str());
    key.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// One model's predictions on the scored subjects.
#[derive(Clone, Debug)]
pub struct Predicted {
    /// The subjects' ids, in order.
    pub units: Vec<String>,
    pub(super) weights: Vec<f64>,
    pub(super) clusters: Vec<u64>,
    /// Per-subject event negative log-likelihood.
    pub(super) nll: Vec<f64>,
    /// `cif[view][time][subject]` for the horizons, then the grid.
    pub(super) cif: Vec<Vec<Vec<f64>>>,
}

/// Predicts `subjects` with `model` for every view at every horizon and grid
/// point of `spec`. Refused when a prediction is not a probability.
pub fn predict_arm(
    model: &TimelineModel,
    subjects: &[Subject],
    spec: &ScoreSpec,
) -> Result<Predicted, TimelineError> {
    spec.check(&[model, model])?;
    let times: Vec<f64> = spec.horizons.iter().copied().chain(grid(spec)).collect();
    let predictions = model.predict(subjects)?;
    let mut cif = Vec::new();
    for view in views(spec) {
        let mut by_time = Vec::with_capacity(times.len());
        for t in &times {
            let mut column = Vec::with_capacity(subjects.len());
            for p in &predictions {
                let risk = match &view.kind {
                    ViewKind::Cause(code) => p.cif(code, *t).ok_or_else(|| {
                        TimelineError::Request(format!("the model predicts no code {code:?}"))
                    })?,
                    ViewKind::AllCause => 1.0 - p.survival(*t),
                };
                if !(0.0..=1.0 + 1e-9).contains(&risk) {
                    return Err(TimelineError::Brain(format!(
                        "a prediction of {} by {t} is {risk}, not a probability",
                        view.name
                    )));
                }
                column.push(risk.clamp(0.0, 1.0));
            }
            by_time.push(column);
        }
        cif.push(by_time);
    }
    let mut nll = Vec::with_capacity(subjects.len());
    for s in subjects {
        nll.push(f64::from(model.event_nll(std::slice::from_ref(s))?));
    }
    Ok(Predicted {
        units: subjects.iter().map(|s| s.subject_id.clone()).collect(),
        weights: subjects.iter().map(|s| s.weight).collect(),
        clusters: subjects.iter().map(cluster_of).collect(),
        nll,
        cif,
    })
}

/// What is shared by both arms: the outcomes each view sees and the
/// censoring distribution it weights by.
pub(super) struct Outcomes {
    pub(super) views: Vec<View>,
    pub(super) obs: Vec<Vec<Obs>>,
    pub(super) g: Vec<Step>,
    pub(super) grid: Vec<f64>,
}

impl Outcomes {
    pub(super) fn of(subjects: &[Subject], spec: &ScoreSpec) -> Self {
        let views = views(spec);
        let obs: Vec<Vec<Obs>> = views.iter().map(|v| view_obs(v, subjects, spec)).collect();
        let g = obs.iter().map(|o| censoring(o)).collect();
        Self {
            views,
            obs,
            g,
            grid: grid(spec),
        }
    }

    /// Events of view `v` by `t` among the subjects `of`.
    pub(super) fn events(&self, v: usize, t: f64, of: impl Iterator<Item = usize>) -> usize {
        of.filter(|&i| {
            let o = &self.obs[v][i];
            o.cause == Some(0) && o.time <= t
        })
        .count()
    }
}

/// Each subject's term of the integrated Brier score of view `v` over the
/// grid (trapezoid rule, divided by the window), for the predictions `cif`.
pub(super) fn ibs_terms(
    out: &Outcomes,
    v: usize,
    cif: &[Vec<f64>],
    spec: &ScoreSpec,
) -> Option<Vec<f64>> {
    let per_t: Option<Vec<Vec<f64>>> = out
        .grid
        .iter()
        .enumerate()
        .map(|(k, t)| brier_terms(&cif[spec.horizons.len() + k], &out.obs[v], 0, *t, &out.g[v]))
        .collect();
    let per_t = per_t?;
    let width = out.grid[out.grid.len() - 1] - out.grid[0];
    Some(
        (0..out.obs[v].len())
            .map(|i| {
                let area: f64 = out
                    .grid
                    .windows(2)
                    .enumerate()
                    .map(|(k, w)| 0.5 * (per_t[k][i] + per_t[k + 1][i]) * (w[1] - w[0]))
                    .sum();
                area / width
            })
            .collect(),
    )
}

pub(super) fn weighted_mean(
    values: &[f64],
    weights: &[f64],
    of: impl Iterator<Item = usize>,
) -> Option<f64> {
    let (num, den) = of.fold((0.0, 0.0), |(n, d), i| {
        (n + values[i] * weights[i], d + weights[i])
    });
    (den > 0.0).then(|| num / den)
}

/// One arm's scores.
pub(super) fn score_arm(pred: &Predicted, out: &Outcomes, spec: &ScoreSpec) -> ArmScores {
    let mut views = BTreeMap::new();
    let all = || 0..pred.units.len();
    for (v, view) in out.views.iter().enumerate() {
        let obs = &out.obs[v];
        let g = &out.g[v];
        let ibs = (out.events(v, spec.last_horizon(), all()) >= spec.min_events)
            .then(|| ibs_terms(out, v, &pred.cif[v], spec))
            .flatten()
            .and_then(|terms| weighted_mean(&terms, &pred.weights, all()));
        let mut horizons = Vec::new();
        for (k, t) in spec.horizons.iter().enumerate() {
            let events = out.events(v, *t, all());
            if events < spec.min_events {
                continue;
            }
            let risk = &pred.cif[v][k];
            let cal = at_horizon(risk, obs, 0, *t, g, spec.calibration_groups);
            let finite = |x: f64| x.is_finite().then_some(x);
            let calibration = match (
                finite(cal.slope),
                finite(cal.intercept),
                finite(cal.oe_ratio),
            ) {
                (Some(slope), Some(intercept), Some(oe)) => Some(Calibration {
                    slope,
                    intercept,
                    oe,
                    ece: cal.ece(),
                    observed: cal.observed,
                    expected: cal.expected,
                }),
                _ => None,
            };
            horizons.push(HorizonScores {
                horizon: *t,
                events,
                uno_c: uno(risk, obs, 0, *t, g),
                auc: auc::at(risk, obs, 0, *t, g),
                brier: brier(risk, obs, 0, *t, g),
                calibration,
            });
        }
        views.insert(view.name.clone(), ViewScores { ibs, horizons });
    }
    let event_nll = weighted_mean(&pred.nll, &pred.weights, all()).unwrap_or(f64::NAN);
    ArmScores { event_nll, views }
}
