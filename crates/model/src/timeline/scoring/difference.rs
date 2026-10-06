// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements honest head-to-head evaluation of
// time-to-event risk models on held-out participants, for its clients. If
// your team needs expertise in discrimination, calibration and paired
// comparison of competing-risk predictions, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The candidate-minus-champion differences with clustered bootstrap
//! intervals, and the subgroup comparisons.

use std::collections::BTreeMap;

use brain::survival::auc;
use brain::survival::brier::brier_terms;
use brain::survival::compare::{cluster_bootstrap, cluster_bootstrap_by, Interval as Boot};
use brain::survival::concordance::uno;
use brain::survival::estimate::Step;
use brain::survival::Obs;

use super::arm::{ibs_terms, score_arm, weighted_mean, Outcomes, Predicted};
use super::{
    Comparison, HorizonDifferences, Interval, ScoreSpec, SubgroupDifference, ViewDifferences,
};
use crate::timeline::training::TimelineError;
use crate::timeline::Subject;

/// A rank statistic of brain's: risks, outcomes, cause, horizon, censoring.
type RankStatistic = fn(&[f64], &[Obs], usize, f64, &Step) -> Option<f64>;

fn interval(b: Boot) -> Interval {
    Interval {
        estimate: b.estimate,
        lo: b.lo,
        hi: b.hi,
    }
}

/// The bootstrap interval of a per-subject paired difference, resampling
/// whole clusters.
fn paired_interval(
    diff: &[f64],
    weights: &[f64],
    clusters: &[u64],
    spec: &ScoreSpec,
    salt: u64,
) -> Option<Interval> {
    cluster_bootstrap(
        diff,
        weights,
        clusters,
        spec.bootstrap_reps,
        spec.level,
        spec.seed ^ salt,
    )
    .map(interval)
}

/// The interval of a rank statistic's difference, recomputed on every
/// resample of clusters for both arms.
fn rank_interval(
    clusters: &[u64],
    spec: &ScoreSpec,
    salt: u64,
    stat: impl Fn(&[usize]) -> Option<f64>,
) -> Option<Interval> {
    cluster_bootstrap_by(
        clusters,
        spec.bootstrap_reps,
        spec.level,
        spec.seed ^ salt,
        stat,
    )
    .map(interval)
}

fn pick<T: Copy>(values: &[T], at: &[usize]) -> Vec<T> {
    at.iter().map(|&i| values[i]).collect()
}

fn minus(candidate: &[f64], champion: &[f64]) -> Vec<f64> {
    candidate.iter().zip(champion).map(|(a, b)| a - b).collect()
}

/// A stable salt per statistic, so two statistics never share a resample
/// stream by accident while each stays reproducible.
fn salt(parts: &[usize]) -> u64 {
    parts.iter().fold(0x9E37_79B9_7F4A_7C15u64, |h, p| {
        h.rotate_left(7) ^ (*p as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
    })
}

/// Scores `champion` and `candidate`, which were predicted for `subjects`
/// (the same ones, in the same order), and differences them. Refused when the
/// two were not predicted for exactly these subjects, when a subject appears
/// twice or when there are none.
pub fn compare(
    champion: &Predicted,
    candidate: &Predicted,
    subjects: &[Subject],
    spec: &ScoreSpec,
) -> Result<Comparison, TimelineError> {
    let ids: Vec<String> = subjects.iter().map(|s| s.subject_id.clone()).collect();
    if ids.is_empty() {
        return Err(TimelineError::Request(
            "scoring: there are no held-out subjects".into(),
        ));
    }
    if champion.units != ids || candidate.units != ids {
        return Err(TimelineError::Request(format!(
            "scoring: the champion ({} units) and the candidate ({} units) were not both predicted \
             for exactly the {} subjects scored, in order",
            champion.units.len(),
            candidate.units.len(),
            ids.len()
        )));
    }
    let distinct: std::collections::BTreeSet<&String> = ids.iter().collect();
    if distinct.len() != ids.len() {
        return Err(TimelineError::Request(
            "scoring: a subject appears more than once".into(),
        ));
    }
    let out = Outcomes::of(subjects, spec);
    let all = || 0..ids.len();
    let mut differences = BTreeMap::new();
    for (v, view) in out.views.iter().enumerate() {
        let obs = &out.obs[v];
        let g = &out.g[v];
        let last_events = out.events(v, spec.last_horizon(), all());
        let ibs = if last_events >= spec.min_events {
            match (
                ibs_terms(&out, v, &champion.cif[v], spec),
                ibs_terms(&out, v, &candidate.cif[v], spec),
            ) {
                (Some(a), Some(b)) => paired_interval(
                    &minus(&b, &a),
                    &champion.weights,
                    &champion.clusters,
                    spec,
                    salt(&[v, 0]),
                ),
                _ => None,
            }
        } else {
            None
        };
        let mut horizons = Vec::new();
        for (k, t) in spec.horizons.iter().enumerate() {
            if out.events(v, *t, all()) < spec.min_events {
                continue;
            }
            let (a, b) = (&champion.cif[v][k], &candidate.cif[v][k]);
            let brier_diff = match (brier_terms(a, obs, 0, *t, g), brier_terms(b, obs, 0, *t, g)) {
                (Some(x), Some(y)) => paired_interval(
                    &minus(&y, &x),
                    &champion.weights,
                    &champion.clusters,
                    spec,
                    salt(&[v, k, 1]),
                ),
                _ => None,
            };
            let on = |stat: RankStatistic, sub: &[usize]| {
                let o = pick(obs, sub);
                Some(stat(&pick(b, sub), &o, 0, *t, g)? - stat(&pick(a, sub), &o, 0, *t, g)?)
            };
            horizons.push(HorizonDifferences {
                horizon: *t,
                brier: brier_diff,
                uno_c: rank_interval(&champion.clusters, spec, salt(&[v, k, 2]), |s| on(uno, s)),
                auc: rank_interval(&champion.clusters, spec, salt(&[v, k, 3]), |s| {
                    on(auc::at, s)
                }),
            });
        }
        differences.insert(view.name.clone(), ViewDifferences { ibs, horizons });
    }
    let event_nll = paired_interval(
        &minus(&candidate.nll, &champion.nll),
        &champion.weights,
        &champion.clusters,
        spec,
        salt(&[usize::MAX]),
    );
    let mut subgroups = BTreeMap::new();
    for group in &spec.subgroups {
        let members: Vec<usize> = subjects
            .iter()
            .enumerate()
            .filter(|(_, s)| group.holds(s))
            .map(|(i, _)| i)
            .collect();
        let mut by_view = BTreeMap::new();
        for (v, view) in out.views.iter().enumerate() {
            let events = out.events(v, spec.last_horizon(), members.iter().copied());
            let ibs = (events >= spec.min_events)
                .then(|| {
                    let a = ibs_terms(&out, v, &champion.cif[v], spec)?;
                    let b = ibs_terms(&out, v, &candidate.cif[v], spec)?;
                    weighted_mean(&minus(&b, &a), &champion.weights, members.iter().copied())
                })
                .flatten();
            by_view.insert(
                view.name.clone(),
                SubgroupDifference {
                    subjects: members.len(),
                    events,
                    ibs,
                },
            );
        }
        subgroups.insert(group.name.clone(), by_view);
    }
    Ok(Comparison {
        units: ids,
        champion: score_arm(champion, &out, spec),
        candidate: score_arm(candidate, &out, spec),
        event_nll,
        differences,
        subgroups,
    })
}
