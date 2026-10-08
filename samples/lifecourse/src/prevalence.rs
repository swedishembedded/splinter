// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Scoring models of prevalent conditions (T4) and undiagnosed-disease
//! screens (T5), paper section 4.5.
//!
//! Swedish Embedded AB implements validation of risk and screening models
//! for its clients. If your team needs expertise in deciding whether a score
//! is good enough to screen with, and saying so honestly, you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! Predictions come from `baselines/conditions.py`
//! (`conditions/<model>/r<repeat>-k<fold>.jsonl`). A subject enters a label's
//! metrics only if its label is known (and, for the fasting label, it has a
//! fasting weight); weights are the survey weights. For T4 each fold is
//! scored and the models are compared with the age-and-sex model by the
//! corrected resampled t-test over the folds. For T5 a call is "score at or
//! above the threshold chosen on the training subjects", sensitivity and
//! specificity are pooled over the folds of one repeat, predictive values are
//! restated at the weighted prevalence of the screened population, and the
//! difference to the comparator's sensitivity has a cluster bootstrap interval.

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use splinter_sdk::model::timeline::survival::binary::{
    auroc, average_precision, brier, predictive_values, rates_at, recalibration,
    threshold_for_specificity,
};
use splinter_sdk::model::timeline::survival::compare::{
    cluster_bootstrap_by, corrected_resampled_t,
};

use crate::build::Design;
use crate::commands::{designs, frozen, Frozen};
use crate::conditions::Conditions;
use crate::report::cluster;

/// The labels T4 scores.
pub const PREVALENT: [&str; 9] = [
    "diabetes",
    "hypertension",
    "kidney_markers",
    "anaemia",
    "high_cholesterol",
    "osteoporosis",
    "depression",
    "sleep_problem",
    "diabetes_fasting",
];
/// The labels T5 screens for.
pub const UNDIAGNOSED: [&str; 4] = [
    "undiagnosed_diabetes",
    "undiagnosed_hypertension",
    "undiagnosed_kidney_markers",
    "undiagnosed_high_cholesterol",
];
/// Gain in AUROC over the age-and-sex model that makes a model useful (section 4.5).
pub const USEFUL_GAIN: f64 = 0.02;
#[cfg(not(test))]
const REPS: usize = 1000;
/// Fewer replicates keep the specifications fast; they test the sign of the interval.
#[cfg(test)]
const REPS: usize = 100;
const SEED: u64 = 20261008;
/// The specificity the screening thresholds aim for.
const SCREEN_SPECIFICITY: f64 = 0.9;

#[derive(Deserialize)]
struct PredLine {
    subject_id: String,
    p: BTreeMap<String, f64>,
}

#[derive(Deserialize, Default)]
struct Meta {
    #[serde(default)]
    thresholds: BTreeMap<String, f64>,
}

/// One fold of one model.
pub struct FoldPreds {
    pub p: HashMap<String, BTreeMap<String, f64>>,
    pub thresholds: BTreeMap<String, f64>,
}

fn read_fold(data: &Path, model: &str, repeat: usize, fold: usize) -> Result<Option<FoldPreds>> {
    let stem = data
        .join("conditions")
        .join(model)
        .join(format!("r{repeat}-k{fold}"));
    let Ok(text) = std::fs::read_to_string(stem.with_extension("jsonl")) else {
        return Ok(None);
    };
    let mut p = HashMap::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let l: PredLine =
            serde_json::from_str(line).with_context(|| format!("{model} r{repeat} k{fold}"))?;
        p.insert(l.subject_id, l.p);
    }
    let meta: Meta = serde_json::from_slice(
        &std::fs::read(stem.with_extension("meta.json"))
            .with_context(|| format!("{model}: no meta"))?,
    )?;
    Ok(Some(FoldPreds {
        p,
        thresholds: meta.thresholds,
    }))
}

/// Labelled subjects of a fold for `label`: probability, outcome, weight.
pub struct Scored {
    pub id: String,
    pub p: f64,
    pub y: bool,
    pub w: f64,
}

fn weight_of(f: &Frozen, c: &Conditions, label: &str) -> Option<f64> {
    if label == "diabetes_fasting" {
        c.fasting_weight
    } else {
        Some(f.subjects[&c.subject_id].weight)
    }
}

fn scored(
    f: &Frozen,
    conds: &HashMap<String, Conditions>,
    preds: &FoldPreds,
    label: &str,
) -> Vec<Scored> {
    let mut out: Vec<Scored> = preds
        .p
        .iter()
        .filter_map(|(id, ps)| {
            let c = conds.get(id)?;
            let y = (*c.labels.get(label)?)?;
            Some(Scored {
                id: id.clone(),
                p: *ps.get(label)?,
                y,
                w: weight_of(f, c, label)?,
            })
        })
        .collect();
    out.sort_by(|a, b| a.id.cmp(&b.id));
    out
}

/// Metrics of one fold for one label.
#[derive(Clone, Debug, PartialEq)]
pub struct FoldMetrics {
    pub n: usize,
    pub positives: usize,
    pub prevalence: f64,
    pub auroc: f64,
    pub average_precision: f64,
    pub brier: f64,
    pub intercept: f64,
    pub slope: f64,
}

/// Metrics of `rows`, or `None` when a class is too small to score.
pub fn fold_metrics(rows: &[Scored]) -> Option<FoldMetrics> {
    let positives = rows.iter().filter(|r| r.y).count();
    if positives < 5 || rows.len() - positives < 5 {
        return None;
    }
    let (p, y, w): (Vec<f64>, Vec<bool>, Vec<f64>) = (
        rows.iter().map(|r| r.p).collect(),
        rows.iter().map(|r| r.y).collect(),
        rows.iter().map(|r| r.w).collect(),
    );
    let total: f64 = w.iter().sum();
    let (intercept, slope) = recalibration(&p, &y, &w);
    Some(FoldMetrics {
        n: rows.len(),
        positives,
        prevalence: rows.iter().filter(|r| r.y).map(|r| r.w).sum::<f64>() / total,
        auroc: auroc(&p, &y, &w),
        average_precision: average_precision(&p, &y, &w),
        brier: brier(&p, &y, &w),
        intercept,
        slope,
    })
}

type Folds = BTreeMap<(usize, usize), FoldPreds>;

fn read_model(data: &Path, f: &Frozen, model: &str) -> Result<Folds> {
    let mut out = Folds::new();
    for repeat in 0..f.partition.repeats.len() {
        for fold in 0..f.partition.spec.folds as usize {
            if let Some(p) = read_fold(data, model, repeat, fold)? {
                out.insert((repeat, fold), p);
            }
        }
    }
    if out.is_empty() {
        bail!(
            "no predictions for {model} under {}/conditions",
            data.display()
        );
    }
    Ok(out)
}

fn read_conditions(data: &Path) -> Result<HashMap<String, Conditions>> {
    std::fs::read_to_string(data.join("conditions.jsonl"))
        .context("conditions.jsonl: run `labels` first")?
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| serde_json::from_str::<Conditions>(l).map(|c| (c.subject_id.clone(), c)))
        .collect::<Result<_, _>>()
        .map_err(Into::into)
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

/// Per-fold metrics of a model for a label, keyed by fold.
fn label_metrics(
    f: &Frozen,
    conds: &HashMap<String, Conditions>,
    folds: &Folds,
    label: &str,
) -> BTreeMap<(usize, usize), FoldMetrics> {
    folds
        .iter()
        .filter_map(|(k, p)| Some((*k, fold_metrics(&scored(f, conds, p, label))?)))
        .collect()
}

/// T4: a table per label of the models against `reference`.
pub fn report_prevalent(data: &Path, models: &[String], reference: &str) -> Result<String> {
    let f = frozen(data)?;
    let conds = read_conditions(data)?;
    let k = f.partition.spec.folds as f64;
    let test_over_train = 1.0 / (k - 1.0);
    let mut by_model: BTreeMap<String, Folds> = BTreeMap::new();
    for m in models.iter().map(String::as_str).chain([reference]) {
        if !by_model.contains_key(m) {
            by_model.insert(m.to_string(), read_model(data, &f, m)?);
        }
    }
    let mut out = String::new();
    writeln!(
        out,
        "A model is useful for a label iff its AUROC beats `{reference}` by at least {USEFUL_GAIN} with a corrected interval excluding zero.\n"
    )?;
    for label in PREVALENT {
        let ref_m = label_metrics(&f, &conds, &by_model[reference], label);
        if ref_m.is_empty() {
            continue;
        }
        let first = ref_m.values().next().map(|m| m.n).unwrap_or(0);
        writeln!(
            out,
            "\n### {label} (weighted prevalence {:.3}, about {first} subjects per fold)\n",
            mean(&ref_m.values().map(|m| m.prevalence).collect::<Vec<_>>())
        )?;
        writeln!(out, "| model | folds | AUROC | AP | Brier | slope | AUROC minus {reference} [95% CI] | useful |")?;
        writeln!(out, "|---|---|---|---|---|---|---|---|")?;
        for name in models {
            let mm = label_metrics(&f, &conds, &by_model[name], label);
            if mm.is_empty() {
                writeln!(out, "| {name} | 0 | not scored | | | | | |")?;
                continue;
            }
            let col = |g: fn(&FoldMetrics) -> f64| mean(&mm.values().map(g).collect::<Vec<_>>());
            let diffs: Vec<f64> = mm
                .iter()
                .filter_map(|(k, m)| Some(m.auroc - ref_m.get(k)?.auroc))
                .collect();
            let (delta, useful) = if name == reference {
                ("reference".to_string(), "-".to_string())
            } else {
                match corrected_resampled_t(&diffs, test_over_train) {
                    Some(t) => (
                        format!("{:+.4} [{:+.4}, {:+.4}]", t.mean, t.ci95.0, t.ci95.1),
                        if t.mean >= USEFUL_GAIN && t.ci95.0 > 0.0 {
                            "yes"
                        } else {
                            "no"
                        }
                        .to_string(),
                    ),
                    None => ("n/a".to_string(), "-".to_string()),
                }
            };
            writeln!(
                out,
                "| {name} | {} | {:.4} | {:.4} | {:.5} | {:.3} | {delta} | {useful} |",
                mm.len(),
                col(|m| m.auroc),
                col(|m| m.average_precision),
                col(|m| m.brier),
                col(|m| m.slope),
            )?;
        }
    }
    Ok(out)
}

/// One pooled screening call.
struct Call {
    y: bool,
    w: f64,
    score: f64,
    positive: bool,
    cluster: u64,
}

fn calls(
    f: &Frozen,
    conds: &HashMap<String, Conditions>,
    folds: &Folds,
    design: &HashMap<String, Design>,
    label: &str,
    repeat: usize,
) -> Vec<Call> {
    let mut out: Vec<(String, Call)> = Vec::new();
    for ((r, _), preds) in folds {
        if *r != repeat {
            continue;
        }
        let Some(&threshold) = preds.thresholds.get(label) else {
            continue;
        };
        for s in scored(f, conds, preds, label) {
            out.push((
                s.id.clone(),
                Call {
                    y: s.y,
                    w: s.w,
                    score: s.p,
                    positive: s.p >= threshold,
                    cluster: design.get(&s.id).map_or(0, cluster),
                },
            ));
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out.into_iter().map(|(_, c)| c).collect()
}

/// Weighted sensitivity and specificity of pooled calls.
pub fn rates(calls: &[(bool, f64, bool)], ix: &[usize]) -> Option<(f64, f64)> {
    let (mut tp, mut pos, mut tn, mut neg) = (0.0, 0.0, 0.0, 0.0);
    for &i in ix {
        let (y, w, call) = calls[i];
        if y {
            pos += w;
            tp += if call { w } else { 0.0 };
        } else {
            neg += w;
            tn += if call { 0.0 } else { w };
        }
    }
    (pos > 0.0 && neg > 0.0).then(|| (tp / pos, tn / neg))
}

/// Sensitivity at exactly `specificity` on the resample `ix`: the threshold
/// is re-chosen on the resample, so every model is read at the same
/// specificity. A supplement to the pre-registered threshold from training
/// subjects, which gives each model its own specificity.
pub fn matched_sensitivity(
    rows: &[(bool, f64, f64)],
    ix: &[usize],
    specificity: f64,
) -> Option<f64> {
    let (mut score, mut label, mut weight) = (vec![], vec![], vec![]);
    for &i in ix {
        let (y, w, s) = rows[i];
        score.push(s);
        label.push(y);
        weight.push(w);
    }
    if !label.iter().any(|l| *l) || label.iter().all(|l| *l) {
        return None;
    }
    let t = threshold_for_specificity(&score, &label, &weight, specificity);
    Some(rates_at(&score, &label, &weight, t).0)
}

/// T5: sensitivity at the stated specificity, predictive values at the
/// population prevalence, and the difference to `comparator`.
pub fn report_screen(
    data: &Path,
    models: &[String],
    comparator: &str,
    repeat: usize,
) -> Result<String> {
    let f = frozen(data)?;
    let conds = read_conditions(data)?;
    let design = designs(data)?;
    let mut by_model: BTreeMap<String, Folds> = BTreeMap::new();
    for m in models.iter().map(String::as_str).chain([comparator]) {
        if !by_model.contains_key(m) {
            by_model.insert(m.to_string(), read_model(data, &f, m)?);
        }
    }
    let mut out = String::new();
    writeln!(
        out,
        "A screen is claimed useful iff its sensitivity at the chosen threshold beats `{comparator}` with a cluster-bootstrap interval excluding zero. Threshold: 90% specificity on cross-fitted training predictions, applied to the test fold. Repeat {repeat}, folds pooled.\n"
    )?;
    for label in UNDIAGNOSED {
        let cmp = calls(&f, &conds, &by_model[comparator], &design, label, repeat);
        if cmp.is_empty() {
            continue;
        }
        let total: f64 = cmp.iter().map(|c| c.w).sum();
        let prevalence = cmp.iter().filter(|c| c.y).map(|c| c.w).sum::<f64>() / total;
        writeln!(
            out,
            "\n### {label}: {} screened subjects, {} with it, weighted prevalence {prevalence:.4}\n",
            cmp.len(),
            cmp.iter().filter(|c| c.y).count()
        )?;
        writeln!(out, "| model | sensitivity | specificity | PPV at prevalence | NPV | sensitivity minus {comparator} [95% CI] | beats | sensitivity at 90% specificity, matched | minus {comparator} [95% CI] |")?;
        writeln!(out, "|---|---|---|---|---|---|---|---|---|")?;
        let cmp_scores: Vec<(bool, f64, f64)> = cmp.iter().map(|c| (c.y, c.w, c.score)).collect();
        let cmp_rows: Vec<(bool, f64, bool)> = cmp.iter().map(|c| (c.y, c.w, c.positive)).collect();
        let clusters: Vec<u64> = cmp.iter().map(|c| c.cluster).collect();
        let all: Vec<usize> = (0..cmp.len()).collect();
        for name in models {
            let c = calls(&f, &conds, &by_model[name], &design, label, repeat);
            if c.len() != cmp.len() {
                writeln!(
                    out,
                    "| {name} | does not cover the same subjects | | | | | |"
                )?;
                continue;
            }
            let rows: Vec<(bool, f64, bool)> = c.iter().map(|c| (c.y, c.w, c.positive)).collect();
            let Some((sens, spec)) = rates(&rows, &all) else {
                continue;
            };
            let (ppv, npv) = predictive_values(sens, spec, prevalence);
            let (delta, beats) = if name == comparator {
                ("comparator".to_string(), "-".to_string())
            } else {
                let iv = cluster_bootstrap_by(&clusters, REPS, 0.95, SEED, |ix| {
                    Some(rates(&rows, ix)?.0 - rates(&cmp_rows, ix)?.0)
                });
                match iv {
                    Some(i) => (
                        format!("{:+.4} [{:+.4}, {:+.4}]", i.estimate, i.lo, i.hi),
                        if i.lo > 0.0 { "yes" } else { "no" }.to_string(),
                    ),
                    None => ("n/a".to_string(), "-".to_string()),
                }
            };
            let scores: Vec<(bool, f64, f64)> = c.iter().map(|c| (c.y, c.w, c.score)).collect();
            let matched = matched_sensitivity(&scores, &all, SCREEN_SPECIFICITY);
            let matched_delta = if name == comparator {
                "comparator".to_string()
            } else {
                cluster_bootstrap_by(&clusters, REPS, 0.95, SEED, |ix| {
                    Some(
                        matched_sensitivity(&scores, ix, SCREEN_SPECIFICITY)?
                            - matched_sensitivity(&cmp_scores, ix, SCREEN_SPECIFICITY)?,
                    )
                })
                .map_or("n/a".to_string(), |i| {
                    format!("{:+.4} [{:+.4}, {:+.4}]", i.estimate, i.lo, i.hi)
                })
            };
            writeln!(
                out,
                "| {name} | {sens:.3} | {spec:.3} | {ppv:.3} | {npv:.3} | {delta} | {beats} | {} | {matched_delta} |",
                matched.map_or("n/a".to_string(), |m| format!("{m:.3}")),
            )?;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(p: f64, y: bool, w: f64) -> Scored {
        Scored {
            id: format!("{p}-{y}"),
            p,
            y,
            w,
        }
    }

    #[test]
    fn a_fold_with_too_few_of_a_class_is_not_scored_and_a_perfect_score_is_one() {
        let mut rows: Vec<Scored> = (0..40)
            .map(|i| row(0.1 + f64::from(i) * 1e-4, false, 2.0))
            .collect();
        assert!(fold_metrics(&rows).is_none());
        rows.extend((0..10).map(|i| row(0.8 + f64::from(i) * 1e-4, true, 2.0)));
        let m = fold_metrics(&rows).unwrap();
        assert_eq!((m.n, m.positives), (50, 10));
        assert_eq!(m.auroc, 1.0);
        assert_eq!(m.average_precision, 1.0);
        assert!((m.prevalence - 0.2).abs() < 1e-12);
    }

    #[test]
    fn pooled_rates_are_weighted_and_follow_the_resample() {
        // (outcome, weight, call): two cases (one caught), three non-cases (one called).
        let calls = [
            (true, 3.0, true),
            (true, 1.0, false),
            (false, 1.0, false),
            (false, 1.0, false),
            (false, 2.0, true),
        ];
        let (sens, spec) = rates(&calls, &[0, 1, 2, 3, 4]).unwrap();
        assert!((sens - 0.75).abs() < 1e-12 && (spec - 0.5).abs() < 1e-12);
        // Resampling the caught case twice and nobody else with the outcome.
        let (sens, _) = rates(&calls, &[0, 0, 2]).unwrap();
        assert_eq!(sens, 1.0);
        assert!(
            rates(&calls, &[0, 1]).is_none(),
            "no non-cases, no specificity"
        );
    }

    #[test]
    fn matched_sensitivity_reads_every_score_at_the_same_specificity() {
        // Negatives score 0..9 /10, cases 0.45..0.9: at 90% specificity the threshold sits above 0.8.
        let mut rows: Vec<(bool, f64, f64)> =
            (0..10).map(|i| (false, 1.0, f64::from(i) / 10.0)).collect();
        rows.extend([
            (true, 1.0, 0.45),
            (true, 1.0, 0.85),
            (true, 1.0, 0.95),
            (true, 1.0, 0.99),
        ]);
        let all: Vec<usize> = (0..rows.len()).collect();
        assert_eq!(matched_sensitivity(&rows, &all, 0.9), Some(0.75));
        // A rescaled copy of the same scores has the same matched sensitivity: only the ranking counts.
        let squashed: Vec<(bool, f64, f64)> =
            rows.iter().map(|&(y, w, s)| (y, w, s * s * 0.01)).collect();
        assert_eq!(matched_sensitivity(&squashed, &all, 0.9), Some(0.75));
        assert_eq!(
            matched_sensitivity(&rows, &[0, 1, 2], 0.9),
            None,
            "no cases in the resample"
        );
    }

    #[test]
    fn a_screen_that_catches_more_cases_at_the_same_specificity_has_an_interval_above_zero() {
        // Two screens over the same subjects; the first also catches half of the cases the second misses.
        let (mut a, mut b, mut clusters) = (vec![], vec![], vec![]);
        for i in 0..4000u64 {
            let case = i % 10 == 0;
            let b_call = case && i % 20 == 0;
            let a_call = case && i % 20 != 0 || b_call;
            let false_alarm = !case && i % 10 == 5;
            a.push((case, 1.0, a_call || false_alarm));
            b.push((case, 1.0, b_call || false_alarm));
            clusters.push(i / 4);
        }
        let iv = cluster_bootstrap_by(&clusters, REPS, 0.95, SEED, |ix| {
            Some(rates(&a, ix)?.0 - rates(&b, ix)?.0)
        })
        .unwrap();
        assert!(iv.estimate > 0.4 && iv.lo > 0.0, "{iv:?}");
    }
}
