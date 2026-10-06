// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The pre-registered success criteria, frozen with the partition before
//! any model is trained, and the report that holds the locked-test results
//! to them.
//!
//! The criteria are data ([`preregistered`]); `freeze` pins their JSON, so a
//! criterion changed after results exist is a different file with a
//! different digest, never an edit.

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Serialize;
use splinter_sdk::measure::metric_gate::{decide, Evidence, Requirement};
use splinter_sdk::model::timeline::survival::brier::brier;
use splinter_sdk::model::timeline::survival::calibration::at_horizon;
use splinter_sdk::model::timeline::survival::compare::{cluster_bootstrap, cluster_bootstrap_by};
use splinter_sdk::model::timeline::survival::concordance::uno;
use splinter_sdk::model::timeline::survival::estimate::censoring;
use splinter_sdk::model::timeline::survival::Obs;
use splinter_sdk::model::timeline::{observed, Subject, Value};

use crate::build::CODES;
use crate::commands::{cv_runs, designs, frozen, LockedDetail, Run, PREREGISTERED_SEED};
use crate::compare::{compared_metrics, Compared};
use crate::experiment::Arm;

/// One arm's runs by `(repeat, fold)`.
type FoldRuns = BTreeMap<(usize, usize), Run>;

/// Training seeds the seed-spread section looks for.
const SEEDS_REPORTED: u64 = 3;

/// Bootstrap replicates of every design-based interval.
pub const BOOTSTRAP_REPS: usize = 2000;
/// Seed of every design-based interval.
pub const BOOTSTRAP_SEED: u64 = 20_261_005;

/// The criteria, as frozen.
#[derive(Serialize)]
pub struct Criteria {
    version: u32,
    primary_metric: &'static str,
    candidate: &'static str,
    baseline: &'static str,
    comparators: [&'static str; 2],
    improvement: &'static str,
    calibration_slope_range: (f64, f64),
    calibration_intercept: &'static str,
    d_calibration_alpha: f64,
    subgroups: [&'static str; 4],
    subgroup_bound: f64,
    permutation: &'static str,
    bootstrap: &'static str,
    cross_validation: &'static str,
    amendments: [&'static str; 2],
}

/// The pre-registered criteria.
pub fn preregistered() -> Criteria {
    Criteria {
        version: 1,
        primary_metric: "integrated Brier score of all-cause death over 1..=15 years, survey-weighted (pooled MEC weight), on the cycles whose every survivor was followed at least 15 years",
        candidate: "horizon",
        baseline: "standard",
        comparators: ["age-sex", "additive"],
        // perf-number: a confidence level of an interval, not a measurement
        improvement: "on the locked test, the 95% cluster-bootstrap interval of the survey-weighted mean per-subject difference in the integrated Brier term (candidate minus baseline) lies entirely below zero",
        calibration_slope_range: (0.9, 1.1),
        // perf-number: a confidence level of an interval, not a measurement
        calibration_intercept: "the 95% cluster-bootstrap interval of the 10-year IPCW logistic recalibration intercept covers zero",
        d_calibration_alpha: 0.05,
        subgroups: ["sex", "age band at examination (18-39, 40-59, 60-79, 80+)", "race and ethnicity", "survey cycle"],
        subgroup_bound: 0.002,
        permutation: "the candidate trained on outcomes shuffled across training subjects has a cross-validated mean integrated Brier score no better (no lower) than the age-sex arm's",
        bootstrap: "2000 replicates resampling clusters (cycle x masked variance stratum x masked PSU) with replacement, seed 20261005, percentile interval",
        cross_validation: "5 repeats of 5 grouped stratified folds over the non-locked subjects; arms compared on identical folds with the corrected resampled t-test (Nadeau-Bengio, test/train = 1/(K-1)); concordance is reported but never decides",
        amendments: [
            "2026-10-05, before any model was trained: the gradient-boosted survival baseline named in the plan is replaced by the additive model on all inputs (a piecewise-exponential GAM) as the equal-information non-deep comparator, because no gradient-boosted survival trainer exists in this stack; a gradient-boosted baseline remains open work",
            "2026-10-05, before any model was trained: each horizon is measured on the cycles whose every survivor was followed that long (administrative end of follow-up 2019-12-31), rather than by weighting long-horizon outcomes up from cycles with no follow-up that long",
        ],
    }
}

/// A change to the criteria made after they were frozen: its own file,
/// pinned in the ledger beside the untouched criteria (`amend`), applied by
/// the report only when pinned.
#[derive(Serialize)]
pub struct Amendment {
    /// Continues the numbering of the amendments inside the criteria.
    pub number: u32,
    when: &'static str,
    change: &'static str,
    reason: &'static str,
}

/// Every amendment made after the freeze.
pub fn amendments() -> Vec<Amendment> {
    vec![Amendment {
        number: 3,
        when: "2026-10-05, after cross-validation of the candidate had begun and before the locked test was scored by any arm",
        change: "D-calibration is reported but no longer decides the claim: the requirement that its p-value is not below the frozen alpha is removed",
        reason: "under the heavy censoring of these data (most subjects are alive at the end of follow-up) the chi-square null of D-calibration does not hold: censored subjects spread their mass almost evenly over the bins, the statistic collapses, and simulation under the true model at this censoring never rejects, so the requirement could not fail for anything short of a grossly wrong model. Removing it makes no criterion easier for any arm to pass; the calibration slope and intercept requirements remain",
    }]
}

/// The criteria as the gate's requirements on the evidence `report` names,
/// with the amendments numbered in `pinned` applied.
pub fn requirements(c: &Criteria, pinned: &[u32]) -> Vec<Requirement> {
    let d_calibration_decides = !pinned.contains(&3);
    let mut v = vec![
        Requirement::Improves {
            interval: "ibs_0_15_diff".into(),
            lower_is_better: true,
        },
        Requirement::Within {
            value: "calibration_slope_10".into(),
            lo: c.calibration_slope_range.0,
            hi: c.calibration_slope_range.1,
        },
        Requirement::Covers {
            interval: "calibration_intercept_10".into(),
            target: 0.0,
        },
        Requirement::NotWorseBy {
            prefix: "subgroup:".into(),
            bound: c.subgroup_bound,
            lower_is_better: true,
        },
        // A model trained on shuffled outcomes scores no better (no lower) than age and sex.
        Requirement::Within {
            value: "permuted_minus_age_sex_ibs".into(),
            lo: 0.0,
            hi: f64::MAX,
        },
    ];
    if d_calibration_decides {
        v.push(Requirement::NotRejected {
            p_value: "d_calibration_p".into(),
            alpha: c.d_calibration_alpha,
        });
    }
    v
}

fn cluster(d: &crate::build::Design) -> u64 {
    (d.cycle as u64) * 1_000_000 + d.stratum * 100 + d.psu
}

fn locked(data: &Path, arm: Arm) -> Option<(Run, LockedDetail)> {
    let run: Run = serde_json::from_slice(
        &std::fs::read(
            data.join("runs")
                .join(format!("{}-s1-locked.json", arm.name())),
        )
        .ok()?,
    )
    .ok()?;
    let detail: LockedDetail = serde_json::from_slice(
        &std::fs::read(
            data.join("runs")
                .join(format!("{}-s1-locked-detail.json", arm.name())),
        )
        .ok()?,
    )
    .ok()?;
    Some((run, detail))
}

fn level(s: &Subject, var: &str) -> Option<String> {
    s.observations
        .iter()
        .find(|o| o.var == var && o.t == s.entry)
        .and_then(|o| match &o.value {
            Value::Category(c) => Some(c.clone()),
            _ => None,
        })
}

fn age_band(s: &Subject) -> String {
    match s.entry as u32 {
        0..=39 => "18-39".into(),
        40..=59 => "40-59".into(),
        60..=79 => "60-79".into(),
        _ => "80+".into(),
    }
}

/// The pre-registered criteria against the locked-test results.
pub fn report(data: &Path) -> Result<()> {
    let f = frozen(data)?;
    let design = designs(data)?;
    let mut out = String::new();
    let (Some((cand, cdet)), Some((_, bdet))) =
        (locked(data, Arm::Horizon), locked(data, Arm::Standard))
    else {
        anyhow::bail!("score the locked test for horizon and standard first (final --arm horizon / --arm standard)");
    };
    writeln!(
        out,
        "# lifecourse report\n\ndataset {}\npartition {}\n",
        f.digests.0, f.digests.1
    )?;
    let mut evidence = Evidence::default();

    // 1. Primary: paired per-subject IBS difference, cluster bootstrap.
    let bterms: HashMap<&str, f64> = bdet
        .ibs_terms
        .iter()
        .map(|(id, x)| (id.as_str(), *x))
        .collect();
    let (mut diffs, mut weights, mut clusters) = (vec![], vec![], vec![]);
    for (id, x) in &cdet.ibs_terms {
        if let (Some(b), Some(s), Some(d)) =
            (bterms.get(id.as_str()), f.subjects.get(id), design.get(id))
        {
            diffs.push(x - b);
            weights.push(s.weight);
            clusters.push(cluster(d));
        }
    }
    let ci = cluster_bootstrap(
        &diffs,
        &weights,
        &clusters,
        BOOTSTRAP_REPS,
        0.95,
        BOOTSTRAP_SEED,
    )
    .context("too few clusters for an interval")?;
    writeln!(
        out,
        "## Primary: integrated Brier score 0-15 years, all-cause death\n"
    )?;
    let ibs = |r: &Run| r.metrics.ibs_0_15.as_ref().map_or(f64::NAN, |m| m.value);
    writeln!(
        out,
        "| arm | IBS 0-15 | n | deaths by 15 y |\n|---|---|---|---|"
    )?;
    for arm in [Arm::Horizon, Arm::Additive, Arm::Standard, Arm::AgeSex] {
        if let Some((r, _)) = locked(data, arm) {
            let m = r.metrics.ibs_0_15.as_ref();
            writeln!(
                out,
                "| {} | {:.5} | {} | {} |",
                arm.name(),
                ibs(&r),
                m.map_or(0, |m| m.n),
                m.map_or(0, |m| m.events)
            )?;
        }
    }
    // perf-number: a confidence level of an interval, not a measurement
    writeln!(out, "\nhorizon minus standard: {:+.5}, 95% design-based interval [{:+.5}, {:+.5}] over {} subjects\n", ci.estimate, ci.lo, ci.hi, diffs.len())?;
    evidence
        .intervals
        .insert("ibs_0_15_diff".into(), (ci.lo, ci.hi));

    // 2. Calibration at 10 years: slope (point) and intercept (interval).
    let ten: Vec<&Subject> = f
        .partition
        .locked
        .iter()
        .filter_map(|id| f.subjects.get(id))
        .filter(|s| f.horizons.supports(s, 10.0))
        .collect();
    let risk: HashMap<&str, f64> = cdet
        .risk_10
        .iter()
        .map(|(id, r)| (id.as_str(), *r))
        .collect();
    let subjects10: Vec<Subject> = ten.iter().map(|s| (*s).clone()).collect();
    let obs10: Vec<Obs> = observed(&subjects10, &CODES)
        .into_iter()
        .map(|o| Obs {
            cause: o.cause.map(|_| 0),
            ..o
        })
        .collect();
    let r10: Vec<f64> = subjects10
        .iter()
        .map(|s| risk.get(s.subject_id.as_str()).copied().unwrap_or(f64::NAN))
        .collect();
    let cl10: Vec<u64> = subjects10
        .iter()
        .map(|s| design.get(&s.subject_id).map_or(0, cluster))
        .collect();
    let intercept = cluster_bootstrap_by(&cl10, BOOTSTRAP_REPS, 0.95, BOOTSTRAP_SEED, |ix| {
        let o: Vec<Obs> = ix.iter().map(|&i| obs10[i]).collect();
        let r: Vec<f64> = ix.iter().map(|&i| r10[i]).collect();
        let g = censoring(&o);
        let c = at_horizon(&r, &o, 0, 10.0, &g, 10);
        c.intercept.is_finite().then_some(c.intercept)
    });
    // Ten-year accuracy and discrimination with design-based intervals, for
    // the candidate and the baseline (reported; the primary decides).
    // perf-number: a confidence level of an interval, not a measurement
    writeln!(out, "## Ten-year Brier score and concordance, locked test (95% design-based intervals)\n\n| arm | Brier 10 | Uno C 10 |\n|---|---|---|")?;
    for (name, det) in [("horizon", &cdet), ("standard", &bdet)] {
        let by_id: HashMap<&str, f64> = det
            .risk_10
            .iter()
            .map(|(id, r)| (id.as_str(), *r))
            .collect();
        let r: Vec<f64> = subjects10
            .iter()
            .map(|s| {
                by_id
                    .get(s.subject_id.as_str())
                    .copied()
                    .unwrap_or(f64::NAN)
            })
            .collect();
        let stat = |which: fn(&[f64], &[Obs]) -> Option<f64>| {
            let point = which(&r, &obs10);
            let iv = cluster_bootstrap_by(&cl10, BOOTSTRAP_REPS, 0.95, BOOTSTRAP_SEED, |ix| {
                let o: Vec<Obs> = ix.iter().map(|&i| obs10[i]).collect();
                let rr: Vec<f64> = ix.iter().map(|&i| r[i]).collect();
                which(&rr, &o)
            });
            match (point, iv) {
                (Some(p), Some(i)) => format!("{p:.4} [{:.4}, {:.4}]", i.lo, i.hi),
                _ => "not measured".into(),
            }
        };
        let brier10 = |r: &[f64], o: &[Obs]| brier(r, o, 0, 10.0, &censoring(o));
        let uno10 = |r: &[f64], o: &[Obs]| uno(r, o, 0, 10.0, &censoring(o));
        writeln!(out, "| {name} | {} | {} |", stat(brier10), stat(uno10))?;
    }
    writeln!(out)?;
    let crit = preregistered();
    writeln!(out, "## Calibration of the candidate at 10 years\n")?;
    if let Some(c) = &cand.metrics.calibration_10 {
        writeln!(
            out,
            "slope {:.3}, intercept {:+.3}, observed/expected {:.3}, mean decile gap {:.4} (n {})",
            c.slope, c.intercept, c.oe_ratio, c.mean_abs_gap, c.n
        )?;
        evidence
            .values
            .insert("calibration_slope_10".into(), c.slope);
    }
    if let Some(i) = &intercept {
        writeln!(
            out,
            // perf-number: a confidence level of an interval, not a measurement
            "intercept 95% design-based interval [{:+.3}, {:+.3}]\n",
            i.lo, i.hi
        )?;
        evidence
            .intervals
            .insert("calibration_intercept_10".into(), (i.lo, i.hi));
    }
    if let Some(p) = cand.metrics.d_calibration_p {
        writeln!(out, "D-calibration p = {p:.4}\n")?;
        evidence.values.insert("d_calibration_p".into(), p);
    }

    // 3. Subgroups: candidate minus baseline IBS per subgroup (point estimates).
    writeln!(out, "## Subgroups (IBS 0-15, candidate minus standard; bound +{})\n\n| subgroup | n | horizon | standard | diff |\n|---|---|---|---|---|", crit.subgroup_bound)?;
    let cterms: HashMap<&str, f64> = cdet
        .ibs_terms
        .iter()
        .map(|(id, x)| (id.as_str(), *x))
        .collect();
    let mut groups: BTreeMap<String, Vec<&str>> = BTreeMap::new();
    for (id, _) in &cdet.ibs_terms {
        let Some(s) = f.subjects.get(id) else {
            continue;
        };
        for (k, v) in [
            ("sex", level(s, "sex")),
            ("age", Some(age_band(s))),
            ("race", level(s, "race_ethnicity")),
            ("cycle", Some(s.source.clone())),
        ] {
            if let Some(v) = v {
                groups.entry(format!("{k}={v}")).or_default().push(id);
            }
        }
    }
    for (g, ids) in &groups {
        let wmean = |t: &HashMap<&str, f64>| {
            let (a, b) = ids
                .iter()
                .filter_map(|id| Some((t.get(id)?, f.subjects[*id].weight)))
                .fold((0.0, 0.0), |(a, b), (x, w)| (a + x * w, b + w));
            a / b
        };
        let (c, b) = (wmean(&cterms), wmean(&bterms));
        evidence.values.insert(format!("subgroup:{g}"), c - b);
        writeln!(
            out,
            "| {g} | {} | {c:.5} | {b:.5} | {:+.5} |",
            ids.len(),
            c - b
        )?;
    }

    // 4. Leakage check: permuted candidate vs age-sex, cross-validated.
    let mean_ibs = |runs: &BTreeMap<(usize, usize), Run>| {
        let v: Vec<f64> = runs
            .values()
            .filter_map(|r| r.metrics.ibs_0_15.as_ref().map(|m| m.value))
            .collect();
        (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64)
    };
    let permuted = permuted_runs(data)?;
    let age_sex = cv_runs(data, Arm::AgeSex, PREREGISTERED_SEED)?;
    writeln!(out, "\n## Leakage check\n")?;
    match (mean_ibs(&permuted), mean_ibs(&age_sex)) {
        (Some(p), Some(a)) => {
            writeln!(
                out,
                "permuted horizon CV mean IBS {p:.5} over {} folds; age-sex {a:.5} over {} folds",
                permuted.len(),
                age_sex.len()
            )?;
            evidence
                .values
                .insert("permuted_minus_age_sex_ibs".into(), p - a);
        }
        _ => writeln!(
            out,
            "not run: cv --arm horizon --permute and cv --arm age-sex are both required"
        )?,
    }

    // 5. Cross-validated comparisons, reported (they inform, the locked test decides).
    writeln!(out, "\n## Cross-validation (mean over folds)\n\nThe two visit arms are secondary: the history read visit by visit through the continuous-time state or attention.\n\n| metric | horizon | additive | standard | age-sex | horizon-state | horizon-attention |\n|---|---|---|---|---|---|---|")?;
    let runs: Vec<BTreeMap<(usize, usize), Run>> = [
        Arm::Horizon,
        Arm::Additive,
        Arm::Standard,
        Arm::AgeSex,
        Arm::HorizonState,
        Arm::HorizonAttention,
    ]
    .iter()
    .map(|a| cv_runs(data, *a, PREREGISTERED_SEED))
    .collect::<Result<_>>()?;
    for Compared { name, get } in compared_metrics() {
        let cells: Vec<String> = runs
            .iter()
            .map(|r| {
                let v: Vec<f64> = r.values().filter_map(|x| get(&x.metrics)).collect();
                if v.is_empty() {
                    "-".into()
                } else {
                    format!(
                        "{:.5} ({})",
                        v.iter().sum::<f64>() / v.len() as f64,
                        v.len()
                    )
                }
            })
            .collect();
        writeln!(out, "| {name} | {} |", cells.join(" | "))?;
    }

    // Baselines made elsewhere, scored on the same folds: secondary comparators.
    let baselines = crate::external::names(data);
    if !baselines.is_empty() {
        writeln!(out, "\n## Secondary: external baselines (cross-validation only)\n\nPredictions made by another program (`baselines/`), scored by the same metrics on the same folds; mean over the folds scored (in brackets). Never scored on the locked test.\n\n| metric | {} |\n|---|{}",
            baselines.join(" | "), "---|".repeat(baselines.len()))?;
        let scored: Vec<_> = baselines
            .iter()
            .map(|b| crate::external::scores(data, b))
            .collect::<Result<_>>()?;
        for Compared { name, get } in compared_metrics() {
            let cells: Vec<String> = scored
                .iter()
                .map(|folds| {
                    let v: Vec<f64> = folds.values().filter_map(get).collect();
                    if v.is_empty() {
                        "-".into()
                    } else {
                        format!(
                            "{:.5} ({})",
                            v.iter().sum::<f64>() / v.len() as f64,
                            v.len()
                        )
                    }
                })
                .collect();
            writeln!(out, "| {name} | {} |", cells.join(" | "))?;
        }
    }

    // Seed spread: the same folds retrained with other seeds.
    writeln!(out, "\n## Secondary: how much a result moves with the training seed\n\nCross-validated mean IBS 1-15 per seed on the folds every listed seed ran, and per fold the spread (largest minus smallest) across seeds, averaged over folds.\n\n| arm | seeds | folds | mean IBS per seed | mean per-fold spread |\n|---|---|---|---|---|")?;
    for arm in [Arm::Horizon, Arm::Standard] {
        let by_seed: Vec<(u64, FoldRuns)> = (1..=SEEDS_REPORTED)
            .map(|seed| cv_runs(data, arm, seed).map(|r| (seed, r)))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .filter(|(_, r)| !r.is_empty())
            .collect();
        if by_seed.len() < 2 {
            continue;
        }
        let folds: Vec<(usize, usize)> = by_seed[0]
            .1
            .keys()
            .filter(|k| by_seed.iter().all(|(_, r)| r.contains_key(k)))
            .copied()
            .collect();
        let ibs = |r: &Run| r.metrics.ibs_0_15.as_ref().map(|m| m.value);
        let means: Vec<String> = by_seed
            .iter()
            .map(|(seed, r)| {
                let v: Vec<f64> = folds.iter().filter_map(|k| ibs(&r[k])).collect();
                format!(
                    "s{seed} {:.5}",
                    v.iter().sum::<f64>() / v.len().max(1) as f64
                )
            })
            .collect();
        let spreads: Vec<f64> = folds
            .iter()
            .filter_map(|k| {
                let v: Vec<f64> = by_seed.iter().filter_map(|(_, r)| ibs(&r[k])).collect();
                (v.len() == by_seed.len()).then(|| {
                    v.iter().copied().fold(f64::MIN, f64::max)
                        - v.iter().copied().fold(f64::MAX, f64::min)
                })
            })
            .collect();
        writeln!(
            out,
            "| {} | {} | {} | {} | {:.5} |",
            arm.name(),
            by_seed.len(),
            folds.len(),
            means.join(", "),
            spreads.iter().sum::<f64>() / spreads.len().max(1) as f64
        )?;
    }

    let terms = crate::commands::nhanes_terms();
    writeln!(
        out,
        "\n## Terms\n\n{}: training {:?}, commercial use {:?}, redistribution {:?}",
        terms.name, terms.training, terms.commercial_use, terms.redistribution
    )?;
    for c in &terms.conditions {
        writeln!(out, "- {c}")?;
    }
    writeln!(out, "\n## Amendments after the freeze\n")?;
    let pinned = crate::commands::pinned_amendments(data)?;
    for a in amendments().iter().filter(|a| pinned.contains(&a.number)) {
        writeln!(
            out,
            "- {}: {} ({}). Why: {}",
            a.number, a.change, a.when, a.reason
        )?;
    }
    if pinned.is_empty() {
        writeln!(out, "none")?;
    }
    writeln!(out, "\n## Verdict\n")?;
    let gate = decide(&requirements(&crit, &pinned), &evidence);
    for (req, check) in &gate.checks {
        let what = serde_json::to_string(req)?;
        let detail = check
            .reason
            .clone()
            .or_else(|| check.measured.clone())
            .unwrap_or_default();
        writeln!(
            out,
            "- [{}] {what}: {detail}",
            if check.passed { "x" } else { " " }
        )?;
    }
    writeln!(
        out,
        "\n**{}**",
        if gate.passed() {
            "Every pre-registered criterion is met."
        } else {
            "Not every pre-registered criterion is met; the claim is not made."
        }
    )?;
    writeln!(out, "\n## Secondary, not pre-registered: ten-year risk as an interval\n\nVenn-Abers intervals calibrated on each run's early-stopping subjects.\n\n| arm | calibration subjects | slope raw / merged | O/E raw / merged | Brier raw / merged | width p10 / p50 / p90 |\n|---|---|---|---|---|---|")?;
    for arm in [Arm::Horizon, Arm::Additive, Arm::Standard, Arm::AgeSex] {
        if let Some(iv) = locked(data, arm).and_then(|(r, _)| r.intervals_10) {
            writeln!(
                out,
                "| {} | {} | {:.3} / {:.3} | {:.3} / {:.3} | {:.5} / {:.5} | {:.4} / {:.4} / {:.4} |",
                arm.name(),
                iv.n_calibration,
                iv.raw.calibration.slope,
                iv.merged.calibration.slope,
                iv.raw.calibration.oe_ratio,
                iv.merged.calibration.oe_ratio,
                iv.raw.brier,
                iv.merged.brier,
                iv.width.0,
                iv.width.1,
                iv.width.2
            )?;
        }
    }
    writeln!(out, "\n## Secondary, not pre-registered: seeded ensembles\n\nThe locked test scored by the mean of several models of one arm trained with different seeds on the same subjects; the spread is the largest minus the smallest member's ten-year risk per subject.\n\n| run | IBS 0-15 | Uno C 10 | spread mean | spread p90 |\n|---|---|---|---|---|")?;
    let mut names: Vec<String> = std::fs::read_dir(data.join("runs"))?
        .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
        .filter(|n| n.contains("-ens") && n.ends_with("-locked.json"))
        .collect();
    names.sort();
    for n in names {
        let v: serde_json::Value =
            serde_json::from_slice(&std::fs::read(data.join("runs").join(&n))?)?;
        let num = |x: &serde_json::Value| {
            x.as_f64()
                .map_or("not measured".to_string(), |v| format!("{v:.4}"))
        };
        writeln!(
            out,
            "| {} | {} | {} | {} | {} |",
            n.trim_end_matches(".json"),
            num(&v["metrics"]["ibs_0_15"]["value"]),
            num(&v["metrics"]["uno_c"]["10"]["value"]),
            num(&v["spread_10"][0]),
            num(&v["spread_10"][1])
        )?;
    }
    writeln!(out, "\n## Secondary, not pre-registered: calendar shift\n\nTrained on the non-locked subjects of the cycles before 2009, scored on those of 2009 and later (the locked test is not read). Later cycles are followed for less time: five years is the longest horizon scored.\n\n| arm | test subjects | Brier 5 | Uno C 5 | D-calibration p |\n|---|---|---|---|---|")?;
    for arm in [Arm::Horizon, Arm::Additive, Arm::Standard, Arm::AgeSex] {
        let path = data
            .join("runs")
            .join(format!("{}-s1-temporal-2009.json", arm.name()));
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        let r: Run = serde_json::from_slice(&bytes)
            .with_context(|| format!("reading {}", path.display()))?;
        let m = &r.metrics;
        let show = |x: Option<f64>| x.map_or("not measured".to_string(), |v| format!("{v:.4}"));
        writeln!(
            out,
            "| {} | {} | {} | {} | {} |",
            arm.name(),
            r.n_test,
            show(m.brier.get(&5).map(|x| x.value)),
            show(m.uno_c.get(&5).map(|x| x.value)),
            show(m.d_calibration_p)
        )?;
    }
    std::fs::write(data.join("report.md"), &out)?;
    print!("{out}");
    Ok(())
}

fn permuted_runs(data: &Path) -> Result<BTreeMap<(usize, usize), Run>> {
    let mut out = BTreeMap::new();
    let Ok(dir) = std::fs::read_dir(data.join("runs")) else {
        return Ok(out);
    };
    for e in dir.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        if name.starts_with("horizon-permuted-") && name.ends_with(".json") {
            let run: Run = serde_json::from_slice(&std::fs::read(e.path())?)?;
            if let Some(fold) = run.fold {
                out.insert(fold, run);
            }
        }
    }
    Ok(out)
}
