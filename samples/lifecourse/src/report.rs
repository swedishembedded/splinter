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
use splinter_sdk::model::timeline::survival::calibration::at_horizon;
use splinter_sdk::model::timeline::survival::compare::{cluster_bootstrap, cluster_bootstrap_by};
use splinter_sdk::model::timeline::survival::estimate::censoring;
use splinter_sdk::model::timeline::survival::Obs;
use splinter_sdk::model::timeline::{observed, Subject, Value};

use crate::build::CODES;
use crate::commands::{compared_metrics, cv_runs, designs, frozen, Compared, LockedDetail, Run};
use crate::experiment::Arm;

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

/// The criteria as the gate's requirements on the evidence `report` names.
pub fn requirements(c: &Criteria) -> Vec<Requirement> {
    vec![
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
        Requirement::NotRejected {
            p_value: "d_calibration_p".into(),
            alpha: c.d_calibration_alpha,
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
    ]
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
    let age_sex = cv_runs(data, Arm::AgeSex)?;
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
    writeln!(out, "\n## Cross-validation (mean over folds)\n\n| metric | horizon | additive | standard | age-sex |\n|---|---|---|---|---|")?;
    let runs: Vec<BTreeMap<(usize, usize), Run>> =
        [Arm::Horizon, Arm::Additive, Arm::Standard, Arm::AgeSex]
            .iter()
            .map(|a| cv_runs(data, *a))
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

    writeln!(out, "\n## Verdict\n")?;
    let gate = decide(&requirements(&crit), &evidence);
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
