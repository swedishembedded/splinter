// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Restricted mean survival time and mortality-equivalent age (T3 of the
//! secondary estimands, paper section 4.5).
//!
//! Swedish Embedded AB implements validation of risk models whose outcomes
//! arrive years later, censored and competing, for its clients. If your team
//! needs expertise in turning a survival model into an expected time lived and
//! checking it against what happened, you can procure our services by sending
//! an email to info@swedishembedded.com.
//!
//! A model's predicted curve gives each subject an expected time lived in the
//! next `tau` years. Calibration compares its mean in equal-weight risk groups
//! with the Kaplan-Meier RMST observed there, only on the cycles whose
//! follow-up reaches `tau`. The mortality-equivalent age is the age at which a
//! Gompertz life table, fitted per sex to the training subjects of the same
//! fold, gives the same expected time: a re-expression of one model against
//! one reference, not a measure of biological ageing. Ages outside the table's
//! range are clamped and the clamps counted.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::Serialize;
use splinter_sdk::model::timeline::survival::concordance::uno;
use splinter_sdk::model::timeline::survival::estimate::censoring;
use splinter_sdk::model::timeline::survival::rmst::{
    calibration, curve_rmst, Calibration, Follow, Gompertz,
};
use splinter_sdk::model::timeline::{observed, Subject, Value};

use crate::build::CODES;
use crate::commands::frozen;
use crate::contributing::{curves, Source};
use crate::metrics::all_cause;

/// The ages the equivalent-age table covers.
const AGE_RANGE: (f64, f64) = (18.0, 85.0);
const GROUPS: usize = 10;

/// One subject's row in the equivalent-age file.
#[derive(Serialize, Debug, PartialEq)]
pub struct Row {
    pub subject_id: String,
    pub age: f64,
    pub sex: String,
    pub rmst: f64,
    pub equivalent_age: f64,
    /// Equivalent age minus age.
    pub acceleration: f64,
    pub clamped: bool,
}

fn sex(s: &Subject) -> String {
    s.observations
        .iter()
        .find(|o| o.var == "sex")
        .and_then(|o| match &o.value {
            Value::Category(c) => Some(c.clone()),
            _ => None,
        })
        .unwrap_or_default()
}

/// The RMST over `tau` years of a cumulative-incidence curve at years 1..=15.
pub fn rmst_of(cif: &[f64], tau: f64) -> f64 {
    let survival: Vec<f64> = cif.iter().map(|c| 1.0 - c).collect();
    curve_rmst(&survival, tau)
}

/// A Gompertz table per sex from `training` subjects, who are followed from
/// their examination age to death or censoring.
pub fn tables(training: &[&Subject]) -> Result<HashMap<String, Gompertz>> {
    let subs: Vec<Subject> = training.iter().map(|s| (*s).clone()).collect();
    let obs = all_cause(observed(&subs, &CODES));
    let mut by: HashMap<String, Vec<Follow>> = HashMap::new();
    for (s, o) in subs.iter().zip(obs) {
        by.entry(sex(s)).or_default().push(Follow {
            entry: s.entry,
            exit: s.entry + o.time,
            died: o.cause.is_some(),
            weight: s.weight,
        });
    }
    by.into_iter()
        .map(|(k, v)| {
            if !v.iter().any(|f| f.died) {
                bail!("no deaths among the training subjects of sex {k:?}");
            }
            Ok((k, Gompertz::fit(&v)))
        })
        .collect()
}

/// What one model gives on the pooled out-of-fold set of one repeat.
pub struct Evaluation {
    pub n: usize,
    pub deaths: usize,
    pub calibration: Calibration,
    pub uno_c: Option<f64>,
    pub rows: Vec<Row>,
}

/// RMST over `tau` for every subject of the repeat whose cycle supports it,
/// its calibration, and the equivalent ages.
pub fn evaluate(data: &Path, model: &str, repeat: usize, tau: f64) -> Result<Evaluation> {
    let f = frozen(data)?;
    let pred = curves(data, &f, &Source::Ranker(model.to_string()), repeat, "")?;
    let mut rows = Vec::new();
    let (mut subs, mut rmst) = (Vec::<Subject>::new(), Vec::<f64>::new());
    for fold in 0..f.partition.spec.folds as usize {
        let (train, test) = f.partition.fold(repeat, fold);
        let table = tables(&train.iter().map(|id| &f.subjects[*id]).collect::<Vec<_>>())?;
        for id in test {
            let s = &f.subjects[id];
            if !f.horizons.supports(s, tau) {
                continue;
            }
            let r = rmst_of(&pred[id], tau);
            let g = table
                .get(&sex(s))
                .with_context(|| format!("{id}: no table for sex {:?}", sex(s)))?;
            let (equivalent_age, clamped) = g.equivalent_age(r, tau, AGE_RANGE.0, AGE_RANGE.1);
            rows.push(Row {
                subject_id: id.to_string(),
                age: s.entry,
                sex: sex(s),
                rmst: r,
                equivalent_age,
                acceleration: equivalent_age - s.entry,
                clamped,
            });
            subs.push(s.clone());
            rmst.push(r);
        }
    }
    let obs = all_cause(observed(&subs, &CODES));
    let g = censoring(&obs);
    let risk: Vec<f64> = rmst.iter().map(|r| -r).collect();
    Ok(Evaluation {
        n: subs.len(),
        deaths: obs
            .iter()
            .filter(|o| o.cause.is_some() && o.time <= tau)
            .count(),
        calibration: calibration(&rmst, &obs, tau, GROUPS),
        uno_c: uno(&risk, &obs, 0, tau, &g),
        rows,
    })
}

/// Markdown for `models`, and the equivalent ages written to `aa_dir`.
pub fn report(
    data: &Path,
    models: &[String],
    repeat: usize,
    tau: f64,
    aa_dir: Option<&Path>,
) -> Result<String> {
    let mut out = String::new();
    writeln!(
        out,
        "\n### Restricted mean survival time over {tau} years, repeat {repeat}\n"
    )?;
    writeln!(out, "Calibrated by the rule of section 4.5 iff the mean absolute gap is at most 0.25 years and the slope lies in [0.9, 1.1].\n")?;
    writeln!(out, "| model | subjects | deaths | mean predicted | mean observed | mean absolute gap (y) | slope | intercept | Uno C of -RMST | clamped ages |")?;
    writeln!(out, "|---|---|---|---|---|---|---|---|---|---|")?;
    let mut tables_md = String::new();
    for m in models {
        let e = evaluate(data, m, repeat, tau)?;
        let w: f64 = e.calibration.groups.iter().map(|g| g.weight).sum();
        let mean = |g: fn(&splinter_sdk::model::timeline::survival::rmst::Group) -> f64| {
            e.calibration
                .groups
                .iter()
                .map(|x| x.weight * g(x))
                .sum::<f64>()
                / w
        };
        let clamped = e.rows.iter().filter(|r| r.clamped).count();
        writeln!(
            out,
            "| {m} | {} | {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {} | {clamped} |",
            e.n,
            e.deaths,
            mean(|g| g.expected),
            mean(|g| g.observed),
            e.calibration.mean_abs_gap,
            e.calibration.slope,
            e.calibration.intercept,
            e.uno_c.map_or("n/a".to_string(), |c| format!("{c:.4}")),
        )?;
        writeln!(
            tables_md,
            "\n{m}: predicted and observed RMST by risk group (ordered from shortest predicted)\n"
        )?;
        writeln!(
            tables_md,
            "| group | weight share | predicted | observed |\n|---|---|---|---|"
        )?;
        for (i, g) in e.calibration.groups.iter().enumerate() {
            writeln!(
                tables_md,
                "| {} | {:.3} | {:.3} | {:.3} |",
                i + 1,
                g.weight / w,
                g.expected,
                g.observed
            )?;
        }
        if let Some(dir) = aa_dir {
            std::fs::create_dir_all(dir)?;
            let path = dir.join(format!("aa-{m}-r{repeat}-t{tau}.jsonl"));
            let mut text = String::new();
            for r in &e.rows {
                text.push_str(&serde_json::to_string(r)?);
                text.push('\n');
            }
            std::fs::write(&path, text).with_context(|| format!("{}", path.display()))?;
        }
    }
    out.push_str(&tables_md);
    Ok(out)
}

/// Print the report.
pub fn run(
    data: &Path,
    models: &[String],
    repeat: usize,
    tau: f64,
    aa_dir: Option<&Path>,
) -> Result<()> {
    if !(tau > 0.0 && tau <= 15.0) {
        bail!("tau must be in (0, 15] years");
    }
    println!("{}", report(data, models, repeat, tau, aa_dir)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use splinter_sdk::model::timeline::{AtRisk, Event, Observation};

    #[test]
    fn rmst_of_a_curve_is_the_area_under_one_minus_incidence() {
        // Constant 1% incidence per year: survival 0.99, 0.98, ... trapezoid area to 10 years.
        let cif: Vec<f64> = (1..=15).map(|y| 0.01 * f64::from(y)).collect();
        let expected: f64 = 0.5 * (1.0 + 0.99)
            + (1..10)
                .map(|k| 0.5 * ((1.0 - 0.01 * f64::from(k)) + (1.0 - 0.01 * f64::from(k + 1))))
                .sum::<f64>();
        assert!((rmst_of(&cif, 10.0) - expected).abs() < 1e-12);
    }

    /// Subjects dying at a Gompertz rate; the equivalent age of a subject
    /// whose predicted RMST is that of age 60 is 60.
    #[test]
    fn a_table_fitted_to_a_cohort_returns_the_age_whose_expectation_a_subject_has() {
        let truth = Gompertz { a: -10.0, b: 0.09 };
        let mut state = 5u64;
        let mut lcg = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        let mut subjects = Vec::new();
        for i in 0..40_000 {
            let entry = 20.0 + 60.0 * lcg();
            let wait =
                (1.0 + (-lcg().ln()) * truth.b / (truth.a + truth.b * entry).exp()).ln() / truth.b;
            let end = entry + 15.0;
            let (exit, died) = if wait <= 15.0 {
                (entry + wait, true)
            } else {
                (end, false)
            };
            subjects.push(Subject {
                subject_id: format!("s{i}"),
                group_id: None,
                weight: 1.0,
                source: "c".into(),
                entry,
                calendar_at_entry: 2000.0,
                observations: vec![Observation {
                    t: entry,
                    var: "sex".into(),
                    value: Value::Category("f".into()),
                    unit: None,
                }],
                events: died
                    .then(|| Event {
                        t: exit,
                        code: CODES[0].into(),
                    })
                    .into_iter()
                    .collect(),
                at_risk: vec![AtRisk {
                    code: "*".into(),
                    from: entry,
                    to: exit,
                }],
            });
        }
        let t = tables(&subjects.iter().collect::<Vec<_>>()).unwrap();
        let g = t["f"];
        assert!((g.b - truth.b).abs() < 0.005, "{g:?}");
        let (age, clamped) = g.equivalent_age(truth.rmst_from(60.0, 10.0), 10.0, 18.0, 85.0);
        assert!(!clamped && (age - 60.0).abs() < 1.5, "{age}");
    }
}
