// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Pooled out-of-fold evaluation with design-based bootstrap intervals
//! (paper sections 4.2 and 5.5).
//!
//! Swedish Embedded AB implements validation of risk models whose outcomes
//! arrive years later, censored and competing, for its clients. If your team
//! needs expertise in intervals that respect a survey's clusters when
//! comparing two risk models, you can procure our services by sending an email
//! to info@swedishembedded.com.
//!
//! The out-of-fold predictions of one repeat give every subject exactly one
//! prediction, so models can be compared on the pooled set and uncertainty
//! taken from resampling the survey's clusters (cycle, stratum, PSU) rather
//! than from the folds. Two uses:
//!
//! * [`difference`]: the paired per-subject difference of the integrated Brier
//!   score over 1 to 15 years and of the Brier score at 10 years between two
//!   models, with a cluster-bootstrap interval of its survey-weighted mean;
//! * [`calibration`]: per cause (and all-cause) at a horizon, the observed over
//!   expected ratio and the recalibration slope with cluster-bootstrap intervals
//!   (the censoring distribution is re-estimated on every resample), and the
//!   point values of the integrated calibration index and the median and
//!   90th-percentile calibration error.
//!
//! A model is named as a baseline (`cox-net-all`) or a recipe (`recipe:d32-s1`);
//! both keep predictions as `r<repeat>-k<fold>.jsonl` files.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use splinter_sdk::model::timeline::survival::brier::brier_terms;
use splinter_sdk::model::timeline::survival::calibration::at_horizon;
use splinter_sdk::model::timeline::survival::compare::{
    cluster_bootstrap, cluster_bootstrap_by, Interval,
};
use splinter_sdk::model::timeline::survival::curve::errors_at_horizon;
use splinter_sdk::model::timeline::survival::estimate::censoring;
use splinter_sdk::model::timeline::survival::Obs;
use splinter_sdk::model::timeline::{observed, Subject};

use crate::build::{Design, CODES};
use crate::commands::{designs, frozen, Frozen};
use crate::external::{baseline_dir, fold_stem, parse, ExternalPrediction};
use crate::metrics::{all_cause, ibs_terms, subset, Outlook};
use crate::recipe_run::recipe_dir;
use crate::report::cluster;

#[cfg(not(test))]
const REPS: usize = 1000;
/// Fewer replicates keep the specifications fast; they test the sign of an interval.
#[cfg(test)]
const REPS: usize = 100;
const SEED: u64 = 20_261_008;
/// Smoothing window of the calibration curve, as a share of the weight.
const SPAN: f64 = 0.3;
/// The horizon of the calibration summaries, in years.
pub const HORIZON: f64 = 10.0;

fn prediction_dir(data: &Path, name: &str) -> Result<PathBuf> {
    match name.strip_prefix("recipe:") {
        Some(r) => recipe_dir(data, r),
        None => baseline_dir(data, name),
    }
}

/// Every subject of `repeat` with its model's prediction, in subject-id order.
pub struct Pooled {
    pub subjects: Vec<Subject>,
    pub preds: Vec<ExternalPrediction>,
}

/// Read the prediction files of `name` for `repeat` (all folds).
pub fn load(data: &Path, f: &Frozen, name: &str, repeat: usize) -> Result<Pooled> {
    let dir = prediction_dir(data, name)?;
    let mut rows: Vec<(String, ExternalPrediction)> = Vec::new();
    for fold in 0..f.partition.spec.folds as usize {
        let file = dir.join(format!("{}.jsonl", fold_stem(repeat, fold)));
        let text = std::fs::read_to_string(&file).with_context(|| format!("{}", file.display()))?;
        let (_, test) = f.partition.fold(repeat, fold);
        let preds = parse(&text, &test).with_context(|| format!("{}", file.display()))?;
        rows.extend(test.iter().map(|id| id.to_string()).zip(preds));
    }
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(Pooled {
        subjects: rows.iter().map(|(id, _)| f.subjects[id].clone()).collect(),
        preds: rows.into_iter().map(|(_, p)| p).collect(),
    })
}

fn clusters(design: &HashMap<String, Design>, subjects: &[Subject]) -> Vec<u64> {
    subjects
        .iter()
        .map(|s| design.get(&s.subject_id).map_or(0, cluster))
        .collect()
}

/// A paired difference with its interval.
struct Diff {
    label: &'static str,
    n: usize,
    estimate: f64,
    lo: f64,
    hi: f64,
}

/// `a` minus `b`: integrated Brier score over 1 to 15 years and Brier score at
/// 10 years, each as the survey-weighted mean of the per-subject difference.
pub fn difference(data: &Path, a: &str, b: &str, repeat: usize) -> Result<String> {
    let f = frozen(data)?;
    let design = designs(data)?;
    let (pa, pb) = (load(data, &f, a, repeat)?, load(data, &f, b, repeat)?);
    let mut out = String::new();
    writeln!(out, "\n{a} minus {b}, repeat {repeat}, pooled out-of-fold predictions, {REPS} cluster resamples\n")?;
    writeln!(
        out,
        "| metric | subjects | difference | 95% interval |\n|---|---|---|---|"
    )?;
    let mut diffs: Vec<Diff> = Vec::new();
    // Integrated Brier score.
    let (ta, tb) = (
        ibs_terms(&pa.subjects, &pa.preds, &f.horizons),
        ibs_terms(&pb.subjects, &pb.preds, &f.horizons),
    );
    diffs.push(paired(
        "integrated Brier, 0 to 15 years",
        &f,
        &design,
        &ta,
        &tb,
    )?);
    // Brier score at ten years, per subject.
    let per_subject = |p: &Pooled| -> Vec<(String, f64)> {
        let (subs, ps) = subset(&p.subjects, &p.preds, &f.horizons, HORIZON);
        let obs = all_cause(observed(&subs, &CODES));
        let g = censoring(&obs);
        let risk: Vec<f64> = ps.iter().map(|q| 1.0 - q.survival(HORIZON)).collect();
        let terms = brier_terms(&risk, &obs, 0, HORIZON, &g).unwrap_or_default();
        subs.iter()
            .map(|s| s.subject_id.clone())
            .zip(terms)
            .collect()
    };
    diffs.push(paired(
        "Brier, 10 years",
        &f,
        &design,
        &per_subject(&pa),
        &per_subject(&pb),
    )?);
    for d in diffs {
        writeln!(
            out,
            "| {} | {} | {:+.5} | [{:+.5}, {:+.5}] |",
            d.label, d.n, d.estimate, d.lo, d.hi
        )?;
    }
    Ok(out)
}

fn paired(
    label: &'static str,
    f: &Frozen,
    design: &HashMap<String, Design>,
    a: &[(String, f64)],
    b: &[(String, f64)],
) -> Result<Diff> {
    let bmap: HashMap<&str, f64> = b.iter().map(|(id, x)| (id.as_str(), *x)).collect();
    let (mut v, mut w, mut c) = (vec![], vec![], vec![]);
    for (id, x) in a {
        if let (Some(y), Some(s), Some(d)) =
            (bmap.get(id.as_str()), f.subjects.get(id), design.get(id))
        {
            v.push(x - y);
            w.push(s.weight);
            c.push(cluster(d));
        }
    }
    let iv = cluster_bootstrap(&v, &w, &c, REPS, 0.95, SEED)
        .context("too few clusters for an interval")?;
    Ok(Diff {
        label,
        n: v.len(),
        estimate: iv.estimate,
        lo: iv.lo,
        hi: iv.hi,
    })
}

/// Observed over expected, slope, intercept at a horizon for `cif`.
fn calibrate(cif: &[f64], obs: &[Obs], t: f64) -> Option<(f64, f64)> {
    let g = censoring(&all_cause(obs.to_vec()));
    let c = at_horizon(cif, obs, 0, t, &g, 10);
    (c.oe_ratio.is_finite() && c.slope.is_finite()).then_some((c.oe_ratio, c.slope))
}

/// Calibration per cause at `HORIZON`, as markdown.
pub fn calibration(data: &Path, models: &[String], repeat: usize) -> Result<String> {
    if models.is_empty() {
        bail!("name at least one model");
    }
    let f = frozen(data)?;
    let design = designs(data)?;
    let mut out = String::new();
    writeln!(
        out,
        "\nCalibration at {HORIZON} years, repeat {repeat}, pooled out-of-fold predictions, {REPS} cluster resamples. Calibrated by the rule of section 4.5 iff the interval of observed over expected contains 1 and the slope lies in [0.8, 1.25].\n"
    )?;
    writeln!(out, "| cause | model | subjects | events | O/E [95% CI] | slope [95% CI] | ICI | E50 | E90 | calibrated |")?;
    writeln!(out, "|---|---|---|---|---|---|---|---|---|---|")?;
    for m in models {
        let p = load(data, &f, m, repeat)?;
        let (subs, ps) = subset(&p.subjects, &p.preds, &f.horizons, HORIZON);
        let cl = clusters(&design, &subs);
        for k in 0..=CODES.len() {
            let (name, obs, cif): (String, Vec<Obs>, Option<Vec<f64>>) = if k == 0 {
                (
                    "all causes".into(),
                    all_cause(observed(&subs, &CODES)),
                    Some(ps.iter().map(|q| 1.0 - q.survival(HORIZON)).collect()),
                )
            } else {
                let code = CODES[k - 1];
                let mut order: Vec<&str> = vec![code];
                order.extend(CODES.iter().filter(|c| **c != code));
                (
                    code.to_string(),
                    observed(&subs, &order),
                    ps.iter()
                        .map(|q| q.cause_cif(code, HORIZON))
                        .collect::<Option<Vec<f64>>>(),
                )
            };
            let Some(cif) = cif else { continue };
            let events = obs
                .iter()
                .filter(|o| o.cause == Some(0) && o.time <= HORIZON)
                .count();
            let Some((oe, slope)) = calibrate(&cif, &obs, HORIZON) else {
                writeln!(
                    out,
                    "| {name} | {m} | {} | {events} | not measurable | | | | | |",
                    subs.len()
                )?;
                continue;
            };
            let stat = |ix: &[usize], pick: fn((f64, f64)) -> f64| -> Option<f64> {
                let o: Vec<Obs> = ix.iter().map(|&i| obs[i]).collect();
                let c: Vec<f64> = ix.iter().map(|&i| cif[i]).collect();
                calibrate(&c, &o, HORIZON).map(pick)
            };
            let oe_iv = cluster_bootstrap_by(&cl, REPS, 0.95, SEED, |ix| stat(ix, |x| x.0));
            let slope_iv = cluster_bootstrap_by(&cl, REPS, 0.95, SEED, |ix| stat(ix, |x| x.1));
            let g = censoring(&all_cause(obs.clone()));
            let curve = errors_at_horizon(&cif, &obs, 0, HORIZON, &g, SPAN);
            let show = |iv: &Option<Interval>, v: f64| match iv {
                Some(i) => format!("{v:.3} [{:.3}, {:.3}]", i.lo, i.hi),
                None => format!("{v:.3} [n/a]"),
            };
            let ok = oe_iv.as_ref().is_some_and(|i| i.lo <= 1.0 && 1.0 <= i.hi)
                && (0.8..=1.25).contains(&slope);
            let (ici, e50, e90) = curve.map_or(("n/a".into(), "n/a".into(), "n/a".into()), |c| {
                (
                    format!("{:.4}", c.ici),
                    format!("{:.4}", c.e50),
                    format!("{:.4}", c.e90),
                )
            });
            writeln!(
                out,
                "| {name} | {m} | {} | {events} | {} | {} | {ici} | {e50} | {e90} | {} |",
                subs.len(),
                show(&oe_iv, oe),
                show(&slope_iv, slope),
                if ok { "yes" } else { "no" }
            )?;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(s: &mut u64) -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    #[test]
    fn a_calibrated_prediction_passes_the_rule_and_a_distorted_one_fails_on_slope() {
        let mut s = 9u64;
        let (mut truth, mut obs) = (vec![], vec![]);
        for _ in 0..8000 {
            let rate = 0.03 * (3.0 * lcg(&mut s) - 1.5).exp();
            let time = -lcg(&mut s).ln() / rate;
            let censor = 12.0 + 8.0 * lcg(&mut s);
            obs.push(if time <= censor {
                Obs::event(time, 0)
            } else {
                Obs::censored(censor)
            });
            truth.push(1.0 - (-rate * 10.0).exp());
        }
        let cl: Vec<u64> = (0..obs.len() as u64).map(|i| i / 6).collect();
        let (oe, slope) = calibrate(&truth, &obs, 10.0).unwrap();
        assert!(
            (oe - 1.0).abs() < 0.1 && (slope - 1.0).abs() < 0.25,
            "{oe} {slope}"
        );
        let iv = cluster_bootstrap_by(&cl, REPS, 0.95, SEED, |ix| {
            let o: Vec<Obs> = ix.iter().map(|&i| obs[i]).collect();
            let c: Vec<f64> = ix.iter().map(|&i| truth[i]).collect();
            calibrate(&c, &o, 10.0).map(|x| x.0)
        })
        .unwrap();
        assert!(iv.lo <= 1.0 && 1.0 <= iv.hi, "{iv:?}");
        // Predictions too extreme: the slope falls well below the bound.
        let bent: Vec<f64> = truth
            .iter()
            .map(|p| 1.0 / (1.0 + (-2.0 * (p / (1.0 - p)).ln()).exp()))
            .collect();
        let (_, bent_slope) = calibrate(&bent, &obs, 10.0).unwrap();
        assert!(bent_slope < 0.8, "{bent_slope}");
    }
}
