// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Cause-specific accuracy of a model's out-of-fold predictions (T1 of the
//! secondary estimands, paper section 4.5).
//!
//! Swedish Embedded AB implements validation of risk models whose outcomes
//! arrive years later, censored and competing, for its clients. If your team
//! needs expertise in showing which causes of death a model predicts well and
//! which it does not, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! For every fold, cause and horizon (5, 10 and 15 years, a horizon only on the
//! cycles whose follow-up reaches it): the IPCW Brier score of the cause's
//! cumulative incidence and that of a constant prediction (the test fold's own
//! Aalen-Johansen estimate, the null model), the time-dependent AUC and Uno's C
//! with the other causes competing, and the calibration at the horizon
//! (observed over expected, recalibration intercept and slope). A model that
//! gives no cause-specific incidence has no entries, never zeros. Folds are
//! compared with the corrected resampled t-test, as everywhere in this study.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use splinter_sdk::model::timeline::survival::auc;
use splinter_sdk::model::timeline::survival::brier::brier;
use splinter_sdk::model::timeline::survival::calibration::at_horizon;
use splinter_sdk::model::timeline::survival::compare::corrected_resampled_t;
use splinter_sdk::model::timeline::survival::concordance::uno;
use splinter_sdk::model::timeline::survival::estimate::{aalen_johansen, censoring};
use splinter_sdk::model::timeline::{observed, Subject};

use crate::build::CODES;
use crate::commands::frozen;
use crate::external::{baseline_dir, fold_stem, parse};
use crate::metrics::{all_cause, subset, Horizons, Outlook};

/// Horizons, in years, at which each cause is scored.
pub const HORIZONS: [u32; 3] = [5, 10, 15];

/// One cause at one horizon on one fold.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CauseHorizon {
    /// Subjects whose cycle reaches the horizon.
    pub n: usize,
    /// Events of the cause among them by the horizon.
    pub events: usize,
    /// IPCW Brier score of the model's cumulative incidence.
    pub brier: f64,
    /// The same for the constant Aalen-Johansen prediction.
    pub null_brier: f64,
    pub auc: Option<f64>,
    pub uno_c: Option<f64>,
    pub oe_ratio: f64,
    /// Recalibration intercept and slope; absent for a constant prediction,
    /// which has no spread to recalibrate over.
    pub intercept: Option<f64>,
    pub slope: Option<f64>,
    pub mean_abs_gap: f64,
}

impl CauseHorizon {
    /// Index of prediction accuracy: `1 - Brier / null Brier`.
    pub fn ipa(&self) -> f64 {
        1.0 - self.brier / self.null_brier
    }
}

/// Death code -> horizon -> score, for one fold.
pub type FoldCause = BTreeMap<String, BTreeMap<u32, CauseHorizon>>;

fn finite(x: f64) -> Option<f64> {
    x.is_finite().then_some(x)
}

/// Score `preds` (in the order of `subjects`) per cause and horizon.
pub fn score_fold<P: Outlook>(subjects: &[Subject], preds: &[P], h: &Horizons) -> FoldCause {
    let mut out = FoldCause::new();
    for t in HORIZONS {
        let (subs, ps) = subset(subjects, preds, h, f64::from(t));
        if subs.len() < 2 {
            continue;
        }
        let g = censoring(&all_cause(observed(&subs, &CODES)));
        for code in CODES {
            // The cause of interest first, the others competing.
            let mut order: Vec<&str> = vec![code];
            order.extend(CODES.iter().filter(|c| **c != code));
            let obs = observed(&subs, &order);
            let Some(cif) = ps
                .iter()
                .map(|p| p.cause_cif(code, f64::from(t)))
                .collect::<Option<Vec<f64>>>()
            else {
                continue;
            };
            let t = f64::from(t);
            let null = vec![aalen_johansen(&obs, 0).at(t); obs.len()];
            let (Some(b), Some(nb)) = (brier(&cif, &obs, 0, t, &g), brier(&null, &obs, 0, t, &g))
            else {
                continue;
            };
            let cal = at_horizon(&cif, &obs, 0, t, &g, 10);
            out.entry(code.to_string()).or_default().insert(
                t as u32,
                CauseHorizon {
                    n: subs.len(),
                    events: obs
                        .iter()
                        .filter(|o| o.cause == Some(0) && o.time <= t)
                        .count(),
                    brier: b,
                    null_brier: nb,
                    auc: auc::at(&cif, &obs, 0, t, &g),
                    uno_c: uno(&cif, &obs, 0, t, &g),
                    oe_ratio: cal.oe_ratio,
                    intercept: finite(cal.intercept),
                    slope: finite(cal.slope),
                    mean_abs_gap: cal.mean_abs_gap,
                },
            );
        }
    }
    // A cause nobody died of in the set has no observed-over-expected ratio.
    for per in out.values_mut() {
        per.retain(|_, c| {
            [c.brier, c.null_brier, c.oe_ratio, c.mean_abs_gap]
                .iter()
                .all(|v| v.is_finite())
        });
    }
    out
}

fn scores_dir(dir: &Path) -> PathBuf {
    dir.join("cause-scores")
}

/// Score every fold of baseline `name` that has a prediction file and keep
/// the result beside the predictions.
pub fn score(data: &Path, name: &str) -> Result<()> {
    let dir = baseline_dir(data, name)?;
    let f = frozen(data)?;
    std::fs::create_dir_all(scores_dir(&dir))?;
    let mut scored = 0;
    for repeat in 0..f.partition.repeats.len() {
        for fold in 0..f.partition.spec.folds as usize {
            let stem = fold_stem(repeat, fold);
            let file = dir.join(format!("{stem}.jsonl"));
            let Ok(text) = std::fs::read_to_string(&file) else {
                continue;
            };
            let (_, test) = f.partition.fold(repeat, fold);
            let preds = parse(&text, &test).with_context(|| format!("{}", file.display()))?;
            let subjects: Vec<Subject> = test.iter().map(|id| f.subjects[*id].clone()).collect();
            let result = score_fold(&subjects, &preds, &f.horizons);
            std::fs::write(
                scores_dir(&dir).join(format!("{stem}.json")),
                serde_json::to_vec_pretty(&result)?,
            )?;
            scored += 1;
        }
    }
    if scored == 0 {
        bail!(
            "no prediction files r<repeat>-k<fold>.jsonl in {}",
            dir.display()
        );
    }
    println!("{name}: cause-specific scores for {scored} folds");
    Ok(())
}

fn read_scores(data: &Path, name: &str) -> Result<BTreeMap<(usize, usize), FoldCause>> {
    let dir = scores_dir(&baseline_dir(data, name)?);
    let mut out = BTreeMap::new();
    for e in std::fs::read_dir(&dir).with_context(|| {
        format!(
            "{} has no cause scores; run `causes --score {name}` first",
            name
        )
    })? {
        let path = e?.path();
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let Some((r, k)) = stem
            .strip_prefix('r')
            .and_then(|s| s.split_once("-k"))
            .and_then(|(r, k)| Some((r.parse().ok()?, k.parse().ok()?)))
        else {
            continue;
        };
        out.insert((r, k), serde_json::from_slice(&std::fs::read(&path)?)?);
    }
    Ok(out)
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

/// Mean over folds of `get`, and the paired difference to `reference` with
/// the corrected confidence interval, over the folds that have both.
fn paired(
    folds: &BTreeMap<(usize, usize), FoldCause>,
    reference: &BTreeMap<(usize, usize), FoldCause>,
    code: &str,
    t: u32,
    get: impl Fn(&CauseHorizon) -> Option<f64>,
    test_over_train: f64,
) -> Option<(f64, f64, f64, f64)> {
    let pick = |m: &BTreeMap<(usize, usize), FoldCause>, k: &(usize, usize)| {
        get(m.get(k)?.get(code)?.get(&t)?)
    };
    let pairs: Vec<(f64, f64)> = folds
        .keys()
        .filter_map(|k| Some((pick(folds, k)?, pick(reference, k)?)))
        .collect();
    if pairs.len() < 2 {
        return None;
    }
    let diffs: Vec<f64> = pairs.iter().map(|(a, b)| a - b).collect();
    let tt = corrected_resampled_t(&diffs, test_over_train)?;
    Some((tt.mean, tt.ci95.0, tt.ci95.1, tt.p_two_sided))
}

/// A markdown table per cause and horizon of the models against `reference`
/// (both are baseline names whose cause scores exist).
pub fn report(data: &Path, models: &[String], reference: &str) -> Result<String> {
    let k = frozen(data)?.partition.spec.folds as f64;
    report_with(data, models, reference, 1.0 / (k - 1.0))
}

fn report_with(
    data: &Path,
    models: &[String],
    reference: &str,
    test_over_train: f64,
) -> Result<String> {
    let reference_scores = read_scores(data, reference)?;
    let mut out = String::new();
    for code in CODES {
        for t in HORIZONS {
            writeln!(out, "\n### {code}, {t} years (difference against {reference}; Brier lower better, AUC higher better)\n")?;
            writeln!(
                out,
                "| model | folds | events/fold | Brier | IPA | AUC | Uno C | O/E | slope | dBrier [95% CI] | dAUC [95% CI] |"
            )?;
            writeln!(out, "|---|---|---|---|---|---|---|---|---|---|---|")?;
            for name in models {
                let folds = read_scores(data, name)?;
                let cells: Vec<&CauseHorizon> = folds
                    .values()
                    .filter_map(|f| f.get(code)?.get(&t))
                    .collect();
                if cells.is_empty() {
                    writeln!(out, "| {name} | 0 | | not scored | | | | | | | |")?;
                    continue;
                }
                let col = |g: fn(&CauseHorizon) -> f64| {
                    mean(&cells.iter().map(|c| g(c)).collect::<Vec<_>>())
                };
                let opt = |g: fn(&CauseHorizon) -> Option<f64>| {
                    let v: Vec<f64> = cells.iter().filter_map(|c| g(c)).collect();
                    if v.is_empty() {
                        f64::NAN
                    } else {
                        mean(&v)
                    }
                };
                let delta = |g: fn(&CauseHorizon) -> Option<f64>| {
                    if name == reference {
                        return "reference".to_string();
                    }
                    match paired(&folds, &reference_scores, code, t, g, test_over_train) {
                        Some((d, lo, hi, _)) => format!("{d:+.5} [{lo:+.5}, {hi:+.5}]"),
                        None => "n/a".to_string(),
                    }
                };
                writeln!(
                    out,
                    "| {name} | {} | {:.0} | {:.5} | {:.3} | {:.4} | {:.4} | {:.3} | {:.3} | {} | {} |",
                    cells.len(),
                    col(|c| c.events as f64),
                    col(|c| c.brier),
                    col(CauseHorizon::ipa),
                    opt(|c| c.auc),
                    opt(|c| c.uno_c),
                    col(|c| c.oe_ratio),
                    opt(|c| c.slope),
                    delta(|c| Some(c.brier)),
                    delta(|c| c.auc),
                )?;
            }
        }
    }
    Ok(out)
}

/// Score `models` where needed (`score_first`) and print the report.
pub fn run(data: &Path, models: &[String], reference: &str, score_first: bool) -> Result<()> {
    if score_first {
        for m in models.iter().map(String::as_str).chain([reference]) {
            score(data, m)?;
        }
    }
    println!("{}", report(data, models, reference)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::external::ExternalPrediction;
    use splinter_sdk::model::timeline::{AtRisk, Event};

    const RATE_OTHER: f64 = 0.02;

    /// Deterministic generator: no random-number dependency in the test.
    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// Subjects who die of cause 0 at rate `r_i` (low or high) or of cause 1
    /// at a common rate, censored at `follow_up`.
    fn cohort(n: usize, follow_up: f64) -> (Vec<Subject>, Vec<f64>) {
        let mut s = 11u64;
        let mut subjects = Vec::new();
        let mut rates = Vec::new();
        for i in 0..n {
            let rate = if i % 2 == 0 { 0.01 } else { 0.06 };
            let t0 = -lcg(&mut s).ln() / rate;
            let t1 = -lcg(&mut s).ln() / RATE_OTHER;
            let (t, code) = if t0 < t1 {
                (t0, Some(CODES[0]))
            } else {
                (t1, Some(CODES[1]))
            };
            let (exit, code) = if t <= follow_up {
                (50.0 + t, code)
            } else {
                (50.0 + follow_up, None)
            };
            subjects.push(Subject {
                subject_id: format!("s{i}"),
                group_id: None,
                weight: 1.0,
                source: "c".into(),
                entry: 50.0,
                calendar_at_entry: 2000.0,
                observations: vec![],
                events: code
                    .map(|c| Event {
                        t: exit,
                        code: c.into(),
                    })
                    .into_iter()
                    .collect(),
                at_risk: vec![AtRisk {
                    code: "*".into(),
                    from: 50.0,
                    to: exit,
                }],
            });
            rates.push(rate);
        }
        (subjects, rates)
    }

    fn curve(rate_cause: f64, rate_other: f64) -> Vec<f64> {
        let total = rate_cause + rate_other;
        (1..=15)
            .map(|y| rate_cause / total * (1.0 - (-total * f64::from(y)).exp()))
            .collect()
    }

    fn predictions(rates: &[f64], informed: bool) -> Vec<ExternalPrediction> {
        rates
            .iter()
            .map(|&r| {
                let r = if informed { r } else { 0.035 };
                let c0 = curve(r, RATE_OTHER);
                let c1 = curve(RATE_OTHER, r);
                let c2 = vec![0.0; 15];
                let all: Vec<f64> = (0..15).map(|i| c0[i] + c1[i]).collect();
                let causes = BTreeMap::from([
                    (CODES[0].to_string(), c0),
                    (CODES[1].to_string(), c1),
                    (CODES[2].to_string(), c2),
                ]);
                ExternalPrediction::new(all, Some(causes)).unwrap()
            })
            .collect()
    }

    fn horizons(follow_up: f64) -> Horizons {
        Horizons(BTreeMap::from([("c".to_string(), follow_up)]))
    }

    #[test]
    fn a_model_that_knows_the_rates_beats_the_constant_one_for_the_cause() {
        let (subjects, rates) = cohort(4000, 16.0);
        let good = score_fold(&subjects, &predictions(&rates, true), &horizons(16.0));
        let flat = score_fold(&subjects, &predictions(&rates, false), &horizons(16.0));
        let (g, f) = (&good[CODES[0]][&10], &flat[CODES[0]][&10]);
        assert!(g.brier < f.brier - 0.002, "{} vs {}", g.brier, f.brier);
        assert!(g.auc.unwrap() > 0.65, "{:?}", g.auc);
        assert!(
            (f.auc.unwrap() - 0.5).abs() < 1e-9,
            "a constant prediction ranks nobody"
        );
        assert!(
            g.ipa() > 0.0 && g.oe_ratio > 0.8 && g.oe_ratio < 1.25,
            "{g:?}"
        );
        assert!(g.events > 100 && g.n == 4000);
    }

    #[test]
    fn a_horizon_beyond_the_cycles_follow_up_is_absent() {
        let (subjects, rates) = cohort(500, 12.0);
        let scored = score_fold(&subjects, &predictions(&rates, true), &horizons(12.0));
        let years: Vec<u32> = scored[CODES[0]].keys().copied().collect();
        assert_eq!(years, vec![5, 10]);
    }

    /// Reads only all-cause survival, like a model with no cause curves.
    struct AllCauseOnly;
    impl Outlook for AllCauseOnly {
        fn survival(&self, _: f64) -> f64 {
            0.9
        }
        fn cause_cif(&self, _: &str, _: f64) -> Option<f64> {
            None
        }
    }

    #[test]
    fn a_model_without_cause_curves_has_no_cause_scores_not_zeros() {
        let (subjects, _) = cohort(200, 16.0);
        let preds: Vec<AllCauseOnly> = subjects.iter().map(|_| AllCauseOnly).collect();
        assert!(score_fold(&subjects, &preds, &horizons(16.0)).is_empty());
    }

    fn cell(brier: f64, events: usize) -> CauseHorizon {
        CauseHorizon {
            n: 1000,
            events,
            brier,
            null_brier: 0.05,
            auc: Some(0.9),
            uno_c: None,
            oe_ratio: 1.0,
            intercept: None,
            slope: Some(1.0),
            mean_abs_gap: 0.0,
        }
    }

    fn write_scores(data: &Path, name: &str, briers: [f64; 5]) {
        let dir = scores_dir(&baseline_dir(data, name).unwrap());
        std::fs::create_dir_all(&dir).unwrap();
        for (k, b) in briers.iter().enumerate() {
            let fold: FoldCause = CODES
                .iter()
                .map(|c| (c.to_string(), BTreeMap::from([(10, cell(*b, 30 + k))])))
                .collect();
            std::fs::write(
                dir.join(format!("r0-k{k}.json")),
                serde_json::to_vec(&fold).unwrap(),
            )
            .unwrap();
        }
    }

    #[test]
    fn the_report_compares_every_model_with_the_reference_over_the_folds() {
        let dir = tempfile::tempdir().unwrap();
        write_scores(dir.path(), "ref", [0.030, 0.031, 0.029, 0.030, 0.032]);
        write_scores(dir.path(), "better", [0.028, 0.030, 0.027, 0.028, 0.031]);
        let text = report_with(dir.path(), &["ref".into(), "better".into()], "ref", 0.25).unwrap();
        assert!(text.contains("### death:cvd, 10 years"));
        let row = text
            .lines()
            .find(|l| l.starts_with("| better | 5 |"))
            .unwrap();
        assert!(row.contains("-0.00160 ["), "{row}");
        assert!(text
            .lines()
            .any(|l| l.starts_with("| ref | 5 |") && l.contains("reference")));
        // 5 and 15 years were never scored: the tables say so rather than inventing numbers.
        assert!(text.contains("| better | 0 | | not scored |"));
    }
}
