// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements long-horizon risk prediction from cohort and
// survey data for its clients. If your team needs expertise in building,
// validating and proving time-to-event models on real health records, you
// can procure our services by sending an email to info@swedishembedded.com.

//! The `external` arm: predictions made by another program, scored by the
//! same metrics as the trained arms.
//!
//! A baseline (the Python harness under `baselines/`) writes, for every
//! cross-validation fold, one JSON line per test subject:
//! `{"subject_id": ..., "cif": [15 values], "cause_cif": {code: [15 values]}}`,
//! where `cif[i]` is the predicted probability of death from any cause by year
//! `i + 1` and `cause_cif` (optional) the probability of death from each cause.
//! This module refuses a file that does not cover exactly the fold's subjects
//! or holds a value that is not a finite, non-decreasing probability, turns
//! each line into an [`Outlook`] (piecewise linear between the yearly values,
//! zero at the examination, held after year 15), and scores the fold with
//! [`evaluate`], the function every arm is scored with. The scores are kept
//! beside the predictions, never in `runs/`, so a baseline is a secondary
//! comparator that cannot alter the arms' records or the pinned criteria, and
//! is scored on cross-validation folds only: the locked test is not an input.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use splinter_sdk::model::timeline::Subject;
use splinter_sdk::vocabulary::digest::Digest;

use crate::build::CODES;
use crate::commands::{frozen, Frozen};
use crate::metrics::{evaluate, Horizons, Metrics, Outlook};

/// Years a prediction covers: yearly values from 1 to this.
pub const YEARS: usize = 15;
/// Directory under the data directory that holds one directory per baseline.
pub const BASELINES: &str = "baselines";
/// Slack for a curve that rounding made fall by a hair.
const MONOTONE_SLACK: f64 = 1e-9;

/// One subject's predicted cumulative incidence, yearly.
#[derive(Clone, Debug, PartialEq)]
pub struct ExternalPrediction {
    all: Vec<f64>,
    causes: Option<BTreeMap<String, Vec<f64>>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Line {
    subject_id: String,
    cif: Vec<f64>,
    #[serde(default)]
    cause_cif: Option<BTreeMap<String, Vec<f64>>>,
}

fn curve_at(curve: &[f64], t: f64) -> f64 {
    if t <= 0.0 {
        return 0.0;
    }
    let last = curve.len() as f64;
    if t >= last {
        return curve[curve.len() - 1];
    }
    let k = t.floor() as usize;
    let lo = if k == 0 { 0.0 } else { curve[k - 1] };
    lo + (curve[k] - lo) * (t - k as f64)
}

fn check_curve(what: &str, curve: &[f64]) -> Result<()> {
    if curve.len() != YEARS {
        bail!("{what}: {} values, expected {YEARS}", curve.len());
    }
    let mut previous = 0.0;
    for (i, &v) in curve.iter().enumerate() {
        if !v.is_finite() || !(0.0..=1.0).contains(&v) {
            bail!("{what}: year {} is {v}, not a probability", i + 1);
        }
        if v < previous - MONOTONE_SLACK {
            bail!("{what}: year {} falls from {previous} to {v}", i + 1);
        }
        previous = v;
    }
    Ok(())
}

impl ExternalPrediction {
    /// A prediction from its yearly values, refused unless every curve is a
    /// finite, non-decreasing probability over years 1 to [`YEARS`], and the
    /// causes (when given) are exactly the cause codes.
    pub fn new(all: Vec<f64>, causes: Option<BTreeMap<String, Vec<f64>>>) -> Result<Self> {
        check_curve("cif", &all)?;
        if let Some(causes) = &causes {
            let names: Vec<&str> = causes.keys().map(String::as_str).collect();
            let mut want = CODES.to_vec();
            want.sort_unstable();
            if names != want {
                bail!("cause_cif has causes {names:?}, expected {want:?}");
            }
            for (code, curve) in causes {
                check_curve(code, curve)?;
            }
        }
        Ok(ExternalPrediction { all, causes })
    }
}

impl ExternalPrediction {
    /// The yearly curves of any model that can answer [`Outlook`], so a
    /// trained model's predictions can be kept in, and scored from, the same
    /// file format as a baseline's. A curve is clamped to a probability and
    /// made non-decreasing (a running maximum) before the usual validation;
    /// causes are kept when the model gives them.
    pub fn from_outlook<O: Outlook>(model: &O) -> Result<Self> {
        let yearly = |f: &dyn Fn(f64) -> f64| -> Vec<f64> {
            let mut top = 0.0f64;
            (1..=YEARS)
                .map(|t| {
                    top = top.max(f(t as f64).clamp(0.0, 1.0));
                    top
                })
                .collect()
        };
        let causes: Option<BTreeMap<String, Vec<f64>>> = CODES
            .iter()
            .map(|c| {
                model.cause_cif(c, 1.0)?;
                Some((
                    c.to_string(),
                    yearly(&|t| model.cause_cif(c, t).unwrap_or(0.0)),
                ))
            })
            .collect();
        ExternalPrediction::new(yearly(&|t| 1.0 - model.survival(t)), causes)
    }

    /// The equal-weight mean of `parts`' curves, for every curve; refused
    /// unless all parts agree on whether they give causes.
    pub fn average(parts: &[ExternalPrediction]) -> Result<Self> {
        let first = parts.first().ok_or_else(|| anyhow!("nothing to average"))?;
        let n = parts.len() as f64;
        let mean = |curve: &dyn Fn(&ExternalPrediction) -> &[f64]| -> Vec<f64> {
            (0..YEARS)
                .map(|i| parts.iter().map(|p| curve(p)[i]).sum::<f64>() / n)
                .collect()
        };
        let causes = match &first.causes {
            None if parts.iter().all(|p| p.causes.is_none()) => None,
            Some(first_causes) => {
                let maps: Vec<&BTreeMap<String, Vec<f64>>> =
                    parts.iter().filter_map(|p| p.causes.as_ref()).collect();
                if maps.len() != parts.len() {
                    bail!("some parts give causes and some do not");
                }
                let mut averaged = BTreeMap::new();
                for cause in first_causes.keys() {
                    let curves: Vec<&Vec<f64>> = maps
                        .iter()
                        .map(|m| {
                            m.get(cause)
                                .ok_or_else(|| anyhow!("a part has no cause {cause}"))
                        })
                        .collect::<Result<_>>()?;
                    averaged.insert(
                        cause.clone(),
                        (0..YEARS)
                            .map(|i| curves.iter().map(|c| c[i]).sum::<f64>() / n)
                            .collect(),
                    );
                }
                Some(averaged)
            }
            None => bail!("some parts give causes and some do not"),
        };
        ExternalPrediction::new(mean(&|p: &ExternalPrediction| &p.all[..]), causes)
    }

    /// The line of the prediction file for `subject_id`.
    pub fn json_line(&self, subject_id: &str) -> Result<String> {
        let round = |v: &[f64]| -> Vec<f64> { v.iter().map(|x| (x * 1e9).round() / 1e9).collect() };
        let mut line = serde_json::json!({"subject_id": subject_id, "cif": round(&self.all)});
        if let Some(causes) = &self.causes {
            line["cause_cif"] = serde_json::to_value(
                causes
                    .iter()
                    .map(|(k, v)| (k.clone(), round(v)))
                    .collect::<BTreeMap<_, _>>(),
            )?;
        }
        Ok(serde_json::to_string(&line)?)
    }
}

impl Outlook for ExternalPrediction {
    fn survival(&self, t: f64) -> f64 {
        1.0 - curve_at(&self.all, t)
    }
    fn cause_cif(&self, code: &str, t: f64) -> Option<f64> {
        self.causes
            .as_ref()?
            .get(code)
            .map(|curve| curve_at(curve, t))
    }
}

/// The predictions in `text` (JSON lines) for exactly the subjects `expected`,
/// returned in that order. A missing, extra or repeated subject, a malformed
/// line, an invalid curve, or a file that gives causes for only some
/// subjects, is an error naming the line or the subject.
pub fn parse(text: &str, expected: &[&str]) -> Result<Vec<ExternalPrediction>> {
    let mut by_id: HashMap<String, ExternalPrediction> = HashMap::new();
    for (n, line) in text.lines().filter(|l| !l.trim().is_empty()).enumerate() {
        let parsed: Line =
            serde_json::from_str(line).with_context(|| format!("prediction line {}", n + 1))?;
        let prediction = ExternalPrediction::new(parsed.cif, parsed.cause_cif)
            .with_context(|| format!("subject {}", parsed.subject_id))?;
        if by_id
            .insert(parsed.subject_id.clone(), prediction)
            .is_some()
        {
            bail!("subject {} is predicted twice", parsed.subject_id);
        }
    }
    let want: HashSet<&str> = expected.iter().copied().collect();
    if want.len() != expected.len() {
        bail!("the expected subjects repeat");
    }
    let missing: Vec<&&str> = expected
        .iter()
        .filter(|id| !by_id.contains_key(**id))
        .collect();
    let extra: Vec<&String> = by_id
        .keys()
        .filter(|id| !want.contains(id.as_str()))
        .collect();
    if !missing.is_empty() || !extra.is_empty() {
        bail!(
            "predictions cover a different subject set than the fold: {} missing (e.g. {:?}), {} extra (e.g. {:?})",
            missing.len(),
            missing.first(),
            extra.len(),
            extra.first()
        );
    }
    let with_causes = by_id.values().filter(|p| p.causes.is_some()).count();
    if with_causes != 0 && with_causes != by_id.len() {
        bail!(
            "cause_cif is given for {with_causes} of {} subjects",
            by_id.len()
        );
    }
    expected
        .iter()
        .map(|id| {
            by_id
                .remove(*id)
                .ok_or_else(|| anyhow!("{id}: missing after validation"))
        })
        .collect()
}

/// A constant prediction has no spread to recalibrate over, and its slope is
/// not a number. Unmeasured is absent, never kept as a value that cannot be
/// read back.
fn drop_unmeasured(metrics: &mut Metrics) {
    if metrics.calibration_10.as_ref().is_some_and(|c| {
        ![c.slope, c.intercept, c.oe_ratio, c.mean_abs_gap]
            .iter()
            .all(|v| v.is_finite())
    }) {
        metrics.calibration_10 = None;
    }
    if metrics.d_calibration_p.is_some_and(|p| !p.is_finite()) {
        metrics.d_calibration_p = None;
    }
}

/// The metrics of `predictions` (in the order of `subjects`) on one fold:
/// the same function, horizons and weights every arm is scored with.
pub fn score_fold(
    subjects: &[Subject],
    predictions: &[ExternalPrediction],
    horizons: &Horizons,
) -> Result<Metrics> {
    if subjects.len() != predictions.len() {
        bail!(
            "{} subjects but {} predictions",
            subjects.len(),
            predictions.len()
        );
    }
    let mut metrics = evaluate(subjects, predictions, horizons);
    drop_unmeasured(&mut metrics);
    Ok(metrics)
}

/// What is kept of one baseline's fold: the metrics, and what they came from.
#[derive(Serialize, Deserialize)]
pub struct ExternalScore {
    /// The baseline's name (its directory under `baselines/`).
    pub baseline: String,
    /// `(repeat, fold)`.
    pub fold: (usize, usize),
    /// Dataset digest.
    pub dataset: String,
    /// Partition digest.
    pub partition: String,
    /// Digest of the prediction file scored.
    pub predictions: String,
    /// Test subjects.
    pub n_test: usize,
    /// The metrics.
    pub metrics: Metrics,
}

/// The directory of one baseline's predictions and scores.
pub fn baseline_dir(data: &Path, name: &str) -> Result<PathBuf> {
    if name.is_empty() || name.contains(['/', '\\']) || name.starts_with('.') {
        bail!("{name:?} is not a baseline name");
    }
    Ok(data.join(BASELINES).join(name))
}

fn fold_stem(repeat: usize, fold: usize) -> String {
    format!("r{repeat}-k{fold}")
}

fn score_path(dir: &Path, repeat: usize, fold: usize) -> PathBuf {
    dir.join("scores")
        .join(format!("{}.json", fold_stem(repeat, fold)))
}

fn score_one(
    f: &Frozen,
    dir: &Path,
    name: &str,
    repeat: usize,
    fold: usize,
) -> Result<Option<ExternalScore>> {
    let file = dir.join(format!("{}.jsonl", fold_stem(repeat, fold)));
    let Ok(bytes) = std::fs::read(&file) else {
        return Ok(None);
    };
    let (_, test) = f.partition.fold(repeat, fold);
    let text = std::str::from_utf8(&bytes).with_context(|| format!("{}", file.display()))?;
    let predictions = parse(text, &test).with_context(|| format!("{}", file.display()))?;
    let subjects: Vec<Subject> = test.iter().map(|id| f.subjects[*id].clone()).collect();
    Ok(Some(ExternalScore {
        baseline: name.to_string(),
        fold: (repeat, fold),
        dataset: f.digests.0.clone(),
        partition: f.digests.1.clone(),
        predictions: Digest::of(&bytes).to_string(),
        n_test: test.len(),
        metrics: score_fold(&subjects, &predictions, &f.horizons)?,
    }))
}

/// Score every fold of baseline `name` that has a prediction file, writing
/// each fold's score beside it. The locked test is not read as an input: the
/// folds are the partition's cross-validation folds.
pub fn score(data: &Path, name: &str) -> Result<()> {
    let dir = baseline_dir(data, name)?;
    let f = frozen(data)?;
    std::fs::create_dir_all(dir.join("scores"))?;
    let mut scored = 0;
    for repeat in 0..f.partition.repeats.len() {
        for fold in 0..f.partition.spec.folds as usize {
            let Some(s) = score_one(&f, &dir, name, repeat, fold)? else {
                continue;
            };
            let path = score_path(&dir, repeat, fold);
            std::fs::write(&path, serde_json::to_vec_pretty(&s)?)
                .with_context(|| format!("writing {}", path.display()))?;
            println!(
                "{name} r{repeat} k{fold}: ibs_0_15 {:?} uno_c10 {:?}",
                s.metrics.ibs_0_15.as_ref().map(|m| m.value),
                s.metrics.uno_c.get(&10).map(|m| m.value)
            );
            scored += 1;
        }
    }
    if scored == 0 {
        bail!(
            "no prediction files r<repeat>-k<fold>.jsonl in {}",
            dir.display()
        );
    }
    Ok(())
}

/// The scored folds of baseline `name`, by `(repeat, fold)`.
pub fn scores(data: &Path, name: &str) -> Result<BTreeMap<(usize, usize), Metrics>> {
    let dir = baseline_dir(data, name)?.join("scores");
    let mut out = BTreeMap::new();
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Ok(out);
    };
    for e in entries.flatten() {
        if e.path().extension().is_none_or(|x| x != "json") {
            continue;
        }
        let s: ExternalScore = serde_json::from_slice(&std::fs::read(e.path())?)
            .with_context(|| format!("{}", e.path().display()))?;
        out.insert(s.fold, s.metrics);
    }
    Ok(out)
}

/// The baselines that have scores, sorted.
pub fn names(data: &Path) -> Vec<String> {
    let mut out: Vec<String> = std::fs::read_dir(data.join(BASELINES))
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().join("scores").is_dir())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    out.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const HAZARD: [f64; 2] = [0.01, 0.08];

    /// Subjects in two groups with constant death hazards (per year), followed
    /// 20 years, from a deterministic generator.
    fn cohort(n: usize) -> Vec<Subject> {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut uniform = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        (0..n)
            .map(|i| {
                let t = -uniform().ln() / HAZARD[i % 2];
                let events = if t < 20.0 {
                    format!(r#"{{"t":{},"code":"death:cvd"}}"#, 50.0 + t)
                } else {
                    String::new()
                };
                Subject::from_json_line(&format!(
                    r#"{{"subject_id":"s{i}","weight":1,"source":"c","entry":50,"calendar_at_entry":2000,"events":[{events}],"at_risk":[{{"code":"*","from":50,"to":70}}]}}"#
                ))
                .unwrap()
            })
            .collect()
    }

    fn horizons() -> Horizons {
        Horizons([("c".to_string(), 20.0)].into_iter().collect())
    }

    fn curve(rate: f64) -> Vec<f64> {
        (1..=YEARS)
            .map(|t| 1.0 - (-rate * t as f64).exp())
            .collect()
    }

    fn predictions(rate_of: impl Fn(usize) -> f64, n: usize) -> Vec<ExternalPrediction> {
        (0..n)
            .map(|i| ExternalPrediction::new(curve(rate_of(i)), None).unwrap())
            .collect()
    }

    fn lines(ids: &[String], curve: &[f64]) -> String {
        ids.iter()
            .map(|id| format!(r#"{{"subject_id":"{id}","cif":{curve:?}}}"#))
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn the_generating_incidence_scores_better_than_a_constant_one() {
        let n = 3000;
        let subjects = cohort(n);
        let truth = score_fold(&subjects, &predictions(|i| HAZARD[i % 2], n), &horizons()).unwrap();
        let mean_rate = (HAZARD[0] + HAZARD[1]) / 2.0;
        let flat = score_fold(&subjects, &predictions(|_| mean_rate, n), &horizons()).unwrap();
        let (t, f) = (truth.ibs_0_15.unwrap().value, flat.ibs_0_15.unwrap().value);
        assert!(t < f, "true curves {t} against a constant {f}");
        assert!(truth.uno_c[&10].value > 0.65 && flat.uno_c[&10].value < 0.55);
        assert!(truth.brier[&10].value < flat.brier[&10].value);
    }

    #[test]
    fn a_metric_that_cannot_be_measured_is_absent_and_the_score_still_reads_back() {
        let mut m = Metrics {
            calibration_10: Some(crate::metrics::Calibration {
                slope: f64::NAN,
                intercept: 0.1,
                oe_ratio: 1.0,
                mean_abs_gap: 0.0,
                n: 5,
            }),
            d_calibration_p: Some(f64::NAN),
            ..Metrics::default()
        };
        // A value that is not a number is written as null, which cannot be
        // read back as a number.
        assert!(serde_json::from_slice::<Metrics>(&serde_json::to_vec(&m).unwrap()).is_err());
        drop_unmeasured(&mut m);
        let back: Metrics = serde_json::from_slice(&serde_json::to_vec(&m).unwrap()).unwrap();
        assert!(back.calibration_10.is_none() && back.d_calibration_p.is_none());
    }

    #[test]
    fn a_mismatched_subject_set_is_refused() {
        let ids: Vec<String> = (0..4).map(|i| format!("s{i}")).collect();
        let expected: Vec<&str> = ids.iter().map(String::as_str).collect();
        let c = curve(0.02);
        assert_eq!(parse(&lines(&ids, &c), &expected).unwrap().len(), 4);
        let missing = parse(&lines(&ids[..3], &c), &expected).unwrap_err();
        assert!(
            missing.to_string().contains("different subject set"),
            "{missing}"
        );
        let mut more = ids.clone();
        more.push("stranger".into());
        assert!(parse(&lines(&more, &c), &expected).is_err());
        let mut twice = ids.clone();
        twice.push("s0".into());
        let err = parse(&lines(&twice, &c), &expected).unwrap_err();
        assert!(err.to_string().contains("twice"), "{err}");
    }

    #[test]
    fn a_prediction_that_is_not_a_probability_curve_is_refused() {
        let ids = vec!["s0".to_string()];
        let expected = ["s0"];
        let nan = format!(
            r#"{{"subject_id":"s0","cif":[NaN{}]}}"#,
            ",0.1".repeat(YEARS - 1)
        );
        assert!(parse(&nan, &expected).is_err(), "NaN text");
        let mut c = curve(0.02);
        c[3] = f64::NAN;
        assert!(ExternalPrediction::new(c, None).is_err(), "NaN value");
        let mut c = curve(0.02);
        c[14] = 1.5;
        assert!(ExternalPrediction::new(c, None).is_err(), "above one");
        let mut c = curve(0.02);
        c[7] = 0.0;
        assert!(ExternalPrediction::new(c, None).is_err(), "falling");
        assert!(
            ExternalPrediction::new(curve(0.02)[..10].to_vec(), None).is_err(),
            "short"
        );
        assert!(parse(&lines(&ids, &curve(0.02)), &expected).is_ok());
    }

    #[test]
    fn an_average_of_predictions_is_their_mean_curve() {
        let (a, b) = (curve(0.01), curve(0.03));
        let mean = ExternalPrediction::average(&[
            ExternalPrediction::new(a.clone(), None).unwrap(),
            ExternalPrediction::new(b.clone(), None).unwrap(),
        ])
        .unwrap();
        for t in 1..=YEARS {
            let want = 1.0 - 0.5 * ((1.0 - a[t - 1]) + (1.0 - b[t - 1]));
            assert!((1.0 - mean.survival(t as f64) - want).abs() < 1e-12);
        }
        let causes = ExternalPrediction::new(
            curve(0.02),
            Some(
                CODES
                    .iter()
                    .map(|c| (c.to_string(), curve(0.005)))
                    .collect(),
            ),
        )
        .unwrap();
        assert!(
            ExternalPrediction::average(&[mean.clone(), causes]).is_err(),
            "causes for some parts only"
        );
        assert!(ExternalPrediction::average(&[]).is_err());
        let line = mean.json_line("s0").unwrap();
        let back = parse(&line, &["s0"]).unwrap();
        assert!((back[0].survival(10.0) - mean.survival(10.0)).abs() < 1e-8);
    }

    #[test]
    fn a_curve_is_linear_between_years_and_held_after_the_last() {
        let p = ExternalPrediction::new(curve(0.02), None).unwrap();
        assert_eq!(p.survival(0.0), 1.0);
        assert!((p.survival(1.0) - (-0.02f64).exp()).abs() < 1e-12);
        let mid = 0.5 * (p.survival(2.0) + p.survival(3.0));
        assert!((p.survival(2.5) - mid).abs() < 1e-12);
        assert_eq!(p.survival(40.0), p.survival(15.0));
        assert_eq!(p.cause_cif("death:cvd", 5.0), None);
    }
}
