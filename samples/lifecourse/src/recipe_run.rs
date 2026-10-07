// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements long-horizon risk prediction from cohort and
// survey data for its clients. If your team needs expertise in building,
// validating and proving time-to-event models on real health records, you
// can procure our services by sending an email to info@swedishembedded.com.

//! Running recipes on cross-validation folds, averaging their predictions,
//! and tabulating the results.
//!
//! Everything lives under `<data>/recipe/<name>/`, never under `runs/`:
//! `r<repeat>-k<fold>.jsonl` (the out-of-fold predictions, the format of the
//! `external` arm) and `r<repeat>-k<fold>.json` (the scores). Every model,
//! trained or averaged, is scored from its prediction file's curves by the
//! function every arm is scored with, so a single run, an ensemble of runs
//! and a baseline are on one footing, and `compare` reads them as
//! `recipe:<name>`. The locked test is not an input: only partition folds.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use splinter_sdk::model::timeline::Subject;
use splinter_sdk::vocabulary::digest::Digest;

use crate::commands::{frozen, nhanes_terms, require_training, Frozen};
use crate::compare::Comparand;
use crate::external::{baseline_dir, parse, score_fold, ExternalPrediction};
use crate::metrics::Metrics;
use crate::recipe::{fit, subsample, Recipe};

/// Directory under the data directory that holds one directory per recipe.
pub const RECIPES: &str = "recipe";

/// What is kept of one fold of one recipe.
#[derive(Serialize, Deserialize)]
pub struct RecipeScore {
    /// The recipe's name (its directory).
    pub name: String,
    /// The overrides it was trained with, or the members it averages.
    pub made_of: Vec<String>,
    /// Training seed (`None` for an average of members).
    pub seed: Option<u64>,
    /// `(repeat, fold)`.
    pub fold: (usize, usize),
    /// Dataset digest.
    pub dataset: String,
    /// Partition digest.
    pub partition: String,
    /// Digest of the prediction file scored.
    pub predictions: String,
    /// Training subjects used, trained on and held out for early stopping
    /// (zero for an average).
    pub subjects: (usize, usize),
    /// Optimiser steps run (zero for an average).
    pub steps: u32,
    /// Event NLL on the early-stopping subjects (`None` for an average).
    pub early_stopping_nll: Option<f32>,
    /// Trainable parameters (zero for an average).
    pub parameters: usize,
    /// Test subjects.
    pub n_test: usize,
    /// Wall-clock seconds of training and scoring (zero for an average).
    pub seconds: f64,
    /// The metrics.
    pub metrics: Metrics,
}

/// The directory of recipe `name`; the name is one path component.
pub fn recipe_dir(data: &Path, name: &str) -> Result<PathBuf> {
    if name.is_empty() || name.contains(['/', '\\']) || name.starts_with('.') {
        bail!("{name:?} is not a recipe name");
    }
    Ok(data.join(RECIPES).join(name))
}

fn stem(repeat: usize, fold: usize) -> String {
    format!("r{repeat}-k{fold}")
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, path).with_context(|| format!("replacing {}", path.display()))
}

/// Write predictions (in the order of `test`) and their score for one fold.
#[allow(clippy::too_many_arguments)]
fn keep(
    f: &Frozen,
    dir: &Path,
    mut score: RecipeScore,
    test: &[&str],
    predictions: &[ExternalPrediction],
    write_predictions: bool,
) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let (repeat, fold) = score.fold;
    let mut text = String::new();
    for (id, p) in test.iter().zip(predictions) {
        text.push_str(&p.json_line(id)?);
        text.push('\n');
    }
    // Scored from the curves as read back: what a reader of the file gets.
    let reread = parse(&text, test)?;
    let subjects: Vec<Subject> = test.iter().map(|id| f.subjects[*id].clone()).collect();
    score.predictions = Digest::of(text.as_bytes()).to_string();
    score.metrics = score_fold(&subjects, &reread, &f.horizons)?;
    if write_predictions {
        write_atomic(
            &dir.join(format!("{}.jsonl", stem(repeat, fold))),
            text.as_bytes(),
        )?;
    }
    write_atomic(
        &dir.join(format!("{}.json", stem(repeat, fold))),
        &serde_json::to_vec_pretty(&score)?,
    )?;
    let m = &score.metrics;
    println!(
        "{} r{repeat} k{fold}: ibs_0_15 {:?} uno_c10 {:?} steps {} in {:.0}s",
        score.name,
        m.ibs_0_15.as_ref().map(|x| x.value),
        m.uno_c.get(&10).map(|x| x.value),
        score.steps,
        score.seconds
    );
    Ok(())
}

fn folds_asked(f: &Frozen, repeats: &[usize], folds: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let all = |given: &[usize], n: usize| {
        if given.is_empty() {
            (0..n).collect()
        } else {
            given.to_vec()
        }
    };
    (
        all(repeats, f.partition.repeats.len()),
        all(folds, f.partition.spec.folds as usize),
    )
}

/// Train `recipe` with each of `seeds` on the folds asked for (all by
/// default), keeping member `<name>-s<seed>`. A fold already kept is skipped.
pub fn run(
    data: &Path,
    name: &str,
    recipe: &Recipe,
    seeds: &[u64],
    repeats: &[usize],
    folds: &[usize],
    write_predictions: bool,
) -> Result<()> {
    require_training(&nhanes_terms())?;
    let f = frozen(data)?;
    let (rs, ks) = folds_asked(&f, repeats, folds);
    let made_of: Vec<String> = recipe
        .overrides()
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect();
    for &repeat in &rs {
        for &fold in &ks {
            for &seed in seeds {
                let member = format!("{name}-s{seed}");
                let dir = recipe_dir(data, &member)?;
                if dir.join(format!("{}.json", stem(repeat, fold))).exists() {
                    println!("{member} r{repeat} k{fold}: done already");
                    continue;
                }
                let t0 = Instant::now();
                let (train, test) = f.partition.fold(repeat, fold);
                let train = match recipe.train_share()? {
                    Some((share, train_seed)) => subsample(&train, share, train_seed),
                    None => train,
                };
                let train: Vec<&Subject> = train.iter().map(|id| &f.subjects[*id]).collect();
                let (model, info) = fit(recipe, &train, seed)?;
                let arm = recipe.arm()?;
                let views: Vec<Subject> =
                    test.iter().map(|id| arm.view(&f.subjects[*id])).collect();
                let predictions = model
                    .predict(&views)?
                    .iter()
                    .map(ExternalPrediction::from_outlook)
                    .collect::<Result<Vec<_>>>()?;
                let score = RecipeScore {
                    name: member,
                    made_of: made_of.clone(),
                    seed: Some(seed),
                    fold: (repeat, fold),
                    dataset: f.digests.0.clone(),
                    partition: f.digests.1.clone(),
                    predictions: String::new(),
                    subjects: info.subjects,
                    steps: info.steps,
                    early_stopping_nll: Some(info.early_stopping_nll),
                    parameters: info.parameters,
                    n_test: test.len(),
                    seconds: t0.elapsed().as_secs_f64(),
                    metrics: Metrics::default(),
                };
                keep(&f, &dir, score, &test, &predictions, write_predictions)?;
            }
        }
    }
    Ok(())
}

/// Where the prediction files of `member` are: a recipe, or `external:<baseline>`.
fn member_dir(data: &Path, member: &str) -> Result<PathBuf> {
    match member.strip_prefix("external:") {
        Some(baseline) => baseline_dir(data, baseline),
        None => recipe_dir(data, member),
    }
}

/// Keep `name`: the mean of `members`' predicted curves on every fold all of
/// them have predictions for (among those asked for), scored like any other.
pub fn blend(
    data: &Path,
    name: &str,
    members: &[String],
    repeats: &[usize],
    folds: &[usize],
) -> Result<()> {
    if members.len() < 2 {
        bail!(
            "an average needs at least two members, not {}",
            members.len()
        );
    }
    let f = frozen(data)?;
    let dir = recipe_dir(data, name)?;
    let (rs, ks) = folds_asked(&f, repeats, folds);
    let dirs: Vec<PathBuf> = members
        .iter()
        .map(|m| member_dir(data, m))
        .collect::<Result<_>>()?;
    let mut blended = 0;
    for &repeat in &rs {
        for &fold in &ks {
            let files: Vec<PathBuf> = dirs
                .iter()
                .map(|d| d.join(format!("{}.jsonl", stem(repeat, fold))))
                .collect();
            if files.iter().any(|p| !p.is_file()) {
                continue;
            }
            let (_, test) = f.partition.fold(repeat, fold);
            let parts: Vec<Vec<ExternalPrediction>> = files
                .iter()
                .map(|p| {
                    let text = std::fs::read_to_string(p)?;
                    parse(&text, &test).with_context(|| format!("{}", p.display()))
                })
                .collect::<Result<_>>()?;
            let predictions = (0..test.len())
                .map(|i| {
                    let each: Vec<ExternalPrediction> =
                        parts.iter().map(|m| m[i].clone()).collect();
                    ExternalPrediction::average(&each)
                })
                .collect::<Result<Vec<_>>>()?;
            let score = RecipeScore {
                name: name.to_string(),
                made_of: members.to_vec(),
                seed: None,
                fold: (repeat, fold),
                dataset: f.digests.0.clone(),
                partition: f.digests.1.clone(),
                predictions: String::new(),
                subjects: (0, 0),
                steps: 0,
                early_stopping_nll: None,
                parameters: 0,
                n_test: test.len(),
                seconds: 0.0,
                metrics: Metrics::default(),
            };
            keep(&f, &dir, score, &test, &predictions, true)?;
            blended += 1;
        }
    }
    if blended == 0 {
        bail!("no fold has predictions from every member");
    }
    Ok(())
}

/// Score the prediction files already in recipe `name`'s directory (written
/// by a program outside Rust, for instance a baseline trained on a subsample)
/// like any other; `made_of` records where they came from.
pub fn score_files(
    data: &Path,
    name: &str,
    made_of: &str,
    repeats: &[usize],
    folds: &[usize],
) -> Result<()> {
    let f = frozen(data)?;
    let dir = recipe_dir(data, name)?;
    let (rs, ks) = folds_asked(&f, repeats, folds);
    let mut scored = 0;
    for &repeat in &rs {
        for &fold in &ks {
            let file = dir.join(format!("{}.jsonl", stem(repeat, fold)));
            let Ok(text) = std::fs::read_to_string(&file) else {
                continue;
            };
            let (_, test) = f.partition.fold(repeat, fold);
            let predictions = parse(&text, &test).with_context(|| format!("{}", file.display()))?;
            let score = RecipeScore {
                name: name.to_string(),
                made_of: vec![made_of.to_string()],
                seed: None,
                fold: (repeat, fold),
                dataset: f.digests.0.clone(),
                partition: f.digests.1.clone(),
                predictions: String::new(),
                subjects: (0, 0),
                steps: 0,
                early_stopping_nll: None,
                parameters: 0,
                n_test: test.len(),
                seconds: 0.0,
                metrics: Metrics::default(),
            };
            keep(&f, &dir, score, &test, &predictions, false)?;
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

/// Write, for every fold asked for, the training subjects a recipe with
/// `train_share=share`, `train_seed=seed` uses, as
/// `<data>/recipe/_subsamples/p<percent>-s<seed>/r<repeat>-k<fold>.ids`
/// (one subject id per line), so a baseline outside Rust trains on the same ones.
pub fn write_subsamples(
    data: &Path,
    share: f64,
    seed: u64,
    repeats: &[usize],
    folds: &[usize],
) -> Result<()> {
    let f = frozen(data)?;
    let (rs, ks) = folds_asked(&f, repeats, folds);
    let dir = data
        .join(RECIPES)
        .join("_subsamples")
        .join(format!("p{}-s{seed}", (share * 100.0).round() as u32));
    std::fs::create_dir_all(&dir)?;
    for &repeat in &rs {
        for &fold in &ks {
            let (train, _) = f.partition.fold(repeat, fold);
            let kept = subsample(&train, share, seed);
            let path = dir.join(format!("{}.ids", stem(repeat, fold)));
            write_atomic(&path, (kept.join("\n") + "\n").as_bytes())?;
            println!(
                "{}: {} of {} subjects",
                path.display(),
                kept.len(),
                train.len()
            );
        }
    }
    Ok(())
}

/// The scored folds of recipe `name`, by `(repeat, fold)`.
pub fn scores(data: &Path, name: &str) -> Result<BTreeMap<(usize, usize), Metrics>> {
    let dir = recipe_dir(data, name)?;
    let mut out = BTreeMap::new();
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Ok(out);
    };
    for e in entries.flatten() {
        if e.path().extension().is_none_or(|x| x != "json") {
            continue;
        }
        let s: RecipeScore = serde_json::from_slice(&std::fs::read(e.path())?)
            .with_context(|| format!("{}", e.path().display()))?;
        out.insert(s.fold, s.metrics);
    }
    Ok(out)
}

/// One line per model: its mean over the folds it shares with every other
/// model listed (and the repeats asked for), so means are comparable.
pub fn summary(data: &Path, models: &[Comparand], repeats: &[usize]) -> Result<()> {
    let per: Vec<BTreeMap<(usize, usize), Metrics>> = models
        .iter()
        .map(|m| m.folds(data))
        .collect::<Result<_>>()?;
    let shared: Vec<(usize, usize)> = per[0]
        .keys()
        .copied()
        .filter(|k| {
            (repeats.is_empty() || repeats.contains(&k.0)) && per.iter().all(|p| p.contains_key(k))
        })
        .collect();
    if shared.is_empty() {
        bail!("no fold ran for every model listed");
    }
    println!("{} shared folds", shared.len());
    println!("| model | ibs_0_15 | brier_10 | uno_c_5 | uno_c_10 | cal_slope_10 |");
    println!("|---|---|---|---|---|---|");
    for (m, folds) in models.iter().zip(&per) {
        let mean = |get: &dyn Fn(&Metrics) -> Option<f64>| -> String {
            let v: Vec<f64> = shared.iter().filter_map(|k| get(&folds[k])).collect();
            if v.is_empty() {
                "-".into()
            } else {
                format!("{:.5}", v.iter().sum::<f64>() / v.len() as f64)
            }
        };
        println!(
            "| {} | {} | {} | {} | {} | {} |",
            m.name(),
            mean(&|x| x.ibs_0_15.as_ref().map(|v| v.value)),
            mean(&|x| x.brier.get(&10).map(|v| v.value)),
            mean(&|x| x.uno_c.get(&5).map(|v| v.value)),
            mean(&|x| x.uno_c.get(&10).map(|v| v.value)),
            mean(&|x| x.calibration_10.as_ref().map(|c| c.slope)),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_recipe_name_is_one_path_component() {
        let data = Path::new("/d");
        assert!(recipe_dir(data, "d32-s1").is_ok());
        for bad in ["", "../x", "a/b", ".hidden", "a\\b"] {
            assert!(recipe_dir(data, bad).is_err(), "{bad:?}");
        }
    }
}
