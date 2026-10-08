// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the full loop from declared cohort data to a
// gated, traceable release of a risk model, for its clients. If your team
// needs expertise in running that loop on real records, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The pipeline's steps as the sample drives them: `import`, `split`, `train`,
//! `eval` and `release`. Each is a call into Splinter's timeline pipeline
//! ([`splinter_sdk::timeline`]); this file decides nothing about training or
//! about whether a candidate is better. It chooses the experiment: which
//! outcomes (checked against the ontology), which horizons, which bands.

use std::path::Path;

use anyhow::{Context as _, Result};
use serde::{Deserialize, Serialize};
use splinter_sdk::data::timeline_dataset::ProjectionSpec;
use splinter_sdk::measure::timeline_metrics::{CalibrationBands, CodePlan, TimelinePlan};
use splinter_sdk::model::timeline::scoring::{ScoreSpec, Subgroup, SubgroupRule};
use splinter_sdk::model::timeline::{Backbone, NextEvents, TimelineTraining};
use splinter_sdk::timeline::data::{
    import_records, split_timeline, ImportRequest, SplitPlan, SplitRequest,
};
use splinter_sdk::timeline::evaluate::{evaluate_timeline, Champion, TimelineEvaluateRequest};
use splinter_sdk::timeline::release::{release_timeline, TimelineReleaseRequest};
use splinter_sdk::timeline::train::{train_timeline_candidate, TimelineTrainRequest};
use splinter_sdk::vocabulary::terms::{Distribution, Terms};
use splinter_sdk::Context;

use crate::ontology::{Ontology, ALL_CAUSE};
use crate::source::SubgroupDeclaration;
use crate::state::{Run, SplitRecord};

/// What an import recorded about its dataset, for the later steps.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ImportRecord {
    /// The dataset's id.
    pub dataset: String,
    /// The outcome codes the dataset supplies.
    pub supplies: Vec<String>,
    /// Subgroups the evaluation is also reported on.
    pub subgroups: Vec<SubgroupDeclaration>,
    /// The usage label of the terms it came under.
    pub usage: String,
}

fn import_path(run: &Run, dataset: &str) -> std::path::PathBuf {
    run.dir.join("imports").join(format!("{dataset}.json"))
}

/// Imports the record file `file` as `dataset` under `terms`.
pub fn import(
    run: &Run,
    file: &Path,
    record: &ImportRecord,
    terms: Terms,
    secret: &str,
) -> Result<serde_json::Value> {
    let ctx = run.context()?;
    let report = import_records(
        &ctx,
        &ImportRequest {
            file: file.to_path_buf(),
            dataset: record.dataset.clone(),
            terms,
            secret: secret.as_bytes().to_vec(),
        },
    )?;
    let path = import_path(run, &record.dataset);
    std::fs::create_dir_all(path.parent().context("an import record has a directory")?)?;
    std::fs::write(&path, serde_json::to_vec_pretty(record)?)?;
    Ok(serde_json::json!({
        "dataset": report.dataset,
        "file": report.file.to_string(),
        "rows": report.rows,
        "participants": report.participants,
        "added": report.added,
        "already_present": report.already_present,
    }))
}

/// The import record of `dataset`.
pub fn imported(run: &Run, dataset: &str) -> Result<ImportRecord> {
    let path = import_path(run, dataset);
    let text = std::fs::read_to_string(&path).with_context(|| {
        format!(
            "reading {}: run `import` for {dataset} first",
            path.display()
        )
    })?;
    Ok(serde_json::from_str(&text)?)
}

/// How `split` cuts, as the command line states it.
#[derive(Clone, Debug)]
pub struct SplitArgs {
    /// The imported dataset; the only one imported when `None`.
    pub dataset: Option<String>,
    /// How to cut.
    pub plan: SplitPlan,
    /// The seed.
    pub seed: u64,
}

/// Cuts the dataset into stored parts and remembers the split.
pub fn split(run: &Run, args: &SplitArgs) -> Result<SplitRecord> {
    let dataset = match &args.dataset {
        Some(d) => d.clone(),
        None => only_import(run)?,
    };
    let ctx = run.context()?;
    let report = split_timeline(
        &ctx,
        &SplitRequest {
            dataset: Some(dataset.clone()),
            plan: args.plan.clone(),
            seed: args.seed,
            projection: ProjectionSpec::at_entry(),
        },
    )?;
    let record = SplitRecord {
        dataset,
        report: (&report).into(),
    };
    run.save_split(&record)?;
    Ok(record)
}

fn only_import(run: &Run) -> Result<String> {
    let dir = run.dir.join("imports");
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .with_context(|| format!("reading {}: run `import` first", dir.display()))?
        .filter_map(|e| {
            e.ok()?
                .path()
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
        })
        .collect();
    names.sort();
    match names.as_slice() {
        [one] => Ok(one.clone()),
        [] => anyhow::bail!(
            "nothing is imported in {}: run `import` first",
            run.dir.display()
        ),
        many => anyhow::bail!(
            "several datasets are imported ({}); name one with --dataset",
            many.join(", ")
        ),
    }
}

/// How `train` trains, as the command line states it.
#[derive(Clone, Debug)]
pub struct TrainArgs {
    /// The label the candidate is named by.
    pub label: String,
    /// Optimiser steps at most.
    pub steps: u32,
    /// The seed.
    pub seed: u64,
    /// The additive proportional-hazards baseline instead of the set encoder.
    pub additive: bool,
    /// Tokens per subject.
    pub max_tokens: Option<u32>,
    /// Subjects per batch.
    pub batch: u32,
    /// Read the history visit by visit, carrying a state over this many
    /// visits.
    pub visits: Option<u32>,
    /// What carries the visits to the prediction time (with `visits`).
    pub backbone: Option<Backbone>,
    /// A group of events modelled for which comes first, with its weight.
    pub next_events: Option<NextEvents>,
    /// A forecast head on this many future measurements per subject, with its
    /// weight.
    pub forecasts: Option<(u32, f32)>,
    /// The horizons the trained model's risks are calibrated at, on the
    /// validation units early stopping does not read; none leaves it raw.
    pub calibrate_at: Vec<f64>,
}

/// The outcome codes a dataset is trained on: what it supplies, each of them
/// requested from the ontology for this dataset.
pub fn outcome_codes(ontology: &Ontology, record: &ImportRecord) -> Result<Vec<String>> {
    ontology.request_all(&record.supplies, &record.dataset)?;
    Ok(record.supplies.clone())
}

/// The commit of Splinter the sample was built in, with `+dirty` when the
/// tree has uncommitted changes; `None` when it is not a git checkout.
pub fn splinter_commit() -> Option<String> {
    let git = |args: &[&str]| {
        let out = std::process::Command::new("git")
            .arg("-C")
            .arg(env!("CARGO_MANIFEST_DIR"))
            .args(args)
            .output()
            .ok()?;
        out.status
            .success()
            .then(|| String::from_utf8_lossy(&out.stdout).trim().to_owned())
    };
    let head = git(&["rev-parse", "HEAD"])?;
    let dirty =
        git(&["status", "--porcelain", "--untracked-files=no"]).is_some_and(|s| !s.is_empty());
    Some(if dirty { format!("{head}+dirty") } else { head })
}

/// Trains one candidate on the stored training and validation parts.
pub fn train(run: &Run, args: &TrainArgs) -> Result<serde_json::Value> {
    let split = run.split()?;
    let record = imported(run, &split.dataset)?;
    let codes = outcome_codes(&Ontology::bundled()?, &record)?;
    let mut config = TimelineTraining::new(codes.clone(), codes);
    config.steps = args.steps;
    config.seed = args.seed;
    config.additive = args.additive;
    config.max_tokens = args.max_tokens;
    config.batch = args.batch;
    config.visits = args.visits;
    config.backbone = args.backbone;
    config.next_events = args.next_events.clone();
    config.forecasts = args.forecasts;
    let ctx = run.context()?;
    let mut request =
        TimelineTrainRequest::new(&split.report.train, &split.report.validation, config);
    if !args.calibrate_at.is_empty() {
        request = request.calibrated_at(args.calibrate_at.clone());
    }
    request.splinter_commit = splinter_commit();
    let trained = train_timeline_candidate(&ctx, &request)?;
    run.label(&args.label, &trained.id)?;
    let c = &trained.candidate;
    Ok(serde_json::json!({
        "candidate": trained.id,
        "label": args.label,
        "steps": c.outcome.steps,
        "held_out_event_nll": c.outcome.held_out_event_nll,
        "parameters": c.outcome.parameters,
        "knots": c.config.knots,
        "config_digest": c.config_digest.to_string(),
        "seed": c.seed,
        "split": c.split.to_string(),
        "fitted_on_units": c.fit.units,
        "checkpoint_sha256": c.checkpoint_sha256.to_string(),
        "round_trip_max_abs_diff": c.round_trip_max_abs_diff,
        "calibration": c.calibration.as_ref().map(|k| serde_json::json!({
            "digest": k.digest.to_string(),
            "units": k.outcome.units,
            "early_stopping_units": k.early_stopping_units,
            "calibrated": k.outcome.calibrated,
            "uncalibrated": k.outcome.uncalibrated,
        })),
        "brain_commit": c.brain_commit,
        "splinter_commit": c.splinter_commit,
    }))
}

/// How `eval` scores, as the command line states it.
#[derive(Clone, Debug)]
pub struct EvalArgs {
    /// The label the evaluation is named by.
    pub label: String,
    /// The candidate, by label or id.
    pub candidate: String,
    /// The champion: a candidate by label or id, or `release:<alias or id>`.
    pub champion: String,
    /// The horizons, in years.
    pub horizons: Vec<f64>,
    /// Judge calibration on the all-cause view alone.
    pub all_cause_only: bool,
    /// Bootstrap resamples.
    pub reps: usize,
}

/// The bands the plan holds a candidate's calibration to, registered in the
/// README beside the reason for each.
pub fn bands() -> CalibrationBands {
    CalibrationBands {
        slope: (0.7, 1.4),
        intercept_in_the_large: (-0.5, 0.5),
        oe: (0.8, 1.25),
        max_ece: 0.05,
    }
}

/// Scores the candidate against the champion on the test part.
pub fn eval(run: &Run, args: &EvalArgs) -> Result<serde_json::Value> {
    let split = run.split()?;
    let record = imported(run, &split.dataset)?;
    let codes = outcome_codes(&Ontology::bundled()?, &record)?;
    let champion = match args.champion.strip_prefix("release:") {
        Some(release) => Champion::Release(release.to_owned()),
        None => Champion::Candidate(run.resolve(&args.champion)?),
    };
    let mut scoring = ScoreSpec::new(codes.clone(), codes.clone(), args.horizons.clone());
    scoring.all_cause = Some(ALL_CAUSE.into());
    scoring.bootstrap_reps = args.reps;
    scoring.subgroups = record
        .subgroups
        .iter()
        .map(|s| Subgroup {
            name: s.name.clone(),
            rule: SubgroupRule::Category {
                var: s.var.clone(),
                level: s.level.clone(),
            },
        })
        .collect();
    let judged = |code: &str, must_improve| CodePlan {
        code: code.into(),
        horizons: args.horizons.clone(),
        must_improve,
    };
    let mut plan_codes: Vec<CodePlan> = if args.all_cause_only {
        Vec::new()
    } else {
        codes.iter().map(|c| judged(c, false)).collect()
    };
    plan_codes.push(judged(ALL_CAUSE, true));
    let plan = TimelinePlan {
        codes: plan_codes,
        improve_nll: true,
        calibration: bands(),
        subgroups: record.subgroups.iter().map(|s| s.name.clone()).collect(),
        subgroup_margin: 0.002,
        min_subgroup_events: 10,
        serve_tolerance: 1e-6,
        batch_tolerance: 1e-6,
        max_abstention_rate: 0.1,
    };
    let ctx = run.context()?;
    let evaluated = evaluate_timeline(
        &ctx,
        &TimelineEvaluateRequest {
            candidate: run.resolve(&args.candidate)?,
            champion,
            test: split.report.test.clone(),
            scoring,
            plan,
        },
    )?;
    run.label(&args.label, &evaluated.id.to_string())?;
    let e = &evaluated.evaluation;
    let intervals: serde_json::Map<String, serde_json::Value> = e
        .evidence
        .intervals
        .iter()
        .map(|(k, (lo, hi))| (k.clone(), serde_json::json!([lo, hi])))
        .collect();
    Ok(serde_json::json!({
        "evaluation": evaluated.id.to_string(),
        "label": args.label,
        "candidate": e.candidate,
        "champion": e.champion,
        "test_units": e.comparison.units.len(),
        "candidate_minus_champion": intervals,
        "values": e.evidence.values.iter().filter(|(k, _)| k.starts_with("candidate:") || k.starts_with("subgroup") || k.starts_with("serve")).collect::<std::collections::BTreeMap<_, _>>(),
    }))
}

/// What `release` decided.
#[derive(Debug)]
pub struct Decision {
    /// What to print.
    pub report: serde_json::Value,
    /// Whether the candidate was released.
    pub released: bool,
}

/// Runs the gate on the evaluation and releases the candidate if it passes.
pub fn release(run: &Run, evaluation: &str, alias: &str, unrestricted: bool) -> Result<Decision> {
    let ctx: Context = run.context()?;
    let released = release_timeline(
        &ctx,
        &TimelineReleaseRequest {
            evaluation: run.resolve(evaluation)?,
            alias: alias.to_owned(),
            distribution: if unrestricted {
                Distribution::Unrestricted
            } else {
                Distribution::Restricted
            },
        },
    )?;
    let mut failed = Vec::new();
    let report = &released.report;
    for (check, section) in [
        ("performance", &report.performance),
        ("calibration", &report.calibration),
        ("retention", &report.retention),
        ("serving", &report.serving),
    ] {
        for d in section.decided.iter().filter(|d| !d.check.passed) {
            failed.push(format!(
                "{check}: {}",
                d.check
                    .reason
                    .clone()
                    .unwrap_or_else(|| "not measured".into())
            ));
        }
    }
    if !report.policy.passed {
        failed.push(format!(
            "data policy: {}",
            report.policy.reason.clone().unwrap_or_default()
        ));
    }
    Ok(Decision {
        released: released.release.is_some(),
        report: serde_json::json!({
            "candidate": released.candidate,
            "alias": released.alias,
            "decision": if released.release.is_some() { "released" } else { "rejected" },
            "champion": released.champion.map(|c| c.to_string()),
            "release": released.release.map(|r| r.to_string()),
            "artifact": released.artifact.map(|p| p.display().to_string()),
            "distribution": if unrestricted { "unrestricted" } else { "restricted" },
            "units": report.units,
            "failed_checks": failed,
        }),
    })
}
