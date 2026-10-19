// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Fixtures the timeline specs share: a deterministic synthetic cohort of
//! known hazards written as a record file (households of two, so groups
//! matter), a context over a scratch state root, and the import and split
//! stages run on it.

use std::path::{Path, PathBuf};

use splinter_core::terms::{Terms, UsagePolicy};
use splinter_data::timeline_dataset::ProjectionSpec;
use splinter_eval::timeline_metrics::{
    CalibrationBands, CodePlan, RequiredHorizon, TimelinePlan, DEFAULT_MIN_TEST_EVENTS,
};
use splinter_model::timeline::scoring::{ScoreSpec, Subgroup, SubgroupRule};
use splinter_model::timeline::synthetic;
use splinter_model::timeline::{CalibrationPlan, TimelineTraining};
use splinter_orchestrator::Context;
use splinter_pipelines::timeline::data::{
    import_records, split_timeline, ImportRequest, SplitPlan, SplitReport, SplitRequest,
};
use splinter_pipelines::timeline::evaluate::{
    evaluate_timeline, Champion, TimelineEvaluateRequest, TimelineEvaluated,
};
use splinter_pipelines::timeline::train::{
    train_timeline_candidate, TimelineTrainRequest, TimelineTrained,
};

use super::{config, Scratch};

/// The outcome codes of the synthetic population.
pub const CODES: [&str; 3] = ["death:a", "death:b", "onset"];
/// The absorbing ones.
pub const ABSORBING: [&str; 2] = ["death:a", "death:b"];

/// A context over a scratch state root, with no model.
pub fn context(test: &str) -> (Scratch, Context) {
    let scratch = Scratch::new(test);
    let ctx = Context::new(config(&scratch), false).unwrap();
    (scratch, ctx)
}

/// `n` synthetic subjects as a record file named `name` in `dir`: the
/// generator's own history and outcomes, a household of two per group, and
/// no interventions. Deterministic in `seed`.
pub fn write_synthetic(dir: &Path, name: &str, n: usize, seed: u64) -> PathBuf {
    write_synthetic_without(dir, name, n, seed, &[])
}

/// [`write_synthetic`] with the variables `without` never measured: the
/// conventional risk factors of a cohort that did not take the new
/// measurements.
pub fn write_synthetic_without(
    dir: &Path,
    name: &str,
    n: usize,
    seed: u64,
    without: &[&str],
) -> PathBuf {
    let (mut subjects, _) = synthetic::population(n, seed);
    for s in &mut subjects {
        s.observations
            .retain(|o| !without.contains(&o.var.as_str()));
    }
    let lines: Vec<String> = subjects
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let mut line = serde_json::to_value(s).unwrap();
            line["group_id"] = format!("household-{}", i / 2).into();
            // Households alternate between two survey cycles: a household is
            // never in two sources.
            line["source"] = ["cycle-a", "cycle-b"][(i / 2) % 2].into();
            line["interventions"] = serde_json::json!([]);
            line.to_string()
        })
        .collect();
    let path = dir.join(name);
    std::fs::write(&path, lines.join("\n") + "\n").unwrap();
    path
}

/// Imports `file` as `dataset` under `policy` and splits it by participant
/// group with a 30 percent test.
pub fn prepare(
    ctx: &Context,
    file: &Path,
    dataset: &str,
    policy: UsagePolicy,
    seed: u64,
) -> SplitReport {
    prepare_with(ctx, file, dataset, policy, seed, 5)
}

/// [`prepare`] with `folds` folds of the rest: more folds, a larger training
/// part and a smaller validation part, the same test.
pub fn prepare_with(
    ctx: &Context,
    file: &Path,
    dataset: &str,
    policy: UsagePolicy,
    seed: u64,
    folds: u32,
) -> SplitReport {
    import_records(
        ctx,
        &ImportRequest {
            file: file.to_path_buf(),
            dataset: dataset.into(),
            terms: policy.terms(dataset),
            secret: b"test campaign secret".to_vec(),
        },
    )
    .unwrap();
    split_timeline(
        ctx,
        &SplitRequest {
            dataset: Some(dataset.into()),
            plan: SplitPlan::Participants {
                locked_share: 0.3,
                folds,
            },
            seed,
            projection: ProjectionSpec::at_entry(),
        },
    )
    .unwrap()
}

/// The terms a policy label states, named `name`.
pub fn terms(policy: UsagePolicy, name: &str) -> Terms {
    policy.terms(name)
}

/// Steps a candidate that is meant to pass the gate is trained for: with the
/// cohorts below, early stopping ends the run near its best held-out likelihood.
pub const STEPS: u32 = 600;
/// Participants of the cohort that took the new measurements.
pub const FULL: usize = 12000;
/// Participants of the earlier cohort that recorded conventional factors only.
pub const CONVENTIONAL: usize = 12000;

/// The all-cause view's name.
pub const ALL_CAUSE: &str = "death:any";
/// The horizons the specs judge at.
pub const HORIZONS: [f64; 2] = [5.0, 10.0];

/// A small, fast training run.
pub fn training(steps: u32, seed: u64) -> TimelineTraining {
    let mut t = TimelineTraining::new(CODES, ABSORBING, seed);
    t.steps = steps;
    t.batch = 64;
    t.max_tokens = Some(16);
    t.eval_interval = 25;
    t
}

/// Trains a candidate on the parts of `split`.
pub fn train(ctx: &Context, split: &SplitReport, config: TimelineTraining) -> TimelineTrained {
    train_timeline_candidate(
        ctx,
        &TimelineTrainRequest::new(
            &split.train.to_string(),
            &split.validation.to_string(),
            config,
        ),
    )
    .unwrap()
}

/// The fewest validation events a code needs by a horizon to be calibrated
/// there in these specs: the cohorts are small, so brain's own minimum (30)
/// would leave the first horizon uncalibrated.
pub const CALIBRATION_MIN_EVENTS: usize = 10;

/// Trains a candidate on the parts of `split` and calibrates its risks at the
/// judged horizons on the share of the validation part early stopping does not
/// read.
pub fn train_calibrated(
    ctx: &Context,
    split: &SplitReport,
    config: TimelineTraining,
) -> TimelineTrained {
    let mut request = TimelineTrainRequest::new(
        &split.train.to_string(),
        &split.validation.to_string(),
        config,
    );
    // The cohorts here are small: calibration takes the half of the
    // validation part its floor gives, not the units a real cohort allows.
    request.calibration_units = 0;
    request.calibration = Some(CalibrationPlan {
        horizons: HORIZONS.to_vec(),
        min_events: Some(CALIBRATION_MIN_EVENTS),
        kind: None,
    });
    train_timeline_candidate(ctx, &request).unwrap()
}

/// How the arms are scored: the all-cause view and a subgroup, a few
/// resamples.
pub fn scoring() -> ScoreSpec {
    // seed: arms scored on separate occasions must resample alike
    let mut s = ScoreSpec::new(CODES, ABSORBING, HORIZONS.to_vec(), 1);
    s.all_cause = Some(ALL_CAUSE.into());
    s.bootstrap_reps = 100;
    s.subgroups = vec![Subgroup {
        name: "group_b".into(),
        rule: SubgroupRule::Category {
            var: "group".into(),
            level: "b".into(),
        },
    }];
    s
}

/// The horizon every calibrated code must be certified at in these specs.
pub const REQUIRED_HORIZON: f64 = HORIZONS[0];

/// The pre-registered requirements: the candidate must beat the champion on
/// the all-cause integrated Brier score and on the held-out likelihood, be
/// calibrated at every horizon the test part can judge, and at
/// [`REQUIRED_HORIZON`] certifiably, for every code in `calibrated` (and for
/// the all-cause view where it can be judged), and not regress on the
/// subgroup. A cohort that did not
/// measure the covariates a cause depends on is judged on the all-cause view
/// alone: its model cannot be asked to separate causes it cannot see.
pub fn plan_for(calibrated: &[&str]) -> TimelinePlan {
    let judged = |code: &str, must_improve| CodePlan {
        code: code.into(),
        horizons: HORIZONS.to_vec(),
        must_improve,
    };
    let mut codes: Vec<CodePlan> = calibrated.iter().map(|c| judged(c, false)).collect();
    codes.push(judged(ALL_CAUSE, true));
    TimelinePlan {
        required: calibrated
            .iter()
            .map(|code| RequiredHorizon {
                code: (*code).into(),
                horizon: REQUIRED_HORIZON,
            })
            .collect(),
        min_test_events: DEFAULT_MIN_TEST_EVENTS,
        codes,
        improve_nll: true,
        calibration: CalibrationBands {
            slope: (0.7, 1.4),
            // The calibration set of these small cohorts holds under a hundred
            // events, so slope and intercept both wander (they move together):
            // the specs test the pipeline, not the calibrator, and the sample's
            // bands are the registered ones.
            intercept_in_the_large: (-1.0, 1.0),
            oe: (0.8, 1.25),
            max_ece: 0.05,
        },
        subgroups: vec!["group_b".into()],
        subgroup_margin: 0.002,
        min_subgroup_events: 10,
        serve_tolerance: 1e-6,
        batch_tolerance: 1e-6,
        max_abstention_rate: 0.1,
    }
}

/// The requirements for a cohort that measured everything: every cause is judged.
pub fn plan() -> TimelinePlan {
    plan_for(&["death:a"])
}

/// Evaluates `candidate` against `champion` on the test part of `split`.
pub fn evaluate(
    ctx: &Context,
    candidate: &str,
    champion: Champion,
    split: &SplitReport,
) -> TimelineEvaluated {
    evaluate_under(ctx, candidate, champion, split, plan())
}

/// [`evaluate`] under the requirements `plan`.
pub fn evaluate_under(
    ctx: &Context,
    candidate: &str,
    champion: Champion,
    split: &SplitReport,
    plan: TimelinePlan,
) -> TimelineEvaluated {
    evaluate_timeline(
        ctx,
        &TimelineEvaluateRequest {
            candidate: candidate.into(),
            champion,
            test: split.test.to_string(),
            scoring: scoring(),
            plan,
        },
    )
    .unwrap()
}

/// The checks of a gate report that did not pass, one line each.
pub fn failures(report: &splinter_eval::predictive_gate::PredictiveReport) -> Vec<String> {
    let mut out = Vec::new();
    for (name, section) in [
        ("performance", &report.performance),
        ("calibration", &report.calibration),
        ("retention", &report.retention),
        ("serving", &report.serving),
    ] {
        for d in section.decided.iter().filter(|d| !d.check.passed) {
            out.push(format!(
                "{name}: {:?} -> {:?} / {:?}",
                d.requirement, d.check.measured, d.check.reason
            ));
        }
    }
    if !report.policy.passed {
        out.push(format!("policy: {:?}", report.policy.reason));
    }
    out
}

/// The longest one of the timeline specs may take. Measured alone each takes
/// well under two minutes; the bound is for a machine so oversubscribed
/// (brain's CPU backend takes every core it is allowed) that training does
/// not finish, which must fail the spec instead of holding the suite.
pub const DEADLINE: std::time::Duration = std::time::Duration::from_secs(15 * 60);

/// Runs `body` on a thread of its own and fails when it has not finished
/// within `limit`, naming the limit. A panic in `body` is the spec's panic.
pub fn bounded(limit: std::time::Duration, body: impl FnOnce() + Send + 'static) {
    let (done, finished) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        body();
        let _ = done.send(());
    });
    match finished.recv_timeout(limit) {
        // Finished, or the body panicked and dropped the sender: either way
        // the thread is done and its result says which.
        Ok(()) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            if let Err(panic) = worker.join() {
                std::panic::resume_unwind(panic);
            }
        }
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            panic!("the spec did not finish within {limit:?}: is the machine oversubscribed?")
        }
    }
}
