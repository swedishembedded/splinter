// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the loop from a failed agent attempt to a
// verified corrected one, for its clients. If your team needs expertise in
// turning agent failures into training data you can procure our services by
// sending an email to info@swedishembedded.com.

//! The hinted retry: rationalisation of a model's failed attempts at
//! train-side tasks of a compiled Harbor dataset into verified trajectories
//! it could have produced unaided.
//!
//! The student's graded Harbor jobs (packed by `bench pack` or not) name the
//! tasks it failed. Every task of them must be one the dataset's
//! `bench-compile.json` puts on the `train` side and no marker seals
//! ([`material`]); any other refuses the run, naming the task and why. A
//! seeded draw, stratified by the tasks' difficulty rung
//! (`[metadata].bench_difficulty`), picks the failed tasks to rationalise,
//! and for each the student's failed trial the attempt answers. Each sample
//! of each picked task is one attempt, keyed by the student's trial, the
//! rationaliser, the sample and what makes the run what it is (the profile,
//! the endpoint's model id, the leak terms).
//!
//! An attempt runs as an ordinary Harbor trial of a staged copy of the task
//! ([`trials`]), through the examiner's user-namespace sandbox, its model
//! gateway (serving a profile derived from the one given, whose main role
//! is the rationaliser) and its Sven adapter, so the model works the task
//! through the task's own tools in the task's own sandbox. The copy's
//! instruction is the task's with its reference solution appended in a
//! delimited hint section ([`filters::hinted_instruction`]). The attempt is
//! then judged by the filters in order ([`filters`]): verified, real
//! observations, no leak, and, for an attempt that passed those three,
//! sufficient: a second trial of a copy whose instruction carries only the
//! attempt's reasoning, no hint, must pass the verifier.
//!
//! A kept attempt becomes an experience of the task on the train side: its
//! trajectory is the attempt's with the user turn the task's own
//! instruction (the hint section nowhere in it), the system prompt and tool
//! schemas of its first model request, and its lineage (`extra.
//! splinter_rationalise`: origin, key, trials, every filter's verdict); its
//! verdict is the verifier's pass; it is registered on the train side of
//! the trial register; and its provenance marks the rationaliser as having
//! seen material the student does not (`teacher`). The student's own passes
//! become experiences the same way, with origin `own` and the trajectory as
//! the student left it. Every record, kept or not, is stored with every
//! verdict reached, so rejection rates can be reported, and the run's kept
//! experiences are one experience set, which `dataset build --view sft-step`
//! makes the training records of.
//!
//! **Idempotent and resumable.** A record that decided (kept or rejected)
//! is never redone; an attempt that passed the first three filters is
//! stored as a candidate and resumes at the sufficiency run; one the
//! infrastructure stopped (no grade, no trajectory) decided nothing and is
//! run again. A Harbor job that finished is read, not run again.

pub mod filters;
pub mod material;
mod record;
mod report;
mod trials;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::json;
use splinter_agent::CancelToken;
use splinter_core::digest::{canonical_json, Digest};
use splinter_core::model_ref::ModelRef;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

use super::route::InterventionKind;
use super::select::rank;
use crate::bench::artifact::sha256_hex;
use crate::bench::practice::{free_port, policy_profile, BenchCheckout, Gateway};
use filters::{LeakCheck, Verdict};
use material::{stratified, student_trials, CompiledDataset, StudentTrial, TrainTask};
use record::{held, keep, Kept};
pub use record::{Filters, Origin, Outcome, RationaliseRecord, RECORD_CLASS};
use report::{report, ReportParts};
pub use report::{Alone, Funnel, RationaliseReport, StudentSummary};
use trials::{read_attempt, run_job, stage_copy, staged_name, JobEnv};

/// The format tag of a [`RationaliseReport`] and of its records and keys.
pub const RATIONALISE_FORMAT: &str = "splinter.bench-rationalise.v1";

/// Attempts of each picked task when the request does not say.
pub const DEFAULT_SAMPLES: usize = 1;

/// Trials run at once when the request does not say.
pub const DEFAULT_RATIONALISE_JOBS: usize = 8;

/// One rationalisation run.
pub struct RationaliseRequest<'a> {
    /// The compiled dataset (holding `bench-compile.json`).
    pub dataset: &'a Path,
    /// The student's graded Harbor jobs (packed or not).
    pub students: &'a [PathBuf],
    /// The examiner that runs and grades the attempts; its sven build is
    /// the agent's.
    pub checkout: &'a BenchCheckout,
    /// The model that rationalises: a local model the brain endpoint serves.
    pub rationaliser: &'a ModelRef,
    /// The profile the rationaliser's is derived from (its sampling, its
    /// sven settings).
    pub profile: &'a Path,
    /// The examiner's cache of task images.
    pub images: &'a Path,
    /// Failed tasks to pick.
    pub count: usize,
    /// Attempts of each picked task.
    pub samples: usize,
    /// The seed of the pick.
    pub seed: u64,
    /// Trials run at once.
    pub jobs: usize,
    /// The leak terms ([`filters::LEAK_TERMS`] unless a run names others).
    pub leak_terms: &'a [String],
    /// Stops the run between its phases.
    pub cancel: CancelToken,
}

/// A failed task the draw picked.
#[derive(Clone, Debug, Serialize)]
pub struct Picked {
    /// The task's name.
    pub task: String,
    /// Its folder in the dataset.
    pub folder: String,
    /// Its difficulty rung.
    pub difficulty: Option<i64>,
    /// The student's failed trial the attempts answer.
    pub student_trial: String,
}

fn refused(why: impl std::fmt::Display) -> OrchestratorError {
    OrchestratorError::Refused(why.to_string())
}

fn digest_of<T: Serialize>(value: &T) -> Result<String, OrchestratorError> {
    let bytes = canonical_json(value).map_err(|e| refused(format!("a key: {e}")))?;
    Ok(Digest::of(&bytes).hex().to_string())
}

/// The key of the student's own pass `trial` of `task`.
fn own_key(trial: &StudentTrial, task: &TrainTask) -> Result<String, OrchestratorError> {
    digest_of(&json!({
        "format": RATIONALISE_FORMAT, "origin": "own", "trial": trial.trial, "task": task.rel,
    }))
}

/// One attempt the run owes: its task, the student's trial and its key.
struct Case<'a> {
    task: &'a TrainTask,
    student: &'a StudentTrial,
    sample: usize,
    key: String,
}

/// Rationalises the student's failed train-side tasks; see the module
/// documentation.
///
/// # Errors
/// [`OrchestratorError`] when the request is malformed, a student trial is
/// not of a train-side task of the dataset, the rationaliser is not served,
/// the examiner or Harbor fails, the store fails, or the run is cancelled.
/// An attempt that could not be graded is a record, not an error.
#[allow(clippy::too_many_lines)]
pub fn rationalise(
    ctx: &Context,
    request: &RationaliseRequest<'_>,
) -> Result<RationaliseReport, OrchestratorError> {
    if request.samples == 0 || request.jobs == 0 || request.count == 0 {
        return Err(refused("count, samples and jobs must each be at least one"));
    }
    if request.leak_terms.iter().any(|t| t.trim().is_empty()) {
        return Err(refused("a leak term is empty"));
    }
    let dataset = CompiledDataset::read(request.dataset).map_err(refused)?;
    let endpoint = ctx.served_endpoint(request.rationaliser)?;
    let profile_bytes = std::fs::read(request.profile)
        .map_err(|e| refused(format!("{}: {e}", request.profile.display())))?;
    let profile_sha256 = sha256_hex(&profile_bytes);

    // Every trial of the student's jobs, each of a train-side task.
    let mut trials = Vec::new();
    for job in request.students {
        trials.extend(student_trials(job).map_err(refused)?);
    }
    let mut tasks: BTreeMap<PathBuf, TrainTask> = BTreeMap::new();
    for trial in &trials {
        if !tasks.contains_key(&trial.task) {
            let task = dataset.train_task(&trial.task).map_err(|why| {
                refused(format!(
                    "student trial {} of {}: {why}",
                    trial.trial,
                    trial.job.display()
                ))
            })?;
            tasks.insert(trial.task.clone(), task);
        }
    }
    let mut failed: BTreeMap<&str, Vec<&StudentTrial>> = BTreeMap::new();
    for trial in trials.iter().filter(|t| t.failed()) {
        failed
            .entry(tasks[&trial.task].rel.as_str())
            .or_default()
            .push(trial);
    }
    let students = StudentSummary {
        trials: trials.len(),
        passed: trials.iter().filter(|t| t.passed()).count(),
        failed: trials.iter().filter(|t| t.failed()).count(),
        failed_tasks: failed.len(),
    };
    let by_rel: BTreeMap<&str, &TrainTask> = tasks.values().map(|t| (t.rel.as_str(), t)).collect();
    let chosen = stratified(
        failed.keys().copied().collect(),
        |rel| by_rel[rel].difficulty,
        |rel| rel,
        request.count,
        request.seed,
    );
    let identity = endpoint.identity.clone();
    let variant = digest_of(&json!({
        "profile_sha256": profile_sha256,
        "served_as": endpoint.model,
        "leak_terms": request.leak_terms,
    }))?;
    let mut picked = Vec::new();
    let mut cases = Vec::new();
    for rel in &chosen {
        let task = by_rel[rel];
        let Some(student) = failed[rel]
            .iter()
            .min_by_key(|t| rank(request.seed, &t.trial))
        else {
            continue;
        };
        picked.push(Picked {
            task: task.entry.name.clone(),
            folder: task.rel.clone(),
            difficulty: task.difficulty,
            student_trial: student.trial.clone(),
        });
        for sample in 0..request.samples {
            let key = digest_of(&json!({
                "format": RATIONALISE_FORMAT,
                "intervention": InterventionKind::HintedRetry.as_str(),
                "student": student.trial,
                "rationaliser": identity,
                "sample": sample,
                "variant": variant,
            }))?;
            cases.push(Case {
                task,
                student,
                sample,
                key,
            });
        }
    }
    let run_name = format!(
        "rationalise-{}",
        &digest_of(&json!({
            "format": RATIONALISE_FORMAT,
            "dataset": dataset.sha256,
            "variant": variant,
            "rationaliser": identity,
            "seed": request.seed,
            "count": request.count,
            "samples": request.samples,
            "students": request.students,
        }))?[..16]
    );
    let dir = ctx.config().state_root.rationalise().join(&run_name);
    std::fs::create_dir_all(&dir).map_err(|e| refused(format!("{}: {e}", dir.display())))?;
    let mut held = held(ctx)?;

    // The student's own passes, kept as they are.
    let copies = dir.join("students");
    for trial in trials.iter().filter(|t| t.passed()) {
        let task = &tasks[&trial.task];
        let key = own_key(trial, task)?;
        if held.get(&key).is_some_and(|r| r.outcome.decided()) {
            continue;
        }
        let mut record = RationaliseRecord::new(&key, Origin::Own, task, trial, None, 0);
        record.filters.verified = Some(Verdict::pass());
        let made = trials::student_trial_dir(request.checkout, &trial.job, &trial.trial, &copies)
            .and_then(|trial_dir| {
                let (trajectory, context) = read_attempt(&trial_dir)?;
                let solver = student_identity(&trajectory);
                Ok(Kept {
                    trajectory,
                    context,
                    trial: trial.trial.clone(),
                    result_sha256: trial_result_sha(&trial_dir),
                    solver,
                    helped: false,
                })
            });
        let record = match made {
            Ok(kept) => keep(ctx, record, task, Some(&kept))?,
            Err(why) => keep(ctx, record.refuse(why), task, None)?,
        };
        held.insert(key, record);
    }

    let hinted: Vec<&Case<'_>> = cases
        .iter()
        .filter(|c| {
            held.get(&c.key)
                .is_none_or(|r| r.outcome == Outcome::Refused)
        })
        .collect();
    let has_candidates = cases.iter().any(|c| {
        held.get(&c.key)
            .is_some_and(|r| r.outcome == Outcome::Candidate)
    });
    if !hinted.is_empty() || has_candidates {
        if request.cancel.is_cancelled() {
            return Err(OrchestratorError::Cancelled);
        }
        let profile = dir.join(format!("{run_name}.toml"));
        let text = policy_profile(request.profile, &endpoint, &run_name)
            .map_err(|e| refused(e.to_string()))?;
        std::fs::write(&profile, text)
            .map_err(|e| refused(format!("{}: {e}", profile.display())))?;
        let port = free_port().map_err(|e| refused(e.to_string()))?;
        let state = dir.join("gateway-state.json");
        let _gateway = Gateway::start(
            request.checkout,
            &profile,
            port,
            &endpoint.key,
            Some(&state),
            &dir.join("gateway.log"),
        )
        .map_err(|why| refused(format!("the gateway: {why}")))?;
        let env = JobEnv {
            checkout: request.checkout,
            profile: &profile,
            gateway_state: &state,
            port,
            images: request.images,
            jobs: request.jobs,
        };
        if !hinted.is_empty() {
            for (key, record) in hinted_phase(ctx, &env, &dir, &hinted, request, &identity)? {
                held.insert(key, record);
            }
        }
        if request.cancel.is_cancelled() {
            return Err(OrchestratorError::Cancelled);
        }
        let candidates: Vec<(&Case<'_>, RationaliseRecord)> = cases
            .iter()
            .filter_map(|c| {
                let record = held.get(&c.key)?;
                (record.outcome == Outcome::Candidate).then(|| (c, record.clone()))
            })
            .collect();
        if !candidates.is_empty() {
            for (key, record) in sufficiency_phase(ctx, &env, &dir, &candidates, &identity)? {
                held.insert(key, record);
            }
        }
    }

    let mut records: Vec<RationaliseRecord> = Vec::new();
    for trial in trials.iter().filter(|t| t.passed()) {
        records.extend(held.get(&own_key(trial, &tasks[&trial.task])?).cloned());
    }
    records.extend(cases.iter().filter_map(|c| held.get(&c.key).cloned()));
    report(
        ctx,
        request,
        records,
        ReportParts {
            rationaliser: identity,
            served_as: endpoint.model,
            dataset: json!({"dir": dataset.dir, "compile_sha256": dataset.sha256,
            "bench_commit": dataset.bench_commit}),
            profile_sha256,
            students,
            picked,
            files: tasks
                .values()
                .map(|t| (t.rel.clone(), (t.files.clone(), t.instruction.clone())))
                .collect(),
            dir,
        },
    )
}

/// The student as a record names it: the agent and the model its first
/// agent step names, as Harbor trials are named elsewhere.
fn student_identity(trajectory: &atif::Trajectory) -> String {
    let model = trajectory
        .steps
        .iter()
        .find_map(|s| s.model_name.as_deref())
        .or(trajectory.agent.model_name.as_deref())
        .unwrap_or("unknown");
    format!(
        "harbor/{}@{}/{model}",
        trajectory.agent.name, trajectory.agent.version
    )
}

/// The sha256 of a trial directory's `result.json`, empty when it has none.
fn trial_result_sha(dir: &Path) -> String {
    std::fs::read(dir.join("result.json"))
        .map(|b| sha256_hex(&b))
        .unwrap_or_default()
}

/// Runs the hinted attempts of `cases` as one Harbor job and judges each by
/// the first three filters; returns the stored records by key.
fn hinted_phase(
    ctx: &Context,
    env: &JobEnv<'_>,
    dir: &Path,
    cases: &[&Case<'_>],
    request: &RationaliseRequest<'_>,
    identity: &str,
) -> Result<Vec<(String, RationaliseRecord)>, OrchestratorError> {
    let mut keys: Vec<&str> = cases.iter().map(|c| c.key.as_str()).collect();
    keys.sort_unstable();
    let name = format!("hinted-{}", &digest_of(&keys)?[..16]);
    let tasks_dir = dir.join(&name).join("tasks");
    let mut staged = BTreeMap::new();
    for case in cases {
        let folder = staged_name(&case.key, &folder_name(case.task));
        let instruction = filters::hinted_instruction(&case.task.instruction, &case.task.hint);
        stage_copy(case.task, &instruction, &tasks_dir, &folder).map_err(refused)?;
        staged.insert(case.key.as_str(), (folder, instruction));
    }
    let attempts =
        run_job(env, &tasks_dir, &dir.join(&name).join("jobs"), &name).map_err(refused)?;
    let mut out = Vec::new();
    for case in cases {
        let (folder, instruction) = &staged[case.key.as_str()];
        let mut record = RationaliseRecord::new(
            &case.key,
            Origin::Rationalised,
            case.task,
            case.student,
            Some(identity),
            case.sample,
        );
        let Some(attempt) = attempts.get(folder) else {
            out.push(stored(
                ctx,
                record.refuse("Harbor left no trial of the staged copy"),
                case,
            )?);
            continue;
        };
        record.hinted_trial = Some(attempt.dir.clone());
        let Some(reward) = attempt.reward else {
            out.push(stored(
                ctx,
                record.refuse("the hinted trial ended without a grade"),
                case,
            )?);
            continue;
        };
        let (trajectory, context) = match read_attempt(&attempt.dir) {
            Ok(read) => read,
            Err(why) => {
                out.push(stored(ctx, record.refuse(why), case)?);
                continue;
            }
        };
        let section = instruction
            .find(filters::HINT_BEGIN)
            .map_or("", |at| &instruction[at..]);
        let shown = format!(
            "{}\n{}",
            context.system,
            serde_json::to_string(&context.tools).unwrap_or_default()
        );
        record.filters.verified = Some(filters::verified(Some(reward)));
        record.filters.real_observations = Some(filters::read_before_write(
            &trajectory,
            &case.task.files,
            &case.task.instruction,
        ));
        record.filters.no_leak = Some(filters::no_leak(
            &trajectory,
            &LeakCheck {
                terms: request.leak_terms,
                hint: section,
                instruction: &case.task.instruction,
                shown: &shown,
                files: &case.task.files,
            },
        ));
        record.outcome = if record.filters.first_three_pass() {
            Outcome::Candidate
        } else {
            Outcome::Rejected
        };
        out.push(stored(ctx, record, case)?);
    }
    Ok(out)
}

/// Stores `record` of `case` with no experience.
fn stored(
    ctx: &Context,
    record: RationaliseRecord,
    case: &Case<'_>,
) -> Result<(String, RationaliseRecord), OrchestratorError> {
    Ok((case.key.clone(), keep(ctx, record, case.task, None)?))
}

/// The task folder's own name.
fn folder_name(task: &TrainTask) -> String {
    task.folder
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| task.rel.replace('/', "-"))
}

/// Runs the sufficiency trial of every candidate as one Harbor job: the
/// task's instruction and the candidate's reasoning, no hint. A pass keeps
/// the candidate; returns the stored records by key.
fn sufficiency_phase(
    ctx: &Context,
    env: &JobEnv<'_>,
    dir: &Path,
    candidates: &[(&Case<'_>, RationaliseRecord)],
    identity: &str,
) -> Result<Vec<(String, RationaliseRecord)>, OrchestratorError> {
    let mut keys: Vec<&str> = candidates.iter().map(|(c, _)| c.key.as_str()).collect();
    keys.sort_unstable();
    let name = format!("sufficiency-{}", &digest_of(&keys)?[..16]);
    let tasks_dir = dir.join(&name).join("tasks");
    let mut hinted = BTreeMap::new();
    let mut out = Vec::new();
    for (case, record) in candidates {
        let read = record
            .hinted_trial
            .as_deref()
            .ok_or_else(|| "the candidate names no hinted trial".to_string())
            .and_then(read_attempt);
        match read {
            Ok((trajectory, context)) => {
                let notes = filters::reasoning_of(&trajectory);
                let instruction = filters::notes_instruction(&case.task.instruction, &notes);
                let folder = staged_name(&case.key, &folder_name(case.task));
                stage_copy(case.task, &instruction, &tasks_dir, &folder).map_err(refused)?;
                hinted.insert(case.key.as_str(), (folder, trajectory, context));
            }
            Err(why) => out.push(stored(ctx, record.clone().refuse(why), case)?),
        }
    }
    if hinted.is_empty() {
        return Ok(out);
    }
    let attempts =
        run_job(env, &tasks_dir, &dir.join(&name).join("jobs"), &name).map_err(refused)?;
    for (case, record) in candidates {
        let Some((folder, trajectory, context)) = hinted.remove(case.key.as_str()) else {
            continue;
        };
        let mut record = record.clone();
        let Some(attempt) = attempts.get(&folder) else {
            record.reason = Some("Harbor left no sufficiency trial; it is run again".into());
            out.push(stored(ctx, record, case)?);
            continue;
        };
        record.sufficiency_trial = Some(attempt.dir.clone());
        let Some(reward) = attempt.reward else {
            record.reason =
                Some("the sufficiency trial ended without a grade; it is run again".into());
            out.push(stored(ctx, record, case)?);
            continue;
        };
        let sufficient = filters::verified(Some(reward));
        record.reason = None;
        record.filters.sufficient = Some(sufficient.clone());
        if !sufficient.passed {
            record.outcome = Outcome::Rejected;
            out.push(stored(ctx, record, case)?);
            continue;
        }
        let hinted_dir = record.hinted_trial.clone().unwrap_or_default();
        let kept =
            filters::kept_trajectory(&trajectory, &case.task.instruction).map(|trajectory| Kept {
                trajectory,
                context,
                trial: hinted_dir
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default(),
                result_sha256: trial_result_sha(&hinted_dir),
                solver: identity.to_string(),
                helped: true,
            });
        let record = match kept {
            Ok(kept) => {
                record.outcome = Outcome::Kept;
                keep(ctx, record, case.task, Some(&kept))?
            }
            Err(why) => {
                record.outcome = Outcome::Rejected;
                record.reason = Some(format!("the kept trajectory cannot be made: {why}"));
                keep(ctx, record, case.task, None)?
            }
        };
        out.push((case.key.clone(), record));
    }
    Ok(out)
}
