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
//! rationaliser, the sample and what makes the run what it is ([`keys`]).
//!
//! An attempt runs as an ordinary Harbor trial of a staged copy of the task
//! ([`trials`]), through the examiner's user-namespace sandbox, its model
//! gateway (serving a profile derived from the one given, whose main role
//! is the rationaliser) and its Sven adapter, so the model works the task
//! through the task's own tools in the task's own sandbox. The copy's
//! instruction is the task's with its reference solution appended in a
//! delimited hint section ([`filters::hinted_instruction`]). The attempt is
//! then judged ([`filters`]): by default it is kept when it is verified and
//! no answer content is present in the model's own text (the hint's wording,
//! a file line nothing had shown it). The real-observations rule and the
//! sufficiency run (a second trial of a copy whose instruction carries only
//! the attempt's reasoning, no hint, must pass the verifier) are [`Checks`]
//! a run opts into; so is a word list whose mention is a leak.
//!
//! **Re-grading** ([`Attempts::Regrade`]) judges again every graded attempt
//! of earlier runs with this run's checks, without running a hinted trial:
//! each attempt's packed job is unpacked in a copy in this run's directory
//! (the earlier run is only read) and its hint section is rebuilt with that
//! run's framing. Without the sufficiency run nothing runs at all: no
//! gateway, no Harbor job, no model.
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
//! is never redone; an attempt that passed the screening filters of a run
//! with the sufficiency run is stored as a candidate and resumes at that
//! run; one the infrastructure stopped (no grade, no trajectory) decided
//! nothing and is run again. A Harbor job that finished is read, not run
//! again.

pub mod filters;
mod keys;
pub mod material;
mod phases;
mod record;
mod regrade;
mod report;
mod trials;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::json;
use splinter_agent::CancelToken;
use splinter_core::model_ref::ModelRef;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

use super::select::rank;
use crate::bench::artifact::sha256_hex;
use crate::bench::practice::{free_port, policy_profile, BenchCheckout, Gateway};
use filters::Verdict;
use keys::digest_of;
use material::{stratified, student_trials, CompiledDataset, StudentTrial, TrainTask};
use phases::{hinted_phase, sufficiency_phase};
use record::{held, keep, Kept};
pub use record::{Filters, Origin, Outcome, RationaliseRecord, RegradeOf, RECORD_CLASS};
use report::{report, ReportParts};
pub use report::{Alone, Funnel, RationaliseReport, StudentSummary};
use trials::{read_attempt, JobEnv};

/// The format tag of a [`RationaliseReport`] and of its records and keys.
pub const RATIONALISE_FORMAT: &str = "splinter.bench-rationalise.v1";

/// Attempts of each picked task when the request does not say.
pub const DEFAULT_SAMPLES: usize = 1;

/// Trials run at once when the request does not say.
pub const DEFAULT_RATIONALISE_JOBS: usize = 8;

/// The checks a run opts into beyond the two every run applies (verified, no
/// answer present). Both are off by default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Checks {
    /// The real-observations rule ([`filters::read_before_write`]).
    pub real_observations: bool,
    /// The sufficiency run: the attempt's reasoning alone must pass again,
    /// in a second Harbor trial.
    pub sufficiency: bool,
}

impl Checks {
    /// Every check on: what every run used before the checks were options.
    pub const ALL: Checks = Checks {
        real_observations: true,
        sufficiency: true,
    };
}

/// Where a run's attempts come from.
#[derive(Clone, Copy, Debug)]
pub enum Attempts<'a> {
    /// Drawn from the student's failed tasks and run as hinted trials.
    Draw {
        /// Failed tasks to pick.
        count: usize,
        /// Attempts of each picked task.
        samples: usize,
        /// The seed of the pick.
        seed: u64,
    },
    /// Every graded attempt of these earlier runs' directories, judged
    /// again; no hinted trial runs.
    Regrade(&'a [PathBuf]),
}

/// One rationalisation run.
pub struct RationaliseRequest<'a> {
    /// The compiled dataset (holding `bench-compile.json`).
    pub dataset: &'a Path,
    /// The student's graded Harbor jobs (packed or not).
    pub students: &'a [PathBuf],
    /// The examiner that runs, grades and unpacks the attempts; its sven
    /// build is the agent's.
    pub checkout: &'a BenchCheckout,
    /// The model that rationalises: a local model the brain endpoint serves.
    /// Needed only when a trial runs (a draw, or the sufficiency run).
    pub rationaliser: Option<&'a ModelRef>,
    /// The profile the rationaliser's is derived from (its sampling, its
    /// sven settings). Needed only when a trial runs.
    pub profile: Option<&'a Path>,
    /// The examiner's cache of task images.
    pub images: &'a Path,
    /// Where the attempts come from.
    pub attempts: Attempts<'a>,
    /// Trials run at once.
    pub jobs: usize,
    /// The leak terms ([`filters::LEAK_TERMS`], none, unless a run names
    /// some).
    pub leak_terms: &'a [String],
    /// How the hint is framed for the rationaliser (a draw; a re-grade takes
    /// each run's own).
    pub framing: filters::HintFraming,
    /// The checks the run opts into.
    pub checks: Checks,
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

/// One attempt the run owes: its task, the student's trial and its key, and
/// for a re-grade the earlier attempt it judges again.
struct Case<'a> {
    task: &'a TrainTask,
    student: StudentTrial,
    sample: usize,
    key: String,
    earlier: Option<regrade::Earlier>,
}

/// What runs trials: the rationaliser's endpoint and the profile.
struct Runner {
    endpoint: splinter_orchestrator::context::ServedEndpoint,
    profile: PathBuf,
    profile_sha256: String,
}

/// Rationalises the student's failed train-side tasks, or re-grades earlier
/// runs' attempts; see the module documentation.
///
/// # Errors
/// [`OrchestratorError`] when the request is malformed, a student trial or
/// a re-graded attempt is not of a train-side task of the dataset, an
/// earlier run cannot be read or is of another dataset, the rationaliser is
/// not served when a trial runs, the examiner or Harbor fails, the store
/// fails, or the run is cancelled. An attempt that could not be graded is a
/// record, not an error.
#[allow(clippy::too_many_lines)]
pub fn rationalise(
    ctx: &Context,
    request: &RationaliseRequest<'_>,
) -> Result<RationaliseReport, OrchestratorError> {
    if request.jobs == 0 {
        return Err(refused("jobs must be at least one"));
    }
    match request.attempts {
        Attempts::Draw { count, samples, .. } if count == 0 || samples == 0 => {
            return Err(refused("count and samples must each be at least one"));
        }
        Attempts::Regrade([]) => {
            return Err(refused("a re-grade names at least one earlier run"));
        }
        Attempts::Regrade(_) if request.framing != filters::HintFraming::default() => {
            return Err(refused(
                "a re-grade takes each attempt's hint framing from its own run; name none",
            ));
        }
        _ => {}
    }
    if request.leak_terms.iter().any(|t| t.trim().is_empty()) {
        return Err(refused("a leak term is empty"));
    }
    let dataset = CompiledDataset::read(request.dataset).map_err(refused)?;
    let runs_trials =
        matches!(request.attempts, Attempts::Draw { .. }) || request.checks.sufficiency;
    let runner = if runs_trials {
        let (Some(rationaliser), Some(profile)) = (request.rationaliser, request.profile) else {
            return Err(refused(
                "trials run (a draw, or the sufficiency run): the rationaliser and the profile \
                 are needed",
            ));
        };
        let endpoint = ctx.served_endpoint(rationaliser)?;
        let bytes =
            std::fs::read(profile).map_err(|e| refused(format!("{}: {e}", profile.display())))?;
        Some(Runner {
            endpoint,
            profile: profile.to_path_buf(),
            profile_sha256: sha256_hex(&bytes),
        })
    } else {
        None
    };

    // Every trial of the student's jobs, each of a train-side task.
    let mut trials = Vec::new();
    for job in request.students {
        trials.extend(student_trials(job).map_err(refused)?);
    }
    let mut tasks: BTreeMap<String, TrainTask> = BTreeMap::new();
    let mut rel_of: BTreeMap<PathBuf, String> = BTreeMap::new();
    for trial in &trials {
        if !rel_of.contains_key(&trial.task) {
            let task = dataset.train_task(&trial.task).map_err(|why| {
                refused(format!(
                    "student trial {} of {}: {why}",
                    trial.trial,
                    trial.job.display()
                ))
            })?;
            rel_of.insert(trial.task.clone(), task.rel.clone());
            tasks.entry(task.rel.clone()).or_insert(task);
        }
    }
    let mut failed: BTreeMap<String, Vec<&StudentTrial>> = BTreeMap::new();
    for trial in trials.iter().filter(|t| t.failed()) {
        failed
            .entry(rel_of[&trial.task].clone())
            .or_default()
            .push(trial);
    }
    let students = StudentSummary {
        trials: trials.len(),
        passed: trials.iter().filter(|t| t.passed()).count(),
        failed: trials.iter().filter(|t| t.failed()).count(),
        failed_tasks: failed.len(),
    };
    let earlier = match request.attempts {
        Attempts::Regrade(runs) => {
            let earlier = regrade::earlier(runs, &dataset)?;
            for attempt in &earlier {
                let folder = &attempt.record.folder;
                if !tasks.contains_key(folder) {
                    let task = dataset
                        .train_task(&dataset.dir.join(folder))
                        .map_err(|why| {
                            refused(format!(
                                "attempt {} of {}: {why}",
                                attempt.record.key,
                                attempt.run.display()
                            ))
                        })?;
                    tasks.insert(folder.clone(), task);
                }
            }
            earlier
        }
        Attempts::Draw { .. } => Vec::new(),
    };
    let identity = runner.as_ref().map(|r| r.endpoint.identity.clone());
    let mut picked = Vec::new();
    let mut cases = Vec::new();
    let run_name = match request.attempts {
        Attempts::Draw {
            count,
            samples,
            seed,
        } => {
            let runner = runner
                .as_ref()
                .ok_or_else(|| refused("a draw runs trials"))?;
            let identity = &runner.endpoint.identity;
            let variant = keys::draw_variant(
                &runner.profile_sha256,
                &runner.endpoint.model,
                request.leak_terms,
                request.framing,
                request.checks,
            )?;
            let chosen = stratified(
                failed.keys().map(String::as_str).collect(),
                |rel| tasks[*rel].difficulty,
                |rel| rel,
                count,
                seed,
            );
            for rel in &chosen {
                let task = &tasks[*rel];
                let Some(student) = failed[*rel].iter().min_by_key(|t| rank(seed, &t.trial)) else {
                    continue;
                };
                picked.push(Picked {
                    task: task.entry.name.clone(),
                    folder: task.rel.clone(),
                    difficulty: task.difficulty,
                    student_trial: student.trial.clone(),
                });
                for sample in 0..samples {
                    cases.push(Case {
                        task,
                        student: (*student).clone(),
                        sample,
                        key: keys::attempt_key(&student.trial, identity, sample, &variant)?,
                        earlier: None,
                    });
                }
            }
            format!(
                "rationalise-{}",
                &digest_of(&json!({
                    "format": RATIONALISE_FORMAT,
                    "dataset": dataset.sha256,
                    "variant": variant,
                    "rationaliser": identity,
                    "seed": seed,
                    "count": count,
                    "samples": samples,
                    "students": request.students,
                }))?[..16]
            )
        }
        Attempts::Regrade(_) => {
            let variant = keys::regrade_variant(
                request.leak_terms,
                request.checks,
                runner.as_ref().map(|r| {
                    (
                        r.profile_sha256.as_str(),
                        r.endpoint.model.as_str(),
                        r.endpoint.identity.as_str(),
                    )
                }),
            )?;
            for attempt in earlier {
                let task = &tasks[&attempt.record.folder];
                cases.push(Case {
                    task,
                    student: attempt.student(),
                    sample: attempt.record.sample,
                    key: keys::regrade_key(&attempt.record.key, &variant)?,
                    earlier: Some(attempt),
                });
            }
            let runs: Vec<&PathBuf> = cases
                .iter()
                .filter_map(|c| c.earlier.as_ref().map(|e| &e.run))
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect();
            format!(
                "regrade-{}",
                &digest_of(&json!({
                    "format": RATIONALISE_FORMAT,
                    "dataset": dataset.sha256,
                    "variant": variant,
                    "regrade": runs,
                    "students": request.students,
                }))?[..16]
            )
        }
    };
    let dir = ctx.config().state_root.rationalise().join(&run_name);
    std::fs::create_dir_all(&dir).map_err(|e| refused(format!("{}: {e}", dir.display())))?;
    let mut held = held(ctx)?;

    // The student's own passes, kept as they are.
    let copies = dir.join("students");
    for trial in trials.iter().filter(|t| t.passed()) {
        let task = &tasks[&rel_of[&trial.task]];
        let key = keys::own_key(trial, task)?;
        if held.get(&key).is_some_and(|r| r.outcome.decided()) {
            continue;
        }
        let mut record = RationaliseRecord::new(&key, Origin::Own, task, trial, None, 0);
        record.filters.verified = Some(Verdict::pass());
        let made = trials::job_trial_dir(request.checkout, &trial.job, &trial.trial, &copies)
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

    let pending: Vec<&Case<'_>> = cases
        .iter()
        .filter(|c| {
            held.get(&c.key)
                .is_none_or(|r| r.outcome == Outcome::Refused)
        })
        .collect();
    // A re-grade judges its pending attempts offline, before any trial.
    let hinted: Vec<&Case<'_>> = if matches!(request.attempts, Attempts::Regrade(_)) {
        if request.cancel.is_cancelled() {
            return Err(OrchestratorError::Cancelled);
        }
        for (key, record) in regrade::phase(ctx, request, &dir, &pending)? {
            held.insert(key, record);
        }
        Vec::new()
    } else {
        pending
    };
    let has_candidates = cases.iter().any(|c| {
        held.get(&c.key)
            .is_some_and(|r| r.outcome == Outcome::Candidate)
    });
    if !hinted.is_empty() || has_candidates {
        if request.cancel.is_cancelled() {
            return Err(OrchestratorError::Cancelled);
        }
        let runner = runner
            .as_ref()
            .ok_or_else(|| refused("trials are owed, and no rationaliser runs them"))?;
        let profile = dir.join(format!("{run_name}.toml"));
        let text = policy_profile(&runner.profile, &runner.endpoint, &run_name)
            .map_err(|e| refused(e.to_string()))?;
        std::fs::write(&profile, text)
            .map_err(|e| refused(format!("{}: {e}", profile.display())))?;
        let port = free_port().map_err(|e| refused(e.to_string()))?;
        let state = dir.join("gateway-state.json");
        let _gateway = Gateway::start(
            request.checkout,
            &profile,
            port,
            &runner.endpoint.key,
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
            let identity = &runner.endpoint.identity;
            for (key, record) in hinted_phase(ctx, &env, &dir, &hinted, request, identity)? {
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
            for (key, record) in sufficiency_phase(ctx, &env, &dir, &candidates)? {
                held.insert(key, record);
            }
        }
    }

    let mut records: Vec<RationaliseRecord> = Vec::new();
    for trial in trials.iter().filter(|t| t.passed()) {
        let key = keys::own_key(trial, &tasks[&rel_of[&trial.task]])?;
        records.extend(held.get(&key).cloned());
    }
    records.extend(cases.iter().filter_map(|c| held.get(&c.key).cloned()));
    let regraded = match request.attempts {
        Attempts::Regrade(_) => regrade::runs_of(&cases),
        Attempts::Draw { .. } => Vec::new(),
    };
    report(
        ctx,
        request,
        records,
        ReportParts {
            rationaliser: identity,
            served_as: runner.as_ref().map(|r| r.endpoint.model.clone()),
            dataset: json!({"dir": dataset.dir, "compile_sha256": dataset.sha256,
            "bench_commit": dataset.bench_commit}),
            profile_sha256: runner.as_ref().map(|r| r.profile_sha256.clone()),
            students,
            picked,
            regraded,
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
