// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that turn an examiner's practice
// material into graded agent experience for its clients. If your team needs
// expertise in on-policy experience collection for agent training you can
// procure our services by sending an email to info@swedishembedded.com.

//! The examiner's practice material run under a policy: the staged,
//! Harbor-runnable train side of a collection (`bench task practice`,
//! `practice.json` of schema `bench.practice/2`) attempted by the policy
//! brain serves, graded by the examiner's verifiers, exported and admitted
//! as experience through the one intake every benchmark trial takes
//! ([`super::ingest_bench_trials`]).
//!
//! The examiner owns everything that runs inside a trial; the policy is
//! named to it the way it documents: a profile whose `main` role is
//! forwarded by its model gateway to the endpoint and id brain serves the
//! policy under. A run therefore
//!
//! 1. derives that profile from the one the material was staged under (or
//!    the one given): the main role's upstream becomes the policy, nothing
//!    else changes, and the examiner's files are never edited;
//! 2. starts the examiner's gateway for it (`bench gateway serve`), which
//!    holds the upstream key and gives each trial its own token;
//! 3. stages the same material again under that profile into the run's own
//!    directory (`bench task practice`), which writes the job config Harbor
//!    runs, registers the job with the gateway and records the run, and
//!    refuses the result unless every staged task, the draw and the
//!    collection's digests are the ones the given `practice.json` names;
//! 4. runs the job (`harbor run -c`) with the attempts, concurrency and
//!    tasks this run owns, under a job name derived from them and the seed;
//! 5. exports the graded trials (`bench export`) and admits them.
//!
//! Each trial comes back with a typed outcome read from the examiner's
//! score record (`bench.score/1`): passed or failed when it is counted,
//! exposed when the agent reached for the hidden answers, a verifier
//! failure, ungraded (a time limit the task does not grade, a spent budget,
//! an unsupported capability, a defective task) or an infrastructure
//! failure. Only a passed, failed or exposed trial has a reward.
//!
//! **Resumable.** A run is keyed by the material, the policy, the profile,
//! the attempts, the tasks and the seed, and keeps its own directory. A
//! finished Harbor job is not run again: a rerun exports again (the export
//! is idempotent) and admits nothing twice. A job that did not finish is
//! staged and run again whole, because the examiner's gateway forgets the
//! jobs registered with it when it stops and only a staging registers one.
//!
//! The seed names the job and is recorded. How the policy samples is the
//! profile's: its `[sven.sampling]` (temperature, top-p, top-k, the run seed
//! each trial's own seed is derived from) when it has one, which the report
//! records as `sampling`; without one, as the policy's server does.

mod commands;
mod executor;
pub mod material;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::Value;
use splinter_agent::CancelToken;
use splinter_core::digest::{canonical_json, Digest};
use splinter_core::experience::ExperienceId;
use splinter_core::model_ref::ModelRef;
use splinter_core::usage::TokenUsage;
use splinter_orchestrator::context::Context;

pub use commands::{BenchCheckout, PracticeDraw};
pub use executor::BenchPractice;
pub use material::{Practice, PracticeFacts, TaskFacts, PRACTICE_FILE, PRACTICE_SCHEMA};

use super::score::{ScoreRecord, Status};
use super::{ingest_bench_trials, split, BenchIngested, IngestOptions};
use crate::episodes::{EpisodeError, EpisodeOutcome, PolicyIdentity};
pub(crate) use commands::KEY_VARIABLE;
use commands::{HarborRun, Skipped};

/// The format tag of a [`PracticeReport`] and of the keys it is found by.
pub const PRACTICE_FORMAT: &str = "splinter.bench-practice.v1";

/// How a practice run is run. The seed has no default: a run is told its
/// seed, and records it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct PracticeSettings {
    /// Attempts of each task.
    pub attempts: u32,
    /// The seed the run is named by.
    pub seed: u64,
    /// Trials run at once.
    pub jobs: usize,
}

/// One practice run.
pub struct PracticeRequest<'a> {
    /// The material.
    pub practice: &'a Practice,
    /// The examiner.
    pub checkout: &'a BenchCheckout,
    /// The profile to derive the policy's from, in place of the one the
    /// material was staged under.
    pub profile: Option<&'a Path>,
    /// The policy.
    pub policy: &'a ModelRef,
    /// The adapter the policy serves, by content, when the caller knows it;
    /// otherwise it is read from the adapter file the reference names.
    pub adapter: Option<Digest>,
    /// How.
    pub settings: PracticeSettings,
    /// The staged tasks to run, by folder name; `None` runs them all.
    pub tasks: Option<&'a [String]>,
    /// Stops the run between its steps.
    pub cancel: CancelToken,
}

/// One trial of the run, as it ended.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PracticeTrial {
    /// The trial's name.
    pub trial: String,
    /// The staged task folder it attempted.
    pub task: String,
    /// The examiner's status, when it gave one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    /// How it ended.
    pub outcome: EpisodeOutcome,
    /// Its reward ([`EpisodeOutcome::reward`]).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reward: Option<f64>,
    /// The experiences it became (one, or one per conversation).
    pub experiences: Vec<ExperienceId>,
    /// The tokens its model requests processed (prompt and completion), by
    /// its score record's usage; `None` when the record states none.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens: Option<u64>,
    /// What its model requests processed, cached prompt tokens apart, by its
    /// score record's usage; `None` when the record states none.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<TokenUsage>,
    /// The share of its task's weighted checks it passed
    /// (`functional_correctness`), as its grade records it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub correctness: Option<f64>,
}

/// Where a run's material came from, recorded with what it produced.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PracticeProvenance {
    /// The practice directory.
    pub dir: PathBuf,
    /// The sha256 of its `practice.json`.
    pub practice_sha256: String,
    /// The collection.
    pub collection: String,
    /// The sha256 of the collection's manifest.
    pub manifest_sha256: Option<String>,
    /// The sha256 of its content.
    pub content_sha256: Option<String>,
    /// The draw, when the material is one.
    pub draw: Option<Value>,
    /// The examiner's version and commit that staged it.
    pub bench: Value,
    /// The profile the policy's was derived from, and its sha256.
    pub profile: PathBuf,
    /// Its sha256.
    pub profile_sha256: String,
}

/// What a practice run did.
#[derive(Clone, Debug, Serialize)]
pub struct PracticeReport {
    /// Always [`PRACTICE_FORMAT`].
    pub format: &'static str,
    /// The policy.
    pub policy: PolicyIdentity,
    /// The id the policy's server lists it under.
    pub served_as: String,
    /// The run's seed.
    pub seed: u64,
    /// The profile's `[sven.sampling]`, as the examiner samples every trial
    /// with it; `None`: the profile sets none, and the policy sampled as its
    /// server does.
    pub sampling: Option<Value>,
    /// Attempts of each task.
    pub attempts: u32,
    /// The material.
    pub provenance: PracticeProvenance,
    /// The run's directory: the derived profile, the staging, the export.
    pub dir: PathBuf,
    /// The Harbor job's name (`<dir>/staged/<collection>/<job>`).
    pub job: String,
    /// Whether Harbor ran now; `false` when the job had finished before.
    pub ran: bool,
    /// What the intake admitted.
    pub ingested: BenchIngested,
    /// Every trial of the job.
    pub trials: Vec<PracticeTrial>,
}

/// What a run is about: the material, the policy and the profile.
#[derive(Serialize)]
struct StageKey<'a> {
    format: &'static str,
    practice: &'a str,
    policy: &'a PolicyIdentity,
    served_as: &'a str,
    profile: &'a str,
}

/// What a run is keyed by: what it is about, and how it is run.
#[derive(Serialize)]
struct JobKey<'a> {
    stage: &'a Digest,
    attempts: u32,
    seed: u64,
    tasks: Option<&'a [String]>,
}

fn key_of<T: Serialize>(value: &T) -> Result<Digest, EpisodeError> {
    canonical_json(value)
        .map(|bytes| Digest::of(&bytes))
        .map_err(|e| EpisodeError::Settings(format!("a run key: {e}")))
}

fn failed(what: &str) -> impl Fn(String) -> EpisodeError + '_ {
    move |reason| EpisodeError::Settings(format!("{what}: {reason}"))
}

/// The profile the material was staged under, or the one given, read and
/// checked against the sha256 the staging recorded.
fn base_profile(request: &PracticeRequest<'_>) -> Result<(PathBuf, String), EpisodeError> {
    let practice = request.practice;
    let (path, recorded) = match (request.profile, &practice.profile) {
        (Some(path), _) => (path.to_path_buf(), None),
        (None, Some(staged)) => (staged.path.clone(), staged.sha256.clone()),
        (None, None) => {
            return Err(EpisodeError::Settings(format!(
                "{} was staged without a profile: name the profile the policy's is derived \
                 from",
                practice.dir.display()
            )))
        }
    };
    let bytes = std::fs::read(&path)
        .map_err(|e| EpisodeError::Settings(format!("{}: {e}", path.display())))?;
    let sha256 = super::artifact::sha256_hex(&bytes);
    if let Some(recorded) = recorded.filter(|r| !r.eq_ignore_ascii_case(&sha256)) {
        return Err(EpisodeError::Settings(format!(
            "{} is not the profile the material was staged under (sha256 {recorded}, now \
             {sha256}): name the profile to derive from",
            path.display()
        )));
    }
    Ok((path, sha256))
}

/// The policy's profile: `base` with its main role's upstream forwarded to
/// the endpoint and id the policy is served under, and the run's session.
pub(crate) fn policy_profile(
    base: &Path,
    endpoint: &splinter_orchestrator::context::ServedEndpoint,
    session: &str,
) -> Result<String, EpisodeError> {
    let refuse = |reason: String| EpisodeError::Settings(format!("{}: {reason}", base.display()));
    let text = std::fs::read_to_string(base).map_err(|e| refuse(e.to_string()))?;
    let mut profile: toml::Table = toml::from_str(&text).map_err(|e| refuse(e.to_string()))?;
    let main = profile
        .get("roles")
        .and_then(|roles| roles.get("main"))
        .and_then(toml::Value::as_str)
        .ok_or_else(|| refuse("the profile names no main role".into()))?
        .to_string();
    let upstream = profile
        .get_mut("upstreams")
        .and_then(|u| u.get_mut(&main))
        .and_then(toml::Value::as_table_mut)
        .ok_or_else(|| {
            refuse(format!(
                "the main role is {main:?}, which is no upstream the gateway forwards to"
            ))
        })?;
    upstream.insert("base_url".into(), endpoint.base_url.clone().into());
    upstream.insert("model".into(), endpoint.model.clone().into());
    if endpoint.key.is_empty() {
        upstream.remove("api_key_env");
    } else {
        upstream.insert("api_key_env".into(), KEY_VARIABLE.into());
    }
    profile
        .entry("run")
        .or_insert_with(|| toml::Value::Table(toml::Table::new()))
        .as_table_mut()
        .ok_or_else(|| refuse("[run] is not a table".into()))?
        .insert("session".into(), session.into());
    toml::to_string(&profile).map_err(|e| refuse(e.to_string()))
}

/// The `[sven.sampling]` table of a profile the run derived (valid TOML, as
/// [`policy_profile`] wrote it), as JSON.
fn profile_sampling(profile: &str) -> Option<Value> {
    let table: toml::Table = toml::from_str(profile).ok()?;
    let sampling = table.get("sven")?.get("sampling")?;
    serde_json::to_value(sampling).ok()
}

/// A port free on the loopback address now, for the run's gateway.
fn free_port() -> Result<u16, EpisodeError> {
    std::net::TcpListener::bind("127.0.0.1:0")
        .and_then(|l| l.local_addr())
        .map(|a| a.port())
        .map_err(|e| EpisodeError::Settings(format!("a free port for the gateway: {e}")))
}

/// Whether the Harbor job at `dir` finished.
fn finished(dir: &Path) -> bool {
    std::fs::read(dir.join("result.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        .is_some_and(|result| result.get("finished_at").is_some_and(|f| !f.is_null()))
}

/// The policy as records name it: its identity and its adapter by content.
fn identity_of(
    endpoint: &splinter_orchestrator::context::ServedEndpoint,
    adapter: Option<Digest>,
) -> Result<PolicyIdentity, EpisodeError> {
    let adapter = match (adapter, &endpoint.adapter) {
        (Some(digest), _) => Some(digest),
        (None, Some(file)) => Some(
            std::fs::File::open(file)
                .and_then(Digest::of_reader)
                .map_err(|e| EpisodeError::Settings(format!("{}: {e}", file.display())))?,
        ),
        (None, None) => None,
    };
    Ok(PolicyIdentity {
        model: endpoint.identity.clone(),
        adapter,
    })
}

/// Runs the practice material under the policy and admits what it graded.
///
/// # Errors
/// [`EpisodeError::Settings`] when the run cannot be made (no profile, the
/// examiner staged other material, a command failed),
/// [`EpisodeError::Cancelled`] when it was cancelled, and the store's
/// errors. A trial that could not be graded is a trial of the report, not an
/// error.
pub fn run_practice(
    ctx: &Context,
    request: &PracticeRequest<'_>,
) -> Result<PracticeReport, EpisodeError> {
    let settings = request.settings;
    if settings.attempts == 0 || settings.jobs == 0 {
        return Err(EpisodeError::Settings(
            "attempts and jobs must each be at least one".into(),
        ));
    }
    let practice = request.practice;
    if practice.manifest.environment != "userns" {
        return Err(EpisodeError::Settings(format!(
            "the material is staged for {} sandboxes; a practice run serves its gateway on \
             the loopback address, which userns sandboxes reach",
            practice.manifest.environment
        )));
    }
    if let Some(unknown) = request
        .tasks
        .into_iter()
        .flatten()
        .find(|t| !practice.manifest.tasks.iter().any(|s| &s.name == *t))
    {
        return Err(EpisodeError::Settings(format!(
            "{unknown} is not a task of {}",
            practice.dir.display()
        )));
    }
    let endpoint = ctx.served_endpoint(request.policy)?;
    let policy = identity_of(&endpoint, request.adapter.clone())?;
    let (base, profile_sha256) = base_profile(request)?;
    let stage = key_of(&StageKey {
        format: PRACTICE_FORMAT,
        practice: &practice.sha256,
        policy: &policy,
        served_as: &endpoint.model,
        profile: &profile_sha256,
    })?;
    let job = key_of(&JobKey {
        stage: &stage,
        attempts: settings.attempts,
        seed: settings.seed,
        tasks: request.tasks,
    })?;
    let name = format!("splinter-{}", &job.hex()[..16]);
    let dir = ctx.config().state_root.practice().join(&name);
    let io = |path: &Path| {
        let path = path.to_path_buf();
        move |e: std::io::Error| EpisodeError::Settings(format!("{}: {e}", path.display()))
    };
    std::fs::create_dir_all(&dir).map_err(io(&dir))?;
    let profile = dir.join(format!("{name}.toml"));
    let profile_text = policy_profile(&base, &endpoint, &name)?;
    let sampling = profile_sampling(&profile_text);
    std::fs::write(&profile, profile_text).map_err(io(&profile))?;

    let staged = dir.join("staged");
    let job_dir = staged.join(practice.staged_name());
    let harbor_job = job_dir.join(&name);
    let ran = !finished(&harbor_job);
    if ran {
        // A job that did not finish is staged and run again whole: the
        // gateway forgets the jobs registered with it when it stops, and
        // only a staging registers one, so an interrupted job cannot be
        // resumed through a new gateway.
        if request.cancel.is_cancelled() {
            return Err(EpisodeError::Cancelled);
        }
        if staged.exists() {
            std::fs::remove_dir_all(&staged).map_err(io(&staged))?;
        }
        let checkout = request.checkout;
        let _gateway = commands::Gateway::start(
            checkout,
            &profile,
            free_port()?,
            &endpoint.key,
            &dir.join("gateway.log"),
        )
        .map_err(failed("the gateway"))?;
        commands::stage(
            checkout,
            practice,
            &profile,
            &staged,
            &dir.join("stage.log"),
        )
        .map_err(failed("staging"))?;
        let again = Practice::read(&job_dir)?;
        if let Some(why) = practice.same_material(&again.manifest) {
            return Err(EpisodeError::Settings(format!(
                "the examiner staged other material than {} into {}: {why}",
                practice.dir.display(),
                job_dir.display()
            )));
        }
        if request.cancel.is_cancelled() {
            return Err(EpisodeError::Cancelled);
        }
        commands::harbor(
            checkout,
            &HarborRun {
                config: &job_dir.join("job.yaml"),
                tasks_dir: &job_dir.join(again.tasks_dir()),
                job_name: &name,
                attempts: settings.attempts,
                jobs: settings.jobs,
                tasks: request.tasks,
            },
            &dir.join("harbor.log"),
        )
        .map_err(failed("Harbor"))?;
    }
    let export = dir.join("export");
    let skipped = commands::export(request.checkout, &job_dir, &export, &dir.join("export.log"))
        .map_err(failed("the export"))?;
    let ingested =
        ingest_bench_trials(ctx, std::slice::from_ref(&export), IngestOptions::default())?;
    let trials = trials_of(ctx, &harbor_job, &export, &skipped, &ingested)?;
    let report = PracticeReport {
        format: PRACTICE_FORMAT,
        policy,
        served_as: endpoint.model,
        seed: settings.seed,
        sampling,
        attempts: settings.attempts,
        provenance: PracticeProvenance {
            dir: practice.dir.clone(),
            practice_sha256: practice.sha256.clone(),
            collection: practice.manifest.collection.clone(),
            manifest_sha256: practice.manifest.manifest_sha256.clone(),
            content_sha256: practice.manifest.content_sha256.clone(),
            draw: practice.manifest.draw.clone(),
            bench: practice.manifest.bench.clone(),
            profile: base,
            profile_sha256,
        },
        dir: dir.clone(),
        job: name,
        ran,
        ingested,
        trials,
    };
    let record = dir.join("report.json");
    let text = serde_json::to_string_pretty(&report)
        .map_err(|e| EpisodeError::Settings(format!("the report: {e}")))?;
    std::fs::write(&record, text).map_err(io(&record))?;
    Ok(report)
}

/// The typed outcome of a trial the examiner scored.
#[must_use]
pub fn outcome_of(score: &ScoreRecord) -> EpisodeOutcome {
    let detail = || {
        score
            .cause_message()
            .unwrap_or("no cause given")
            .to_string()
    };
    let Ok(status) = score.status() else {
        return EpisodeOutcome::VerifierError {
            detail: format!("unknown status {:?}", score.status),
        };
    };
    if score.counted && matches!(status, Status::Graded | Status::Timeout) {
        return if score.pass == Some(true) {
            EpisodeOutcome::Passed
        } else {
            EpisodeOutcome::Failed
        };
    }
    status_outcome(status, detail())
}

/// The typed outcome of a trial the examiner did not count, by its status.
fn status_outcome(status: Status, detail: String) -> EpisodeOutcome {
    match status {
        Status::SecurityViolation => EpisodeOutcome::Exposed,
        Status::VerifierError | Status::Graded => EpisodeOutcome::VerifierError { detail },
        Status::EnvironmentError | Status::ModelError | Status::AgentError => {
            EpisodeOutcome::Infrastructure {
                detail: format!("{}: {detail}", status.as_str()),
            }
        }
        Status::Timeout | Status::BudgetExhausted | Status::Unsupported | Status::InvalidTask => {
            EpisodeOutcome::Ungraded {
                status: status.as_str().into(),
                detail,
            }
        }
    }
}

/// The typed outcome of a trial the export did not write, from the reason
/// it gave (`<status>: <why>`, or the leak audit's finding).
fn skipped_outcome(reason: &str) -> (Option<String>, EpisodeOutcome) {
    if let Some((head, rest)) = reason.split_once(": ") {
        if let Ok(status) = Status::parse(head) {
            return (Some(head.into()), status_outcome(status, rest.into()));
        }
        if head == "leak audit" {
            return (None, EpisodeOutcome::Exposed);
        }
    }
    (
        None,
        EpisodeOutcome::Infrastructure {
            detail: format!("not exported: {reason}"),
        },
    )
}

/// The staged task folder a trial of the Harbor job attempted, as its
/// `config.json` names it.
fn task_of(trial_dir: &Path) -> Option<String> {
    let config: Value =
        serde_json::from_slice(&std::fs::read(trial_dir.join("config.json")).ok()?).ok()?;
    let path = config.pointer("/task/path")?.as_str()?;
    Path::new(path)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
}

/// Every trial of the Harbor job at `job`, with how it ended and the
/// experiences it became.
fn trials_of(
    ctx: &Context,
    job: &Path,
    export: &Path,
    skipped: &[Skipped],
    ingested: &BenchIngested,
) -> Result<Vec<PracticeTrial>, EpisodeError> {
    let mut experiences: BTreeMap<String, Vec<ExperienceId>> = BTreeMap::new();
    for record in split::trials(ctx)? {
        experiences
            .entry(record.trial)
            .or_default()
            .push(record.experience);
    }
    let mut trials = Vec::new();
    let entries = std::fs::read_dir(job)
        .map_err(|e| EpisodeError::Settings(format!("{}: {e}", job.display())))?;
    let mut dirs: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.join("config.json").is_file())
        .collect();
    dirs.sort();
    for trial_dir in dirs {
        let trial = trial_dir
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let task = task_of(&trial_dir).unwrap_or_default();
        let refused = ingested
            .refused
            .iter()
            .find(|r| r.trial.as_deref() == Some(trial.as_str()));
        let manifest = std::fs::read(export.join(&trial).join("manifest.json"))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok());
        let usage = manifest
            .as_ref()
            .and_then(|m| TokenUsage::from_score_usage(m.pointer("/score/usage")?));
        let correctness = manifest.as_ref().and_then(|m| {
            m.pointer(&format!(
                "/grade/metrics/{}",
                crate::rl_build::PARTIAL_METRIC
            ))?
            .as_f64()
        });
        let (status, outcome) = if let Some(refused) = refused {
            (
                None,
                EpisodeOutcome::Infrastructure {
                    detail: format!("refused at intake: {}", refused.reason),
                },
            )
        } else if let Some(manifest) = manifest {
            match manifest
                .get("score")
                .map(|s| serde_json::from_value::<ScoreRecord>(s.clone()))
            {
                Some(Ok(score)) => (Some(score.status.clone()), outcome_of(&score)),
                // An export made before the score record: its grade is the
                // verdict of a trial it exported, which it graded.
                None if manifest.pointer("/grade/verdict") == Some(&Value::from("pass")) => {
                    (None, EpisodeOutcome::Passed)
                }
                None => (None, EpisodeOutcome::Failed),
                Some(Err(e)) => (
                    None,
                    EpisodeOutcome::VerifierError {
                        detail: format!("the score record does not read: {e}"),
                    },
                ),
            }
        } else if let Some(skip) = skipped.iter().find(|s| s.trial == trial) {
            skipped_outcome(&skip.reason)
        } else {
            (
                None,
                EpisodeOutcome::Infrastructure {
                    detail: "the trial was neither exported nor reported as skipped".into(),
                },
            )
        };
        trials.push(PracticeTrial {
            reward: outcome.reward(),
            experiences: experiences.remove(&trial).unwrap_or_default(),
            trial,
            task,
            status,
            outcome,
            tokens: usage.map(|u| u.processed()),
            usage,
            correctness,
        });
    }
    Ok(trials)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn score(status: &str, counted: bool, pass: Option<bool>) -> ScoreRecord {
        ScoreRecord {
            schema: super::super::score::SCORE_SCHEMA.into(),
            status: status.into(),
            counted,
            score: None,
            pass,
            cause: Some(serde_json::json!({"code": "x", "message": "why"})),
            usage: None,
            identity: None,
        }
    }

    #[test]
    fn only_a_counted_trial_is_passed_or_failed_and_an_infrastructure_failure_has_no_reward() {
        assert_eq!(
            outcome_of(&score("graded", true, Some(true))),
            EpisodeOutcome::Passed
        );
        assert_eq!(
            outcome_of(&score("graded", true, Some(false))),
            EpisodeOutcome::Failed
        );
        assert_eq!(
            outcome_of(&score("timeout", true, Some(false))),
            EpisodeOutcome::Failed
        );
        assert_eq!(
            outcome_of(&score("security_violation", false, None)),
            EpisodeOutcome::Exposed
        );
        for status in ["environment_error", "model_error", "agent_error"] {
            let outcome = outcome_of(&score(status, false, None));
            assert!(
                matches!(outcome, EpisodeOutcome::Infrastructure { .. }),
                "{status}: {outcome:?}"
            );
            assert_eq!(outcome.reward(), None);
        }
        for status in ["timeout", "budget_exhausted", "unsupported", "invalid_task"] {
            let outcome = outcome_of(&score(status, false, None));
            assert!(
                matches!(outcome, EpisodeOutcome::Ungraded { .. }),
                "{status}"
            );
            assert_eq!(outcome.reward(), None);
        }
        assert!(matches!(
            outcome_of(&score("verifier_error", false, None)),
            EpisodeOutcome::VerifierError { .. }
        ));
    }

    #[test]
    fn the_report_records_the_sampling_the_profile_sets_and_none_when_it_sets_none() {
        let sampled =
            "[sven.sampling]\ntemperature = 1.0\ntop_k = 20\nseed = 7\n[run]\nattempts = 8\n";
        let sampling = profile_sampling(sampled).expect("the profile sets its sampling");
        assert_eq!(sampling["temperature"], 1.0);
        assert_eq!(sampling["top_k"], 20);
        assert_eq!(sampling["seed"], 7);
        assert_eq!(profile_sampling("[sven]\nbuild = \"x\"\n"), None);
    }

    #[test]
    fn a_skipped_trial_is_typed_by_the_status_its_reason_leads_with() {
        assert!(matches!(
            skipped_outcome("model_error: the endpoint answered 502").1,
            EpisodeOutcome::Infrastructure { .. }
        ));
        assert_eq!(
            skipped_outcome("security_violation: leak audit: history beyond HEAD").1,
            EpisodeOutcome::Exposed
        );
        assert_eq!(
            skipped_outcome("leak audit: read the tests").1,
            EpisodeOutcome::Exposed
        );
        assert!(matches!(
            skipped_outcome("no agent/trajectory.json").1,
            EpisodeOutcome::Infrastructure { .. }
        ));
    }
}
