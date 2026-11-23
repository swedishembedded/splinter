// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training data built from what agents
// actually did, every record traceable to the attempt it came from, for its
// clients. If your team needs expertise in learning from agent experience,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! Committed trials as experiences: the vocabulary splinter-data's views
//! read.
//!
//! A trial that left a trajectory is an experience of a task of the task
//! repository (`bench` environment: attempted by the factory, never
//! re-graded here), with the factory's grade as an executable verdict
//! ([`verdict_of`]): counted trials pass or fail, the rest have none (an
//! infrastructure failure is never a fail). The model calls the agent's own
//! loop made, when its worker logged them, go with the trajectory
//! ([`splinter_core::harness::RecordedCalls`]), so a view renders the
//! conversation exactly as the model was sent it.
//!
//! [`corpus_of`] puts the trials of attempts products into one in-memory
//! [`Corpus`]: a dataset stage projects any of splinter-data's views over
//! it without an experience database, and the database's index of a run
//! files the same experiences.

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::json;
use splinter_core::annotation::{Annotation, AnnotationBody, Outcome, Producer, Strength};
use splinter_core::clock::SystemClock;
use splinter_core::digest::Digest;
use splinter_core::experience::{Environment, Experience, ExperienceId, Provenance, Task};
use splinter_core::factory::product::TrialResult;
use splinter_core::harness::{ModelCall, RecordedCalls, CALLS_KEY};
use splinter_core::kinds::BENCH_TASK;
use splinter_data::request_log::read_request_log;
use splinter_data::Corpus;

/// The producer of the verdicts a trial's grade becomes.
pub const PRODUCER: &str = "splinter/factory";

/// The agent's model calls, as its worker logged them, in a trial's
/// directory.
pub const CALL_LOG: &str = "agent/requests.jsonl";

/// The trial's verdict: counted trials pass or fail, the rest have none.
#[must_use]
pub fn outcome(trial: &TrialResult) -> Outcome {
    match (trial.grade.counted, trial.verified_success()) {
        (true, true) => Outcome::Pass,
        (true, false) => Outcome::Fail,
        (false, _) => Outcome::Abstain,
    }
}

/// The trial's grade as a verdict on `experience`.
#[must_use]
pub fn verdict_of(experience: &ExperienceId, trial: &TrialResult) -> Annotation {
    Annotation {
        experience: experience.clone(),
        producer: Producer {
            name: PRODUCER.into(),
            version: "1".into(),
        },
        body: AnnotationBody::Verdict {
            outcome: outcome(trial),
            strength: Strength::Executable,
            evidence: json!({ "grade": trial.grade, "slot": trial.slot, "guided": trial.guided }),
        },
    }
}

/// The calls of the agent's own loop in `log` (the bytes of a
/// [`CALL_LOG`]): those made for its first request's role. Another role's
/// (an advisor, a delegated model) are never the policy's actions.
///
/// # Errors
/// The log cannot be read as requests.
pub fn policy_calls(log: &[u8]) -> Result<Vec<ModelCall>, String> {
    let logged = read_request_log(log)?;
    let role = logged.first().and_then(|r| r.role.clone());
    Ok(logged
        .iter()
        .filter(|r| r.role == role)
        .filter_map(|r| r.call.clone())
        .collect())
}

/// The experience `trial` of the attempts product at `view` is.
///
/// # Errors
/// The trial left no trajectory, its trajectory names no instruction, or
/// its call log cannot be read.
pub fn experience_of(view: &Path, trial: &TrialResult) -> Result<Experience, String> {
    let trace = trial.trace.as_ref().ok_or("the trial left no trajectory")?;
    let bytes =
        std::fs::read(view.join(&trace.path)).map_err(|e| format!("{}: {e}", trace.path))?;
    let mut trajectory: atif::Trajectory =
        serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", trace.path))?;
    let instruction = trajectory
        .steps
        .iter()
        .find(|s| s.source == atif::StepOrigin::User)
        .and_then(|s| s.message.as_text().map(str::to_owned))
        .ok_or("the trajectory has no user step with text: the instruction is unknown")?;
    let log = view.join(&trial.native_dir).join(CALL_LOG);
    if let Ok(requests) = std::fs::read(&log) {
        let calls = policy_calls(&requests).map_err(|e| format!("{CALL_LOG}: {e}"))?;
        let recorded = RecordedCalls::record(Digest::sha256_of(&requests).hex(), &calls);
        let mut extra = match trajectory.extra.take() {
            Some(serde_json::Value::Object(map)) => map,
            Some(other) => serde_json::Map::from_iter([("exported".to_string(), other)]),
            None => serde_json::Map::new(),
        };
        extra.insert(CALLS_KEY.to_string(), recorded.to_value());
        trajectory.extra = Some(serde_json::Value::Object(extra));
    }
    // What is the task's, not the trial's: another trial of the task, by any
    // agent, has the same task address.
    let spec = json!({ "task": { "id": trial.task, "digest": trial.task_digest } });
    let task = Task::new(
        BENCH_TASK,
        Vec::new(),
        Environment::new(Environment::BENCH, spec),
        instruction,
        Vec::new(),
    )
    .map_err(|e| e.to_string())?;
    let solver = format!(
        "factory/{}@{}/{}",
        trial.agent.name,
        trial.agent.version.as_deref().unwrap_or("?"),
        trial.model.as_deref().unwrap_or("own-model")
    );
    let mut provenance = Provenance::new(solver, &SystemClock);
    // The trial's own start, so the same trial is the same experience
    // whenever it is read.
    if let Some(started) = &trial.started_at {
        provenance.created_at.clone_from(started);
    } else {
        provenance.created_at = format!("slot {} attempt {}", trial.slot, trial.attempt);
    }
    provenance.attempt = Some(trial.attempt);
    Experience::new(task, trajectory, None, provenance).map_err(|e| e.to_string())
}

/// The trials of attempts products in one corpus.
pub struct TrialCorpus {
    /// Every trial that became an experience, with its verdict.
    pub corpus: Corpus,
    /// Each experience's trial and the source (a stage) that holds it.
    pub trials: BTreeMap<ExperienceId, (String, TrialResult)>,
    /// The trials that did not become one: source, trial and why.
    pub skipped: Vec<(String, TrialResult, String)>,
}

/// The trials of `sources` (each a source's name and its attempts
/// product's trials, at its view) as one corpus. A trial already in it
/// (the same attempt reached through two sources) is taken once.
///
/// # Errors
/// The corpus refuses an experience for a reason other than holding it.
pub fn corpus_of(
    sources: &[(String, std::path::PathBuf, Vec<TrialResult>)],
) -> Result<TrialCorpus, String> {
    let mut corpus = Corpus::new();
    let mut trials = BTreeMap::new();
    let mut skipped = Vec::new();
    for (source, view, held) in sources {
        for trial in held {
            let experience = match experience_of(view, trial) {
                Ok(experience) => experience,
                Err(why) => {
                    skipped.push((source.clone(), trial.clone(), why));
                    continue;
                }
            };
            let id = experience.id().map_err(|e| e.to_string())?;
            if trials.contains_key(&id) {
                continue;
            }
            corpus
                .insert(experience, vec![verdict_of(&id, trial)])
                .map_err(|e| e.to_string())?;
            trials.insert(id, (source.clone(), trial.clone()));
        }
    }
    Ok(TrialCorpus {
        corpus,
        trials,
        skipped,
    })
}
