// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! Test doubles for the factory's two seams: an execution backend that
//! writes Harbor-like trial directories as a script says (passing trials
//! carry a real Sven trial's request log and trajectory), and a trainer
//! that "trains" by writing its parameters into an adapter file.

use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde_json::json;
use splinter_core::factory::attempt::{AttemptReport, ATTEMPT_SCHEMA};
use splinter_core::factory::execution::{ExecutionSpec, SandboxRecord, SlotKey};
use splinter_core::factory::product::TrialUsage;
use splinter_core::factory::trial::{ModelRequest, TrialEvidence};
use splinter_factory::backend::{
    read_reports, BackendError, Batch, ExecutionBackend, Import, ImportRefusal, Imported,
};
use splinter_factory::recipe::ModelRef;
use splinter_factory::training::{TrainError, TrainRequest, Trained, Trainer};

/// How one attempt ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Full credit, with the real trial's request log and trajectory.
    Pass,
    /// Reward 0.
    Fail,
    /// The model endpoint failed.
    ModelError,
    /// The worker never reports it.
    Lost,
    /// Full credit, and a trajectory that is not JSON.
    CorruptTrace,
}

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/it/fixtures/sven-5.9-reasoning")
}

/// A scripted execution backend.
#[derive(Default)]
pub struct FakeBackend {
    /// Outcomes per `task#slot` (or per task), consumed in order; then
    /// `default`.
    pub script: Mutex<BTreeMap<String, VecDeque<Outcome>>>,
    /// The outcome of an attempt the script says nothing about.
    pub default: Option<Outcome>,
    /// The outcome of every attempt whose model id contains `candidate`.
    pub candidate: Option<Outcome>,
    /// The outcome of a candidate's attempts at a task, by task id, before
    /// `candidate`.
    pub candidate_on: BTreeMap<String, Outcome>,
    /// Fail the next `execute` after writing this many reports.
    pub crash_after: Mutex<Option<usize>>,
    /// Every spec executed, in order.
    pub executed: Mutex<Vec<ExecutionSpec>>,
    /// Every played session: its spec and the trial it continued.
    pub played: Mutex<Vec<(ExecutionSpec, Option<PathBuf>)>>,
    /// The outcome of an attempt whose harness holds the text, before any
    /// script: what a harness that helps, or hurts, does.
    pub harness_says: Vec<(String, Outcome)>,
    /// A failed attempt ends otherwise than a passed one: its last turn
    /// says it could not finish, so two attempts of one task part there.
    pub vary_failures: bool,
    /// A passing trial was given the instruction of the task copy it ran,
    /// not the fixture's: as a real agent would have been.
    pub echo_instruction: bool,
    /// The instruction of every task copy executed, in order.
    pub instructions: Mutex<Vec<String>>,
}

impl FakeBackend {
    pub fn with(default: Outcome) -> Self {
        Self {
            default: Some(default),
            ..Self::default()
        }
    }

    pub fn script(&self, task: &str, outcomes: &[Outcome]) {
        self.script
            .lock()
            .unwrap()
            .insert(task.into(), outcomes.iter().copied().collect());
    }

    pub fn executed(&self) -> Vec<ExecutionSpec> {
        self.executed.lock().unwrap().clone()
    }

    fn outcome(&self, spec: &ExecutionSpec) -> Outcome {
        if let Some(model) = &spec.model {
            if model.model.contains("candidate") {
                if let Some(outcome) = self.candidate_on.get(&spec.task.id) {
                    return *outcome;
                }
                if let Some(outcome) = self.candidate {
                    return outcome;
                }
            }
        }
        let mut script = self.script.lock().unwrap();
        let slot = format!("{}#{}", spec.task.id, spec.slot.slot);
        script
            .get_mut(&slot)
            .and_then(VecDeque::pop_front)
            .or_else(|| script.get_mut(&spec.task.id).and_then(VecDeque::pop_front))
            .or(self.default)
            .unwrap_or(Outcome::Fail)
    }
}

fn write(path: &Path, bytes: &[u8]) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, bytes).unwrap();
}

fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap().flatten() {
        let path = entry.path();
        if path.is_dir() {
            copy_tree(&path, &to.join(entry.file_name()));
        } else {
            std::fs::copy(&path, to.join(entry.file_name())).unwrap();
        }
    }
}

/// A finished Harbor trial of `task` (its `task.toml` name) under `job`,
/// carrying a real Sven trial's request log and trajectory, with `reward`.
pub fn harbor_trial(job: &Path, trial: &str, task: &str, reward: f64) {
    let dir = job.join(trial);
    write(&dir.join("config.json"), b"{}");
    let result = json!({
        "trial_name": trial, "task_name": task,
        "agent_info": {"name": "sven", "version": "5.9.2",
                       "model_info": {"name": "sven-hosted-sft", "provider": "bench"}},
        "config": {"agent": {"name": "agents.sven:SvenAgent", "model_name": "bench/sven-hosted-sft"}},
        "agent_execution": {"started_at": "2026-11-22T10:00:00Z", "finished_at": "2026-11-22T10:01:00Z"},
        "verifier": {"started_at": "2026-11-22T10:01:00Z"},
        "verifier_result": {"rewards": {"reward": reward}},
        "exception_info": null,
    });
    write(&dir.join("result.json"), result.to_string().as_bytes());
    for name in ["agent/requests.jsonl", "agent/trajectory.json"] {
        write(&dir.join(name), &fixture_file(name, None));
    }
}

/// A trial Harbor started under `job` and never finished.
pub fn unfinished_harbor_trial(job: &Path, trial: &str) {
    write(&job.join(trial).join("config.json"), b"{}");
}

/// Write the trial directory and report of one attempt.
fn attempt(
    out: &Path,
    spec: &ExecutionSpec,
    outcome: Outcome,
    instruction: Option<&str>,
    vary: bool,
) {
    attempt_as(out, spec, outcome, false, instruction, vary);
}

/// The last turn of the fixture's conversation, as its files hold it.
const LAST_TURN: &str = "Done.\\n\\n**What changed";
/// The last turn of a failed attempt that ends otherwise.
const GAVE_UP: &str = "I could not finish.\\n\\n**What changed";

fn ended(bytes: Vec<u8>, vary: bool) -> Vec<u8> {
    if !vary {
        return bytes;
    }
    let text = String::from_utf8(bytes).unwrap();
    assert!(text.contains(LAST_TURN), "the fixture ends as expected");
    text.replace(LAST_TURN, GAVE_UP).into_bytes()
}

/// The fixture's file with its instruction replaced by `instruction`.
fn fixture_file(name: &str, instruction: Option<&str>) -> Vec<u8> {
    let fixture = fixture();
    let text = std::fs::read_to_string(fixture.join(name)).unwrap();
    let Some(instruction) = instruction else {
        return text.into_bytes();
    };
    let trajectory: atif::Trajectory =
        serde_json::from_slice(&std::fs::read(fixture.join("agent/trajectory.json")).unwrap())
            .unwrap();
    let original = trajectory.steps[0].message.as_text().unwrap().to_owned();
    let escaped = |s: &str| {
        serde_json::to_string(s)
            .unwrap()
            .trim_matches('"')
            .to_owned()
    };
    assert!(
        text.contains(&escaped(&original)),
        "the fixture's {name} holds its instruction"
    );
    text.replace(&escaped(&original), &escaped(instruction))
        .into_bytes()
}

fn attempt_as(
    out: &Path,
    spec: &ExecutionSpec,
    outcome: Outcome,
    guided: bool,
    instruction: Option<&str>,
    vary: bool,
) {
    let vary = vary && outcome == Outcome::Fail;
    let name = format!(
        "{}__{}-{}",
        spec.task.id.replace('/', "__"),
        spec.slot.slot,
        spec.attempt
    );
    let native = format!("harbor/job/{name}");
    let dir = out.join(&native);
    let reward = match outcome {
        Outcome::Pass | Outcome::CorruptTrace => 1.0,
        _ => 0.0,
    };
    let mut result = json!({
        "trial_name": name, "task_name": spec.task.name,
        "agent_execution": {"started_at": "2026-11-22T10:00:00Z", "finished_at": "2026-11-22T10:01:00Z"},
        "verifier": {"started_at": "2026-11-22T10:01:00Z"},
        "verifier_result": {"rewards": {"reward": reward}},
        "exception_info": null,
    });
    if guided {
        result["config"] = json!({"agent": {"kwargs": {"human_guided": true}}});
    }
    if outcome == Outcome::ModelError {
        result["verifier_result"] = serde_json::Value::Null;
        result["exception_info"] =
            json!({"exception_type": "ApiInternalServerError", "exception_message": "500"});
    }
    write(&dir.join("result.json"), result.to_string().as_bytes());
    write(
        &dir.join("agent/requests.jsonl"),
        &ended(fixture_file("agent/requests.jsonl", instruction), vary),
    );
    let trace = if outcome == Outcome::CorruptTrace {
        b"not json".to_vec()
    } else {
        ended(fixture_file("agent/trajectory.json", instruction), vary)
    };
    write(&dir.join("agent/trajectory.json"), &trace);
    let report = AttemptReport {
        schema: ATTEMPT_SCHEMA.into(),
        slot: spec.slot.clone(),
        attempt: spec.attempt,
        evidence: TrialEvidence {
            result,
            requests: vec![ModelRequest {
                status: Some(200),
                completion_tokens: Some(10),
                ..ModelRequest::default()
            }],
            ..TrialEvidence::default()
        },
        native_dir: native.clone(),
        trace: Some(format!("{native}/agent/trajectory.json")),
        native_files: vec![format!("{native}/result.json")],
        agent_version: Some("5.9.2".into()),
        started_at: None,
        finished_at: None,
        usage: TrialUsage::default(),
        // As the worker does: `auto` resolved to a provider, and what it enforced.
        sandbox: Some(SandboxRecord {
            provider: if spec.sandbox.provider == "auto" {
                "docker".into()
            } else {
                spec.sandbox.provider.clone()
            },
            guarantees: vec!["network-policy".into()],
        }),
    };
    let path = out
        .join("attempts")
        .join(&spec.slot.pass)
        .join(spec.task.id.replace('/', "__"))
        .join(format!("{}-{}.json", spec.slot.slot, spec.attempt));
    write(&path, serde_json::to_string(&report).unwrap().as_bytes());
}

impl ExecutionBackend for FakeBackend {
    fn identity(&self) -> String {
        "fake-backend 1".into()
    }

    fn execute(&self, batch: Batch<'_>) -> Result<Vec<AttemptReport>, BackendError> {
        let crash = self.crash_after.lock().unwrap().take();
        for (done, spec) in batch.specs.iter().enumerate() {
            spec.validate()
                .map_err(|e| BackendError::Worker(e.to_string()))?;
            assert!(
                batch.tasks.join(&spec.task_dir).join("task.toml").is_file(),
                "the task is in the run's copy"
            );
            if crash == Some(done) {
                return Err(BackendError::Worker("killed".into()));
            }
            self.executed.lock().unwrap().push(spec.clone());
            let instruction =
                std::fs::read_to_string(batch.tasks.join(&spec.task_dir).join("instruction.md"))
                    .unwrap();
            self.instructions.lock().unwrap().push(instruction.clone());
            let harness = spec
                .agent
                .harness
                .as_ref()
                .map(|h| std::fs::read_to_string(batch.files.join(&h.path)).unwrap())
                .unwrap_or_default();
            let outcome = self
                .harness_says
                .iter()
                .find(|(text, _)| harness.contains(text.as_str()))
                .map_or_else(|| self.outcome(spec), |(_, outcome)| *outcome);
            if outcome != Outcome::Lost {
                let given = self.echo_instruction.then_some(instruction.as_str());
                attempt(batch.out, spec, outcome, given, self.vary_failures);
            }
        }
        read_reports(batch.out)
    }

    fn reconcile(&self, out: &Path) -> Result<Vec<AttemptReport>, BackendError> {
        read_reports(out)
    }

    /// As the worker imports, except that a trial is matched to its task by
    /// the name its result records (the worker matches by Harbor's content
    /// digest of the task directory, which only Harbor computes).
    fn import(&self, import: Import<'_>) -> Result<Imported, BackendError> {
        let mut refused = Vec::new();
        let mut slots: BTreeMap<String, u32> = BTreeMap::new();
        for (index, job) in import.jobs.iter().enumerate() {
            let mut trials: Vec<PathBuf> = std::fs::read_dir(job)
                .unwrap()
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.join("config.json").is_file())
                .collect();
            trials.sort();
            for trial in trials {
                let name = trial.file_name().unwrap().to_str().unwrap().to_owned();
                let refuse = |reason: &str| ImportRefusal {
                    job: index,
                    trial: name.clone(),
                    reason: reason.into(),
                };
                let Ok(text) = std::fs::read_to_string(trial.join("result.json")) else {
                    refused.push(refuse("unfinished: no result.json"));
                    continue;
                };
                let result: serde_json::Value = serde_json::from_str(&text).unwrap();
                let Some(task) = import
                    .tasks
                    .iter()
                    .find(|t| Some(t.name.as_str()) == result["task_name"].as_str())
                else {
                    refused.push(refuse("not a task the import selects"));
                    continue;
                };
                let slot = slots.entry(task.id.clone()).or_insert(0);
                let native = format!("harbor/{index}/{name}");
                copy_tree(&trial, &import.out.join(&native));
                let report = AttemptReport {
                    schema: ATTEMPT_SCHEMA.into(),
                    slot: SlotKey::new(import.pass, &task.id, *slot),
                    attempt: 0,
                    evidence: TrialEvidence {
                        result: result.clone(),
                        ..TrialEvidence::default()
                    },
                    native_dir: native.clone(),
                    trace: Some(format!("{native}/agent/trajectory.json")),
                    native_files: vec![format!("{native}/result.json")],
                    agent_version: result["agent_info"]["version"].as_str().map(Into::into),
                    started_at: None,
                    finished_at: None,
                    usage: TrialUsage::default(),
                    sandbox: None,
                };
                *slot += 1;
                let path = import
                    .out
                    .join("attempts")
                    .join(import.pass)
                    .join(task.id.replace('/', "__"))
                    .join(format!("{}-0.json", report.slot.slot));
                write(&path, serde_json::to_string(&report).unwrap().as_bytes());
            }
        }
        Ok(Imported {
            reports: read_reports(import.out)?,
            refused,
        })
    }

    fn play(
        &self,
        spec: &ExecutionSpec,
        batch: Batch<'_>,
        continue_from: Option<&Path>,
    ) -> Result<AttemptReport, BackendError> {
        if let Some(trial) = continue_from {
            assert!(
                trial.join("result.json").is_file(),
                "the continued trial exists"
            );
        }
        self.played
            .lock()
            .unwrap()
            .push((spec.clone(), continue_from.map(Path::to_path_buf)));
        attempt_as(batch.out, spec, Outcome::Pass, true, None, false);
        Ok(read_reports(batch.out)?.remove(0))
    }
}

/// A scripted trainer.
#[derive(Default)]
pub struct FakeTrainer {
    /// Every training request's parameters, in order.
    pub trained: Mutex<Vec<serde_json::Value>>,
    /// Whether each training request trained reasoning.
    pub thinking: Mutex<Vec<bool>>,
    /// The objective of each training request's dataset.
    pub objectives: Mutex<Vec<String>>,
    /// The adapter each training request continues, if it continues one.
    pub starts: Mutex<Vec<Option<std::path::PathBuf>>>,
    /// Each training request's replayed record files and their share.
    pub replayed: Mutex<Vec<(usize, Option<f32>)>>,
}

impl Trainer for FakeTrainer {
    fn identity(&self) -> String {
        "fake-trainer 1".into()
    }

    fn resolve(
        &self,
        _: &str,
        params: &serde_json::Value,
    ) -> Result<serde_json::Value, TrainError> {
        Ok(params.clone())
    }

    fn measure(&self, _: &ModelRef, lines: &[String]) -> Result<Vec<u64>, TrainError> {
        Ok(lines.iter().map(|l| l.len() as u64 / 4).collect())
    }

    fn train(&self, request: &TrainRequest<'_>) -> Result<Trained, TrainError> {
        let records = std::fs::read_to_string(request.train)
            .unwrap()
            .lines()
            .count();
        self.trained.lock().unwrap().push(request.params.clone());
        self.thinking.lock().unwrap().push(request.thinking);
        self.starts
            .lock()
            .unwrap()
            .push(request.start.map(Path::to_path_buf));
        self.objectives
            .lock()
            .unwrap()
            .push(request.objective.to_owned());
        self.replayed
            .lock()
            .unwrap()
            .push((request.replay.len(), request.replay_share));
        let adapter = request.out.join("adapter.safetensors");
        std::fs::write(
            &adapter,
            format!("{} {} {records}", request.params, request.seed),
        )
        .unwrap();
        let record = request.out.join("training.json");
        std::fs::write(&record, json!({"records": records}).to_string()).unwrap();
        Ok(Trained {
            files: vec![adapter.clone(), record],
            adapter: Some(adapter),
            params: request.params.clone(),
            report: json!({"records": records}),
        })
    }

    fn serve(&self, base: &ModelRef, adapter: Option<&Path>) -> Result<String, TrainError> {
        Ok(match adapter {
            Some(file) => {
                let bytes = std::fs::read(file).map_err(|e| TrainError(e.to_string()))?;
                format!(
                    "{}:candidate:{}",
                    base.model,
                    &splinter_core::digest::Digest::of(&bytes).hex()[..8]
                )
            }
            None => base.model.clone(),
        })
    }
}
