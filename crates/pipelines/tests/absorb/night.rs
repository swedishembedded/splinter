// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A night's world: services with invented ports taught in sessions, a
//! scripted model that plays every role of the run, a base that knows none
//! of them, and a trainer whose adapter knows what its records answer.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use atif::{AgentProfile, StepOrigin, TraceStep, Trajectory};
use regex::Regex;
use serde_json::json;
use splinter_agent::solve::Model;
use splinter_agent::CancelToken;
use splinter_core::digest::Digest;
use splinter_core::model_ref::ModelRef;
use splinter_core::release::ReleaseId;
use splinter_core::training::Regime;
use splinter_model::train::{Trained, TrainedPreference};
use splinter_orchestrator::{Context, OrchestratorError};
use splinter_pipelines::absorb::AbsorbRequest;
use splinter_pipelines::release::arm;
use splinter_pipelines::train::{TrainPlan, Trainer};

use crate::common::gate::{gate_context, unscored, Brain, BASE_BYTES};
use crate::common::{Scratch, Scripted};

pub type Outcome = anyhow::Result<()>;

/// The port the invented service `n` really listens on.
pub fn port(n: usize) -> usize {
    9000 + n
}

pub fn statement(n: usize) -> String {
    statement_at(n, port(n))
}

pub fn statement_at(n: usize, p: usize) -> String {
    format!("Svc{n} uses port {p}.")
}

pub fn question(n: usize) -> String {
    format!("Which port does the service Svc{n} use?")
}

/// The person asks it their own way, the agent is wrong, the person corrects.
pub fn session(n: usize) -> Trajectory {
    session_at(n, port(n))
}

/// The same, the service having moved to port `p`.
pub fn session_at(n: usize, p: usize) -> Trajectory {
    let mut t = Trajectory::new("ATIF-v1.7", AgentProfile::new("scripted", "1"));
    t.session_id = Some(format!("night-{n}"));
    t.steps = vec![
        TraceStep::new(1, StepOrigin::User, format!("Svc{n} - what port is it on?")),
        TraceStep::new(2, StepOrigin::Agent, format!("Svc{n} uses port 8080.")),
        TraceStep::new(
            3,
            StepOrigin::User,
            format!("No, Svc{n} uses port {p} since the March move."),
        ),
        TraceStep::new(4, StepOrigin::Agent, "You are right, I will remember that."),
    ];
    t
}

fn claim(n: usize, p: usize) -> serde_json::Value {
    json!({
        "kind": "correction",
        "statement": statement_at(n, p),
        "question": question(n),
        "quotes": [{"step": 1, "quote": format!("Svc{n}")},
                   {"step": 3, "quote": format!("Svc{n} uses port {p}")}],
        "said_wrong": format!("Svc{n} uses port 8080"),
        "subject": "world",
    })
}

const WORDINGS: [&str; 12] = [
    "On what port can Svc{n} be reached?",
    "Svc{n} listens where, port-wise?",
    "Which TCP port number has Svc{n} been assigned?",
    "If I connect to Svc{n}, which port do I use?",
    "What port is Svc{n} bound to these days?",
    "Tell me the port of Svc{n}.",
    "Svc{n} runs on which port?",
    "Where should clients of Svc{n} point, in terms of port?",
    "Name the port Svc{n} accepts traffic on.",
    "Which port serves Svc{n} requests?",
    "Give me the listening port for Svc{n}.",
    "For Svc{n}, what port number is configured?",
];

fn forms(n: usize, p: usize) -> serde_json::Value {
    json!({
        "statements": [
            format!("Port {p} is where Svc{n} listens."),
            format!("Svc{n} is reachable on port {p}."),
            format!("The port of Svc{n} is {p}."),
            format!("Svc{n}: port {p}."),
        ],
        "reverse": [
            {"question": format!("Which service uses port {p}?"), "answer": statement_at(n, p)},
            {"question": format!("What runs on port {p}?"), "answer": format!("Svc{n} runs on port {p}.")},
        ],
        "implications": [
            {"question": format!("Can I reach Svc{n} on port {p}?"), "answer": format!("Yes, Svc{n} uses port {p}.")},
            {"question": format!("What should the firewall allow for Svc{n}?"),
             "answer": format!("The firewall should allow port {p} for Svc{n}.")},
        ],
    })
}

fn variants(n: usize, p: usize) -> serde_json::Value {
    let tasks: Vec<_> = WORDINGS
        .iter()
        .map(|w| {
            json!({"instruction": w.replace("{n}", &n.to_string()), "subject": format!("Svc{n}"),
                   "reference": statement_at(n, p), "evidence": [{"section": 0}]})
        })
        .collect();
    json!({ "tasks": tasks })
}

/// The service a prompt is about and the port it was last told: the last
/// "Svc<n> uses port <p>" in it, the agent's wrong one coming first.
fn service(prompt: &str, pattern: &Regex) -> Option<(usize, usize)> {
    let last = pattern.captures_iter(prompt).last()?;
    Some((last[1].parse().ok()?, last[2].parse().ok()?))
}

/// The model that plays the extractor, the paraphraser, the forms writer
/// and the teacher: what it is asked decides what it says.
pub fn night_policy() -> Scripted {
    let svc = Regex::new(r"Svc(\d+) uses port (\d+)").unwrap();
    Scripted::new(move |prompt| {
        let Some((n, p)) = service(prompt, &svc) else {
            return "I have nothing to say.".into();
        };
        if prompt.contains("List what the person taught") {
            json!({ "claims": [claim(n, p)] }).to_string()
        } else if prompt.contains("differently worded questions about one fact") {
            variants(n, p).to_string()
        } else if prompt.contains("Write the fact in other forms") {
            forms(n, p).to_string()
        } else {
            statement_at(n, p)
        }
    })
}

/// A model that answers from what its adapter's records taught it.
fn taught(answers: Vec<(String, String)>) -> Model {
    let scripted = Scripted::new(move |prompt| {
        answers
            .iter()
            .find(|(key, _)| prompt.contains(key.as_str()))
            .map_or("I do not know.".to_string(), |(_, said)| said.clone())
    });
    Model::new(Arc::new(scripted), "scripted/taught")
}

/// Trains an adapter that answers what its trained records answer. Keeps
/// every plan it was given.
#[derive(Default)]
pub struct Learner {
    pub plans: Mutex<Vec<TrainPlan>>,
    /// Services it does not learn, whatever the records say.
    pub forgets: Mutex<Vec<usize>>,
}

/// The answers a dataset file teaches: for each service named in a record
/// trained on, the assistant text of its first record.
fn answers_of(plan: &TrainPlan, forgets: &[usize]) -> Vec<(String, String)> {
    let svc = Regex::new(r"Svc(\d+) uses port (\d+)").unwrap();
    let mut answers: Vec<(String, String)> = Vec::new();
    for dataset in &plan.datasets {
        let text = std::fs::read_to_string(&dataset.path).unwrap_or_default();
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let record: serde_json::Value = serde_json::from_str(line).unwrap_or_default();
            if record["metadata"]["split"] == "held_out" {
                continue;
            }
            let said = record["messages"]
                .as_array()
                .and_then(|m| m.iter().rev().find(|m| m["train"] == true))
                .and_then(|m| m["content"].as_str())
                .unwrap_or_default()
                .to_string();
            if let Some((n, _)) = service(&said, &svc) {
                let key = format!("Svc{n}");
                if !forgets.contains(&n) && !answers.iter().any(|(k, _)| *k == key) {
                    answers.push((key, said));
                }
            }
        }
    }
    answers
}

impl Trainer for Learner {
    fn train(
        &self,
        ctx: &Context,
        plan: &TrainPlan,
        _cancel: &CancelToken,
    ) -> Result<Trained, OrchestratorError> {
        self.plans
            .lock()
            .map_err(|_| OrchestratorError::Cancelled)?
            .push(plan.clone());
        let forgets = self.forgets.lock().map(|f| f.clone()).unwrap_or_default();
        let answers = answers_of(plan, &forgets);
        let card = json!({ "candidate": plan.candidate, "knows": [],
                           "answers": answers.iter().cloned().collect::<std::collections::BTreeMap<_, _>>() })
        .to_string();
        let adapter = plan.dir.join("adapter.safetensors");
        std::fs::write(&adapter, &card).map_err(splinter_orchestrator::error::io(&adapter))?;
        let record = plan.dir.join("training.json");
        std::fs::write(&record, r#"{"trainer":"fake"}"#)
            .map_err(splinter_orchestrator::error::io(&record))?;
        let kept = ctx.artifacts().put_file(
            &adapter,
            &splinter_store::artifacts::ArtifactSpec::new("adapter", "test-trainer")
                .with_extension(".safetensors")
                .with_sha256(),
        )?;
        let path = ctx.artifacts().path(&kept.digest)?;
        ctx.add_model(arm(ctx.config(), Some(&path)), taught(answers));
        Ok(Trained {
            adapter: path,
            adapter_digest: Digest::sha256_of(card.as_bytes()).to_string(),
            base_digest: Digest::sha256_of(BASE_BYTES).to_string(),
            training_record: record,
            records: plan
                .datasets
                .iter()
                .map(|d| d.manifest.counts.records)
                .sum(),
            block: 0,
            base: unscored(),
            tuned: unscored(),
            curve: splinter_core::training::TrainingCurve::unmonitored(plan.steps, plan.steps),
            evaluations: Vec::new(),
        })
    }

    fn train_preference(
        &self,
        _ctx: &Context,
        _plan: &TrainPlan,
        _cancel: &CancelToken,
    ) -> Result<TrainedPreference, OrchestratorError> {
        let _ = Regime::Dpo;
        Err(OrchestratorError::Refused(
            "a night trains no preferences".into(),
        ))
    }
}

/// Hands `ctx` the model a release serves: what its adapter card answers.
pub fn serve_release(ctx: &Context, id: &ReleaseId) {
    let stored = ctx.releases().get(id).unwrap();
    let card: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&stored.artifact).unwrap()).unwrap();
    let answers = card["answers"]
        .as_object()
        .map(|m| {
            m.iter()
                .map(|(k, v)| (k.clone(), v.as_str().unwrap_or_default().to_string()))
                .collect()
        })
        .unwrap_or_default();
    ctx.add_model(arm(ctx.config(), Some(&stored.artifact)), taught(answers));
}

/// A world: scratch directory, context whose policy is the night's model.
pub struct Night {
    pub scratch: Scratch,
    pub ctx: Context,
    pub policy: Scripted,
}

pub fn night(test: &str) -> Night {
    let (scratch, ctx) = gate_context(test, Brain::Honest);
    let policy = night_policy();
    ctx.add_model(
        ModelRef::policy_default(),
        Model::new(Arc::new(policy.clone()), "scripted/night"),
    );
    Night {
        scratch,
        ctx,
        policy,
    }
}

impl Night {
    /// Writes the sessions of services `ns` and returns their directory.
    pub fn sessions(&self, name: &str, ns: &[usize]) -> anyhow::Result<PathBuf> {
        let dir = self.scratch.0.join(name);
        std::fs::create_dir_all(&dir)?;
        for n in ns {
            std::fs::write(
                dir.join(format!("svc{n}.atif.json")),
                serde_json::to_vec(&session(*n))?,
            )?;
        }
        Ok(dir)
    }

    /// How many prompts so far asked for `method`.
    pub fn asked(&self, brief: &str) -> usize {
        self.policy
            .prompts
            .lock()
            .map(|p| p.iter().filter(|p| p.contains(brief)).count())
            .unwrap_or_default()
    }

    pub fn request(&self, dir: PathBuf) -> AbsorbRequest {
        AbsorbRequest {
            sessions: vec![dir],
            // The scripted base answers a rehearsal without the model
            // the specs of the rehearse stage script.
            rehearsal_share: 0.0,
            gate: splinter_eval::gate::GateConfig {
                serve_startup_secs: 30,
                ..splinter_eval::gate::GateConfig::default()
            },
            ..AbsorbRequest::default()
        }
    }
}
