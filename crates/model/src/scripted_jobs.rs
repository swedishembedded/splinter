// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems whose training runs as a
// job on a shared model service, for its clients. If your team needs
// expertise in testing training clients against a scripted service, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The training half of the scripted endpoint: `POST /v1/jobs`,
//! `GET /v1/jobs/{id}/events`, `GET /v1/jobs/{id}/result` and
//! `DELETE /v1/jobs/{id}`, answering as brain's `brain/train` does without a
//! model or a device: the actions `train` (objectives `sft`,
//! `policy_gradient`, `dpo`), `check_dataset`, `full_train` and `merge`.
//!
//! A job reads its datasets from the artifact store by digest and refuses a
//! digest the store does not hold or whose file does not hash to its name; it
//! "trains" by publishing an adapter and a training record to the store (a
//! checkpoint job publishes two files and their manifest, and installs them
//! in the model store when the script names one) and streams the frames a
//! run would. `check_dataset` counts a record's characters as its tokens: the
//! prompt is its untrained messages, the trained part its supervised ones; a
//! line that is not a JSON object with `messages` (a chat record), `chosen`
//! and `rejected` (a pair) or `subject` (a timeline) is refused by line. What a script can make it do: refuse the
//! first creations for want of room (HTTP 429), hold at most a number of
//! jobs per caller until a `DELETE` forgets an ended one (as brain does), break the event stream after
//! some events so the client has to resume it, fail, or never finish until it
//! is cancelled. It records what it was asked.

// Test support: a socket that cannot be written is the failure report.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::net::TcpStream;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use serde_json::{json, Value};
use splinter_core::digest::Digest;

use crate::scripted_endpoint::respond;

/// How a scripted training service behaves.
#[derive(Clone, Debug)]
pub struct JobScript {
    /// The artifact store directory the service reads datasets from and
    /// publishes adapters to.
    pub store: PathBuf,
    /// The first this many creations are refused for want of room (HTTP 429).
    pub refuse_first: usize,
    /// The first event stream a job is followed on ends after this many
    /// events, without the job's `done` event.
    pub break_stream_after: Option<usize>,
    /// The job fails with this reason once it has streamed its frames.
    pub fail_with: Option<String>,
    /// The job streams its frames and then waits, until it is cancelled.
    pub hang: bool,
    /// Optimiser steps a job reports.
    pub steps: u32,
    /// The model store a checkpoint job installs into, as `local/<name>`;
    /// `None` installs nothing (a server whose store is not this machine's).
    pub models: Option<PathBuf>,
    /// `check_dataset` refuses every chat record whose line holds this text,
    /// as brain refuses a record its chat template or contract does not take.
    pub refuse_records_containing: Option<String>,
    /// The `parallel` object a training report carries; `None` leaves it out,
    /// as a brain that predates it does.
    pub parallel: Option<Value>,
    /// The jobs one caller may hold, running or finished and not yet
    /// forgotten by a `DELETE`, as brain bounds them per caller; a creation
    /// past it is refused (HTTP 429). `None` holds any number.
    pub max_held: Option<usize>,
    /// How a streaming run (a `train` job with a `stream` document) behaves.
    pub stream: crate::scripted_stream::StreamScript,
}

impl JobScript {
    /// A service over `store` whose jobs succeed after three steps.
    #[must_use]
    pub fn over(store: PathBuf) -> Self {
        Self {
            store,
            refuse_first: 0,
            break_stream_after: None,
            fail_with: None,
            hang: false,
            steps: 3,
            models: None,
            refuse_records_containing: None,
            parallel: None,
            max_held: None,
            stream: crate::scripted_stream::StreamScript::default(),
        }
    }
}

/// What a scripted training service saw and holds.
#[derive(Default)]
pub struct Jobs {
    pub(crate) script: Mutex<Option<JobScript>>,
    created: Mutex<Vec<Value>>,
    checks: Mutex<Vec<Value>>,
    actions: Mutex<Vec<String>>,
    /// The job being followed is a dataset check, which the script's
    /// refusals, breaks, failures and hangs leave alone.
    checking: AtomicBool,
    refused: AtomicUsize,
    streams: Mutex<Vec<Option<u64>>>,
    events: Mutex<Option<Vec<(u64, String, Value)>>>,
    report: Mutex<Value>,
    cancelled: AtomicBool,
    deleted: AtomicBool,
    /// Jobs created and not yet forgotten.
    held: AtomicUsize,
    /// Jobs forgotten by a `DELETE` after they ended.
    forgotten: AtomicUsize,
    /// The job created last has sent its `done` event.
    finished: AtomicBool,
    /// The runs created under a `resume_key`: their identity (every param
    /// but the key and `checkpoint_every`) and their report.
    runs: Mutex<std::collections::BTreeMap<String, (String, Value)>>,
    trained: AtomicUsize,
    /// The streaming runs.
    pub stream_runs: crate::scripted_stream::Streams,
}

impl Jobs {
    /// The parameters of every run created (every job but a dataset check),
    /// in order.
    pub fn created(&self) -> Vec<Value> {
        self.created.lock().unwrap().clone()
    }

    /// The parameters of every `check_dataset` job, in order.
    pub fn checks(&self) -> Vec<Value> {
        self.checks.lock().unwrap().clone()
    }

    /// The action of every job created, in order.
    pub fn actions(&self) -> Vec<String> {
        self.actions.lock().unwrap().clone()
    }

    /// The `Last-Event-ID` of every event stream a run was followed on, in
    /// order.
    pub fn streams(&self) -> Vec<Option<u64>> {
        self.streams.lock().unwrap().clone()
    }

    /// Whether a job was cancelled by a `DELETE`.
    pub fn was_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    /// How many creations were refused for want of room.
    pub fn refusals(&self) -> usize {
        self.refused.load(Ordering::SeqCst)
    }

    /// How many ended jobs a `DELETE` forgot.
    pub fn forgotten(&self) -> usize {
        self.forgotten.load(Ordering::SeqCst)
    }

    /// How many jobs the service still holds for the caller.
    pub fn held(&self) -> usize {
        self.held.load(Ordering::SeqCst)
    }

    /// How many adapter runs actually trained: a run resubmitted under a
    /// `resume_key` whose record is completed is answered from the record
    /// and does not count.
    pub fn trained(&self) -> usize {
        self.trained.load(Ordering::SeqCst)
    }
}

/// The identity of a resumable run: its action and every param but the ones
/// that configure resumption, as brain keys a run record.
fn run_identity(action: &str, params: &Value) -> String {
    let mut params = params.clone();
    if let Some(map) = params.as_object_mut() {
        map.remove("resume_key");
        map.remove("checkpoint_every");
    }
    json!({"action": action, "params": params}).to_string()
}

const JOB_ID: &str = "11111111-2222-3333-4444-555555555555";

fn error(stream: &mut TcpStream, status: &str, message: &str) {
    let body = json!({"error": {"message": message}}).to_string();
    let _ = respond(stream, status, "application/json", &body);
}

/// Answers `request_line` when it is a job route; `false` when it is not one.
pub(crate) fn route(
    stream: &mut TcpStream,
    jobs: &Jobs,
    request_line: &str,
    last_event_id: Option<u64>,
    body: &[u8],
) -> bool {
    let Some(script) = jobs.script.lock().unwrap().clone() else {
        return false;
    };
    if request_line.contains(&format!("/v1/jobs/{JOB_ID}/"))
        && crate::scripted_stream::route(
            stream,
            &jobs.stream_runs,
            &script.stream,
            &script.store,
            request_line,
            last_event_id,
            body,
        )
    {
        return true;
    }
    if request_line.starts_with("POST /v1/jobs ") {
        create(stream, jobs, &script, body);
    } else if request_line.starts_with(&format!("GET /v1/jobs/{JOB_ID}/events")) {
        events(stream, jobs, &script, last_event_id);
    } else if request_line.starts_with(&format!("GET /v1/jobs/{JOB_ID}/result")) {
        let outputs = jobs.report.lock().unwrap().clone();
        let _ = respond(
            stream,
            "200 OK",
            "application/json",
            &json!({"outputs": outputs, "blobs": {}}).to_string(),
        );
    } else if request_line.starts_with(&format!("DELETE /v1/jobs/{JOB_ID}"))
        && jobs.stream_runs.cancel()
    {
        let _ = respond(
            stream,
            "200 OK",
            "application/json",
            &json!({"id": JOB_ID, "status": "cancelled"}).to_string(),
        );
    } else if request_line.starts_with(&format!("DELETE /v1/jobs/{JOB_ID}")) {
        // As brain: a running job is cancelled (and still held); an ended one
        // is forgotten, which frees its place.
        let status = if jobs.finished.load(Ordering::SeqCst) {
            if jobs.held.load(Ordering::SeqCst) > 0 {
                jobs.held.fetch_sub(1, Ordering::SeqCst);
                jobs.forgotten.fetch_add(1, Ordering::SeqCst);
            }
            "deleted"
        } else {
            jobs.cancelled.store(true, Ordering::SeqCst);
            jobs.deleted.store(true, Ordering::SeqCst);
            jobs.finished.store(true, Ordering::SeqCst);
            "cancelled"
        };
        let _ = respond(
            stream,
            "200 OK",
            "application/json",
            &json!({"id": JOB_ID, "status": status}).to_string(),
        );
    } else {
        return false;
    }
    true
}

/// The digests a `data` document names.
fn digests_of(data: &Value) -> Vec<String> {
    let mut found = Vec::new();
    let mut take = |v: &Value| {
        if let Some(d) = v["digest"].as_str() {
            found.push(d.to_string());
        }
    };
    for name in [
        "train",
        "held_out",
        "held_out_text",
        "monitor",
        "start_adapter",
    ] {
        take(&data[name]);
    }
    for entry in data["replay"].as_array().into_iter().flatten() {
        take(entry);
    }
    for entry in data["weighted_replay"].as_array().into_iter().flatten() {
        take(entry);
    }
    found
}

fn create(stream: &mut TcpStream, jobs: &Jobs, script: &JobScript, body: &[u8]) {
    let call: Value = serde_json::from_slice(body).unwrap_or_default();
    let action = call["action"].as_str().unwrap_or_default().to_string();
    if call["model"] != "brain/train"
        || ![
            "train",
            "check_dataset",
            "full_train",
            "merge",
            "speech_sft",
        ]
        .contains(&action.as_str())
    {
        error(stream, "404 Not Found", "no such model or action");
        return;
    }
    let checking = action == "check_dataset";
    if script
        .max_held
        .is_some_and(|max| jobs.held.load(Ordering::SeqCst) >= max)
    {
        error(
            stream,
            "429 Too Many Requests",
            "too many jobs: fetch or cancel some before starting another",
        );
        return;
    }
    if !checking && jobs.refused.load(Ordering::SeqCst) < script.refuse_first {
        jobs.refused.fetch_add(1, Ordering::SeqCst);
        error(
            stream,
            "429 Too Many Requests",
            "every card is busy serving",
        );
        return;
    }
    let params = call["params"].clone();
    if action == "train" && params["stream"].is_string() {
        match crate::scripted_stream::create(&jobs.stream_runs, &script.stream, &params) {
            Ok(_) => {
                jobs.created.lock().unwrap().push(params);
                jobs.actions.lock().unwrap().push(action);
                let _ = respond(
                    stream,
                    "202 Accepted",
                    "application/json",
                    &json!({"id": JOB_ID, "status": "running"}).to_string(),
                );
            }
            Err(crate::scripted_stream::NotCreated::Invalid(why)) => {
                error(stream, "400 Bad Request", &format!("train: {why}"));
            }
            Err(crate::scripted_stream::NotCreated::Busy(why)) => {
                error(stream, "429 Too Many Requests", &format!("train: {why}"));
            }
        }
        return;
    }
    jobs.stream_runs.clear();
    let data: Value =
        serde_json::from_str(params["data"].as_str().unwrap_or("{}")).unwrap_or_default();
    // brain resolves each digest in its store and verifies the file against it.
    let named = ["dataset_digest", "held_out_digest", "adapter_digest"]
        .iter()
        .filter_map(|name| params[*name].as_str().map(str::to_string));
    for hex in digests_of(&data).into_iter().chain(named) {
        let digest = Digest::from_content_hex(&hex);
        let file = script.store.join(&hex);
        let verified = digest.ok().and_then(|d| {
            let reader = std::fs::File::open(&file).ok()?;
            (Digest::of_reader(reader).ok()? == d).then_some(())
        });
        if verified.is_none() {
            error(
                stream,
                "400 Bad Request",
                &format!("train: artifact {hex} is not in the artifact store"),
            );
            return;
        }
    }
    if checking {
        jobs.checks.lock().unwrap().push(params.clone());
    } else {
        jobs.created.lock().unwrap().push(params.clone());
    }
    jobs.checking.store(checking, Ordering::SeqCst);
    jobs.held.fetch_add(1, Ordering::SeqCst);
    jobs.finished.store(false, Ordering::SeqCst);
    jobs.actions.lock().unwrap().push(action.clone());
    *jobs.events.lock().unwrap() = None;
    jobs.cancelled.store(false, Ordering::SeqCst);
    let resume = params["resume_key"].as_str().map(str::to_string);
    let identity = run_identity(&action, &params);
    let recorded = resume
        .as_ref()
        .and_then(|key| jobs.runs.lock().unwrap().get(key).cloned());
    let (frames, report) = match (recorded, action.as_str()) {
        // brain answers a completed run's resubmission from its record.
        (Some((known, mut report)), _) if known == identity => {
            report["resume"] = json!({"state": "completed"});
            (Vec::new(), report)
        }
        (Some(_), _) => {
            error(
                stream,
                "400 Bad Request",
                &format!(
                    "resume_key {:?}: its record names another run; one key is one run",
                    resume.unwrap_or_default()
                ),
            );
            return;
        }
        (None, "check_dataset") => (Vec::new(), check(script, &params)),
        (None, "full_train" | "merge" | "speech_sft") => checkpoint(script, &action, &params),
        (None, _) => {
            jobs.trained.fetch_add(1, Ordering::SeqCst);
            let (frames, report) = train(script, &params);
            if let Some(key) = &resume {
                jobs.runs
                    .lock()
                    .unwrap()
                    .insert(key.clone(), (identity, report.clone()));
            }
            (frames, report)
        }
    };
    *jobs.events.lock().unwrap() = Some(frames);
    *jobs.report.lock().unwrap() = report;
    let _ = respond(
        stream,
        "202 Accepted",
        "application/json",
        &json!({"id": JOB_ID, "status": "running"}).to_string(),
    );
}

/// "Trains": publishes an adapter and a record to the store, and returns the
/// frames a run streams and its report.
fn train(script: &JobScript, params: &Value) -> (Vec<(u64, String, Value)>, Value) {
    let pg = params["objective"] == "policy_gradient";
    let dpo = params["objective"] == "preference";
    let kto = params["objective"] == "feedback";
    let publish = |bytes: &[u8]| {
        let digest = Digest::of(bytes);
        std::fs::write(script.store.join(digest.hex()), bytes).unwrap();
        digest
    };
    let adapter = format!("scripted adapter, seed {}", params["seed"]).into_bytes();
    let adapter_digest = publish(&adapter);
    let step_series: Vec<Value> = (1..=script.steps)
        .map(|s| {
            json!({"step": s, "loss": 1.0 / f64::from(s), "clip_fraction": 0.0, "kl": 0.01 * f64::from(s),
                   "ratio_mean": 1.0, "ratio_min": 0.9, "ratio_max": 1.1, "mean_advantage": 0.5})
        })
        .collect();
    let record = json!({"policy_gradient": {"steps": step_series}});
    let record_digest = publish(record.to_string().as_bytes());
    let mut frames: Vec<(u64, String, Value)> = vec![
        (
            1,
            "frame".into(),
            json!({"type": "stage", "stage": "staging the datasets"}),
        ),
        (
            2,
            "frame".into(),
            json!({"type": "stage", "stage": "loading the base and training"}),
        ),
    ];
    for s in 1..=script.steps {
        let mut frame = json!({"type": "step", "step": s, "steps": script.steps, "loss": 1.0 / f64::from(s), "lr": 0.0003, "seconds": f64::from(s)});
        if pg {
            frame["pg"] = json!({"kl": 0.01 * f64::from(s), "clip_fraction": 0.0, "ratio_mean": 1.0, "scoring_seconds": 0.1});
        }
        frames.push((u64::from(s) + 2, "frame".into(), frame));
    }
    let score = |loss: f64| json!({"loss": loss, "token_accuracy": 0.5, "positions": 10, "records": 2, "skipped": 0});
    let mut report = json!({
        "status": "completed",
        "adapter_digest": adapter_digest.hex(),
        "record_digest": record_digest.hex(),
        "adapter_sha256": Digest::sha256_of(&adapter).as_str(),
        "base_digest": format!("sha256:{}", "ab".repeat(32)),
        "block": 128, "steps": script.steps, "steps_completed": script.steps,
        "eval_every": params["eval_every"], "patience": params["patience"],
        "monitor_records": 0, "curve": [], "selected_step": script.steps,
        "selection": "last_step", "stopped_early": false, "evaluations": [],
        "base_score": if pg { Value::Null } else { score(2.0) },
        "tuned_score": if pg { Value::Null } else { score(1.5) },
        "policy_gradient": if pg { json!({"steps_skipped_nonfinite": 0, "steps_taken": script.steps}) } else { Value::Null },
    });
    if let Some(parallel) = &script.parallel {
        report["parallel"] = parallel.clone();
    }
    if dpo {
        let pairs = json!({"accuracy": 0.75, "mean_margin": 0.5, "pairs": 4, "skipped": 0});
        report["beta"] = params["beta"].clone();
        report["nll_weight"] = params["nll_weight"].clone();
        report["trained_from"] = Value::Null;
        report["train_score"] = pairs.clone();
        report["held_out_score"] = pairs;
        report["base_score"] = Value::Null;
        report["tuned_score"] = Value::Null;
    }
    if kto {
        report["beta"] = json!(0.1);
        report["desirable_weight"] = params["desirable_weight"].clone();
        report["undesirable_weight"] = params["undesirable_weight"].clone();
        report["desirable_examples"] = json!(2);
        report["undesirable_examples"] = json!(1);
        report["initial_loss"] = json!(0.69);
        report["final_loss"] = json!(0.5);
        report["desirable_logratio"] = json!(0.2);
        report["undesirable_logratio"] = json!(-0.3);
        report["base_score"] = Value::Null;
        report["tuned_score"] = Value::Null;
    }
    (frames, report)
}

/// The record lengths `check_dataset` reports for the dataset of `params`.
fn check(script: &JobScript, params: &Value) -> Value {
    let hex = params["dataset_digest"].as_str().unwrap_or_default();
    let text = std::fs::read_to_string(script.store.join(hex)).unwrap_or_default();
    let kind = params["kind"].as_str().unwrap_or("chat");
    let max_block = params["max_block"].as_u64().unwrap_or(8192) as usize;
    let chars = |m: &Value| m["content"].as_str().map_or(0, |c| c.chars().count());
    let (mut lengths, mut invalid, mut over) = (Vec::new(), Vec::new(), Vec::new());
    let mut records = 0usize;
    for (index, line) in text
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
    {
        records += 1;
        let line_no = index + 1;
        let record: Value = serde_json::from_str(line).unwrap_or_default();
        if kind == "chat"
            && script
                .refuse_records_containing
                .as_deref()
                .is_some_and(|marker| line.contains(marker))
        {
            invalid.push(json!({"line": line_no, "error": "the scripted trainer refuses this record", "rule": "scripted", "field": null}));
            continue;
        }
        let measured = match kind {
            "chat" => record["messages"].as_array().map(|messages| {
                let trained: usize = messages
                    .iter()
                    .filter(|m| m["train"] == true)
                    .map(chars)
                    .sum();
                let prompt: usize = messages
                    .iter()
                    .filter(|m| m["train"] != true)
                    .map(chars)
                    .sum();
                (prompt, trained, 1usize)
            }),
            "preference" => (record.get("chosen").is_some() && record.get("rejected").is_some())
                .then(|| (line.chars().count(), 0, 2)),
            "feedback" => (record.get("completion").is_some() && record["desirable"].is_boolean())
                .then(|| (line.chars().count(), 0, 1)),
            _ => record.get("subject").map(|_| (0, 0, 1)),
        };
        match measured {
            Some((prompt, trained, examples)) => {
                if prompt + trained > max_block {
                    over.push(line_no);
                }
                lengths.push((line_no, prompt + trained, prompt, trained, examples));
            }
            None => invalid.push(json!({"line": line_no, "error": format!("not a {kind} record"), "rule": null, "field": null})),
        }
    }
    let spread = |values: Vec<usize>| {
        let mut values = values;
        values.sort_unstable();
        let total: usize = values.iter().sum();
        let at = |q: f64| {
            values
                .get(((q * values.len() as f64).ceil() as usize).clamp(1, values.len().max(1)) - 1)
                .copied()
                .unwrap_or(0)
        };
        json!({"total": total, "min": values.first().copied().unwrap_or(0),
               "mean": if values.is_empty() { 0.0 } else { total as f64 / values.len() as f64 },
               "p50": at(0.5), "p90": at(0.9), "p99": at(0.99), "max": values.last().copied().unwrap_or(0)})
    };
    let examples: usize = lengths.iter().map(|l| l.4).sum();
    let mut report = json!({
        "kind": kind, "records": records, "valid_records": lengths.len(), "examples": examples,
        "tokens": spread(lengths.iter().map(|l| l.1).collect()),
        "supervised_tokens": spread(lengths.iter().map(|l| l.3).collect()),
        "examples_without_supervision": lengths.iter().filter(|l| kind == "chat" && l.3 == 0).count(),
        "max_block": max_block, "over_max_block": over.len(), "over_max_block_lines": over,
        "invalid_total": invalid.len(), "invalid": invalid.clone(), "listed_at_most": 20,
    });
    if params["per_record"] == true {
        let offset = params["offset"].as_u64().unwrap_or(0) as usize;
        let limit = params["limit"].as_u64().unwrap_or(50_000) as usize;
        let page: Vec<Value> = lengths
            .iter()
            .skip(offset)
            .take(limit)
            .map(|l| json!([l.0, l.1, l.2, l.3, l.4]))
            .collect();
        let next = (offset + limit < lengths.len()).then_some(offset + limit);
        report["per_record"] = json!(page);
        // The refusals are listed once, on the first page.
        report["per_record_refused"] = if offset == 0 {
            json!(invalid)
        } else {
            json!([])
        };
        report["per_record_next"] = json!(next);
    }
    report
}

/// A checkpoint job (`full_train`, `speech_sft` or `merge`): two files and
/// their manifest published to the store, installed in the model store when
/// the script names one.
fn checkpoint(
    script: &JobScript,
    action: &str,
    params: &Value,
) -> (Vec<(u64, String, Value)>, Value) {
    let publish = |bytes: &[u8]| {
        let digest = Digest::of(bytes);
        std::fs::write(script.store.join(digest.hex()), bytes).unwrap();
        digest
    };
    let weights = format!("scripted {action} weights, seed {}", params["seed"]).into_bytes();
    let config = b"{\"model_type\": \"qwen3\"}".to_vec();
    let files = [("config.json", &config), ("model.safetensors", &weights)];
    let listed: Vec<Value> = files
        .iter()
        .map(|(name, bytes)| json!({"name": name, "digest": publish(bytes).hex(), "bytes": bytes.len()}))
        .collect();
    let manifest = json!({"format": "brain-checkpoint-manifest-v1", "files": listed});
    let manifest_digest = publish(manifest.to_string().as_bytes());
    let name = params["install_as"].as_str().unwrap_or_default();
    if let (Some(models), false) = (&script.models, name.is_empty()) {
        let dir = models.join("local").join(name);
        std::fs::create_dir_all(&dir).unwrap();
        for (file, bytes) in files {
            std::fs::write(dir.join(file), bytes).unwrap();
        }
    }
    // Every checkpoint job but a merge trains, in steps, as ranks.
    let full = action != "merge";
    let steps = if full { script.steps } else { 0 };
    let frames = (1..=steps)
        .map(|s| (u64::from(s), "frame".to_string(), json!({"type": "step", "step": s, "steps": steps, "loss": 1.0 / f64::from(s), "seconds": f64::from(s)})))
        .collect();
    let report = json!({
        "status": "completed",
        "objective": match action { "full_train" => "full_sft", other => other },
        "weights_sha256": Digest::sha256_of(&weights).as_str(),
        "steps_completed": steps,
        "initial_loss": if full { json!(2.0) } else { Value::Null },
        "final_loss": if full { json!(1.0) } else { Value::Null },
        "checkpoint_digest": manifest_digest.hex(),
        "files": listed,
        "installed_as": if name.is_empty() { Value::Null } else { json!(format!("local/{name}")) },
    });
    let mut report = report;
    if let (true, Some(parallel)) = (full, &script.parallel) {
        report["parallel"] = parallel.clone();
    }
    (frames, report)
}

fn events(stream: &mut TcpStream, jobs: &Jobs, script: &JobScript, last_event_id: Option<u64>) {
    if jobs.checking.load(Ordering::SeqCst) {
        // A check answers at once, whatever the script makes a run do.
        let frames = jobs.events.lock().unwrap().clone().unwrap_or_default();
        let last = frames.last().map_or(0, |(id, _, _)| *id) + 1;
        let done = json!({"status": "succeeded"});
        jobs.finished.store(true, Ordering::SeqCst);
        let _ = respond(
            stream,
            "200 OK",
            "text/event-stream",
            &format!("id: {last}\nevent: done\ndata: {done}\n\n"),
        );
        return;
    }
    let first_stream = {
        let mut streams = jobs.streams.lock().unwrap();
        streams.push(last_event_id);
        streams.len() == 1
    };
    let frames = jobs.events.lock().unwrap().clone().unwrap_or_default();
    let mut sse = String::new();
    let after = last_event_id.unwrap_or(0);
    for (sent, (id, name, data)) in frames.iter().filter(|(id, _, _)| *id > after).enumerate() {
        if first_stream && script.break_stream_after.is_some_and(|n| sent >= n) {
            // The connection ends here, with no `done` event.
            let _ = respond(stream, "200 OK", "text/event-stream", &sse);
            return;
        }
        sse.push_str(&format!("id: {id}\nevent: {name}\ndata: {data}\n\n"));
    }
    let last = frames.last().map_or(0, |(id, _, _)| *id) + 1;
    let done = if script.hang {
        // Stream what there is, then hold the connection until cancelled.
        let mut stream_ref = stream.try_clone().unwrap();
        let _ = std::io::Write::write_all(
            &mut stream_ref,
            b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n",
        );
        let _ = std::io::Write::write_all(&mut stream_ref, sse.as_bytes());
        let _ = std::io::Write::flush(&mut stream_ref);
        while !jobs.deleted.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(10));
        }
        let tail = format!(
            "id: {last}\nevent: done\ndata: {}\n\n",
            json!({"status": "cancelled"})
        );
        let _ = std::io::Write::write_all(&mut stream_ref, tail.as_bytes());
        return;
    } else if let Some(reason) = &script.fail_with {
        json!({"status": "failed", "error": {"message": reason, "code": 500}})
    } else {
        json!({"status": "succeeded"})
    };
    if last > after {
        sse.push_str(&format!("id: {last}\nevent: done\ndata: {done}\n\n"));
        jobs.finished.store(true, Ordering::SeqCst);
    }
    let _ = respond(stream, "200 OK", "text/event-stream", &sse);
}
