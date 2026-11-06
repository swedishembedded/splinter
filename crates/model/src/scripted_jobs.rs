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
//! model or a device.
//!
//! A job reads its datasets from the artifact store by digest and refuses a
//! digest the store does not hold or whose file does not hash to its name; it
//! "trains" by publishing an adapter and a training record to the store and
//! streams the frames a run would. What a script can make it do: refuse the
//! first creations for want of room (HTTP 429), break the event stream after
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
        }
    }
}

/// What a scripted training service saw and holds.
#[derive(Default)]
pub struct Jobs {
    pub(crate) script: Mutex<Option<JobScript>>,
    created: Mutex<Vec<Value>>,
    refused: AtomicUsize,
    streams: Mutex<Vec<Option<u64>>>,
    events: Mutex<Option<Vec<(u64, String, Value)>>>,
    report: Mutex<Value>,
    cancelled: AtomicBool,
    deleted: AtomicBool,
}

impl Jobs {
    /// The parameters of every job created, in order.
    pub fn created(&self) -> Vec<Value> {
        self.created.lock().unwrap().clone()
    }

    /// The `Last-Event-ID` of every event stream opened, in order.
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
    } else if request_line.starts_with(&format!("DELETE /v1/jobs/{JOB_ID}")) {
        jobs.cancelled.store(true, Ordering::SeqCst);
        jobs.deleted.store(true, Ordering::SeqCst);
        let _ = respond(
            stream,
            "200 OK",
            "application/json",
            &json!({"id": JOB_ID, "status": "cancelled"}).to_string(),
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
    for name in ["train", "held_out", "held_out_text", "monitor", "start_adapter"] {
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
    if call["model"] != "brain/train" || call["action"] != "train" {
        error(stream, "404 Not Found", "no such model or action");
        return;
    }
    if jobs.refused.load(Ordering::SeqCst) < script.refuse_first {
        jobs.refused.fetch_add(1, Ordering::SeqCst);
        error(stream, "429 Too Many Requests", "every card is busy serving");
        return;
    }
    let params = call["params"].clone();
    let data: Value =
        serde_json::from_str(params["data"].as_str().unwrap_or("{}")).unwrap_or_default();
    // brain resolves each digest in its store and verifies the file against it.
    for hex in digests_of(&data) {
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
    jobs.created.lock().unwrap().push(params.clone());
    *jobs.events.lock().unwrap() = None;
    jobs.cancelled.store(false, Ordering::SeqCst);
    let (frames, report) = train(script, &params);
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
        (1, "frame".into(), json!({"type": "stage", "stage": "staging the datasets"})),
        (2, "frame".into(), json!({"type": "stage", "stage": "loading the base and training"})),
    ];
    for s in 1..=script.steps {
        let mut frame = json!({"type": "step", "step": s, "steps": script.steps, "loss": 1.0 / f64::from(s), "lr": 0.0003, "seconds": f64::from(s)});
        if pg {
            frame["pg"] = json!({"kl": 0.01 * f64::from(s), "clip_fraction": 0.0, "ratio_mean": 1.0, "scoring_seconds": 0.1});
        }
        frames.push((u64::from(s) + 2, "frame".into(), frame));
    }
    let score = |loss: f64| json!({"loss": loss, "token_accuracy": 0.5, "positions": 10, "records": 2, "skipped": 0});
    let report = json!({
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
    (frames, report)
}

fn events(stream: &mut TcpStream, jobs: &Jobs, script: &JobScript, last_event_id: Option<u64>) {
    let first_stream = {
        let mut streams = jobs.streams.lock().unwrap();
        streams.push(last_event_id);
        streams.len() == 1
    };
    let frames = jobs.events.lock().unwrap().clone().unwrap_or_default();
    let mut sse = String::new();
    let after = last_event_id.unwrap_or(0);
    for (sent, (id, name, data)) in frames
        .iter()
        .filter(|(id, _, _)| *id > after)
        .enumerate()
    {
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
        let tail = format!("id: {last}\nevent: done\ndata: {}\n\n", json!({"status": "cancelled"}));
        let _ = std::io::Write::write_all(&mut stream_ref, tail.as_bytes());
        return;
    } else if let Some(reason) = &script.fail_with {
        json!({"status": "failed", "error": {"message": reason, "code": 500}})
    } else {
        json!({"status": "succeeded"})
    };
    if last > after {
        sse.push_str(&format!("id: {last}\nevent: done\ndata: {done}\n\n"));
    }
    let _ = respond(stream, "200 OK", "text/event-stream", &sse);
}
