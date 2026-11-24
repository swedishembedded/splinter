// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements online reinforcement learning against a
// training service that serves each policy version it trains, for its
// clients. If your team needs expertise in testing streaming training clients
// against a scripted service, you can procure our services by sending an
// email to info@swedishembedded.com.

//! The streaming half of the scripted training service: a `brain/train` run
//! with a `stream` document, answering as brain's streaming policy-gradient
//! job does without a model.
//!
//! Records (`POST /v1/jobs/{id}/records?epoch=N`) must be complete groups
//! (each group two rollouts or more); a group's policy tokens are the
//! `token_ids` of its supervised messages and its age the oldest
//! `policy_version` of its records. A step trains once `step_tokens` are
//! queued (or, after `finish`, whatever is queued): whole groups in arrival
//! order, a group more than `max_staleness` versions behind dropped instead.
//! Every `publish_every` steps a version is published (an adapter blob in the
//! artifact store); every `checkpoint_every` steps (and at the end) what was
//! applied is committed. An offer past `queue_tokens` is refused (429) unless
//! less than a step is queued. Every key is accepted once per run: a resumed
//! run (the same `resume_key`) starts in the next epoch with its committed
//! keys and its published version, and answers them as duplicates.
//!
//! Steps are taken when the run's events are read (or it is told to finish),
//! never while an offer is answered, so a test can fill the queue between
//! two reads. What a script can make it do ([`StreamScript`]): restart once a number of
//! steps is reached (the job is then unknown, 404, and a new create of the
//! same run continues it in the next epoch, losing what was not committed).

// Test support: a poisoned lock or a socket that cannot be written is the
// failure report.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::{BTreeMap, HashSet, VecDeque};
use std::net::TcpStream;
use std::sync::Mutex;

use serde_json::{json, Value};
use splinter_core::digest::Digest;

use crate::scripted_endpoint::respond;

/// How a scripted streaming run behaves beyond its stream document.
#[derive(Clone, Debug, Default)]
pub struct StreamScript {
    /// The run restarts (its job forgotten, uncommitted work lost) the first
    /// time it reaches this many steps.
    pub restart_at_step: Option<u64>,
    /// While set, the run takes no step: a test fills its queue.
    pub paused: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

/// One group waiting in the queue.
#[derive(Clone, Debug)]
struct Queued {
    key: String,
    tokens: u64,
    oldest: u64,
}

/// What a run keeps across its epochs: its checkpoint.
#[derive(Clone, Debug, Default)]
struct Durable {
    epoch: u64,
    version: u64,
    step: u64,
    committed: HashSet<String>,
    committed_step: u64,
    restarted: bool,
}

/// The run being streamed now.
#[derive(Debug)]
struct Live {
    resume_key: String,
    doc: Value,
    checkpoint_every: u64,
    queue: VecDeque<Queued>,
    held: HashSet<String>,
    applied: Vec<String>,
    frames: Vec<(u64, Value)>,
    finishing: bool,
    done: bool,
    gone: bool,
    dropped_stale: u64,
}

/// The streaming runs of the scripted service.
#[derive(Default)]
pub struct Streams {
    live: Mutex<Option<Live>>,
    durable: Mutex<BTreeMap<String, Durable>>,
    offered: Mutex<Vec<String>>,
    trained: Mutex<Vec<String>>,
    finished: Mutex<usize>,
}

impl Streams {
    /// Ends the streaming run's hold on the job routes: a job of another
    /// kind was created.
    pub(crate) fn clear(&self) {
        *self.live.lock().unwrap() = None;
    }

    /// Every group key offered, in order, each time it was offered.
    pub fn offered(&self) -> Vec<String> {
        self.offered.lock().unwrap().clone()
    }

    /// Every group key a step applied, in order (a key applied in a lost
    /// epoch and again later is listed twice).
    pub fn applied(&self) -> Vec<String> {
        self.trained.lock().unwrap().clone()
    }

    /// Every key committed by every run, sorted.
    pub fn committed(&self) -> Vec<String> {
        let mut keys: Vec<String> = self
            .durable
            .lock()
            .unwrap()
            .values()
            .flat_map(|d| d.committed.iter().cloned())
            .collect();
        keys.sort();
        keys
    }

    /// How many times a run was told to finish.
    pub fn finishes(&self) -> usize {
        *self.finished.lock().unwrap()
    }
}

fn number(doc: &Value, key: &str, default: u64) -> u64 {
    doc[key].as_u64().unwrap_or(default)
}

/// Creates a streaming run from `params` (which hold a `stream` document);
/// returns its first frame's epoch, or the refusal.
pub(crate) fn create(streams: &Streams, params: &Value) -> Result<u64, String> {
    let doc: Value = serde_json::from_str(params["stream"].as_str().unwrap_or("{}"))
        .map_err(|e| format!("stream: {e}"))?;
    for field in [
        "publish_adapter",
        "publish_every",
        "step_tokens",
        "queue_tokens",
    ] {
        if doc.get(field).is_none() {
            return Err(format!("stream: missing {field}"));
        }
    }
    if number(&doc, "queue_tokens", 0) < number(&doc, "step_tokens", 1) {
        return Err("stream: queue_tokens is below step_tokens".into());
    }
    let resume_key = params["resume_key"].as_str().unwrap_or("").to_string();
    let mut durable = streams.durable.lock().unwrap();
    let record = durable.entry(resume_key.clone()).or_default();
    record.epoch += 1;
    // A run continued from its checkpoint: what came after it is lost.
    record.step = record.committed_step;
    let first = json!({
        "type": "stream", "epoch": record.epoch, "publish_adapter": doc["publish_adapter"],
        "version": record.version, "step_tokens": doc["step_tokens"],
        "queue_tokens": doc["queue_tokens"], "max_staleness": doc["max_staleness"],
    });
    let epoch = record.epoch;
    *streams.live.lock().unwrap() = Some(Live {
        resume_key,
        checkpoint_every: params["checkpoint_every"].as_u64().unwrap_or(1).max(1),
        doc,
        queue: VecDeque::new(),
        held: HashSet::new(),
        applied: Vec::new(),
        frames: vec![(1, first)],
        finishing: false,
        done: false,
        gone: false,
        dropped_stale: 0,
    });
    Ok(epoch)
}

fn refuse(stream: &mut TcpStream, status: &str, message: &str, detail: Value) {
    let body = json!({"error": {"message": message}, "detail": detail}).to_string();
    let _ = respond(stream, status, "application/json", &body);
}

/// The epoch a request line names (`?epoch=N`).
fn epoch_of(request_line: &str) -> Option<u64> {
    let query = request_line.split_whitespace().nth(1)?.split_once('?')?.1;
    query
        .split('&')
        .find_map(|pair| pair.strip_prefix("epoch="))
        .and_then(|e| e.parse().ok())
}

/// Whether the live run may be addressed by `request_line`; answers the
/// refusal when not.
fn addressed<'a>(
    stream: &mut TcpStream,
    live: &'a mut Option<Live>,
    durable: &BTreeMap<String, Durable>,
    request_line: &str,
) -> Option<&'a mut Live> {
    let Some(run) = live.as_mut().filter(|run| !run.gone) else {
        refuse(stream, "404 Not Found", "no such job", Value::Null);
        return None;
    };
    let epoch = durable.get(&run.resume_key).map_or(0, |d| d.epoch);
    if epoch_of(request_line).is_some_and(|asked| asked != epoch) {
        refuse(
            stream,
            "409 Conflict",
            "another epoch",
            json!({"epoch": epoch}),
        );
        return None;
    }
    if run.done || run.finishing && request_line.contains("/records") {
        refuse(
            stream,
            "409 Conflict",
            "the run is finishing",
            json!({"epoch": epoch}),
        );
        return None;
    }
    Some(run)
}

/// A group of the body: its key, policy tokens, oldest version and rollouts.
fn groups_of(body: &[u8]) -> Result<Vec<Queued>, String> {
    let text = std::str::from_utf8(body).map_err(|e| e.to_string())?;
    let mut groups: BTreeMap<String, (u64, u64, HashSet<String>)> = BTreeMap::new();
    let mut order = Vec::new();
    for (n, line) in text
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
    {
        let record: Value =
            serde_json::from_str(line).map_err(|e| format!("line {}: {e}", n + 1))?;
        let key = record["group"]
            .as_str()
            .ok_or_else(|| format!("line {}: no group", n + 1))?
            .to_string();
        if record["reward"].as_f64().is_none() {
            return Err(format!("line {}: no reward", n + 1));
        }
        let tokens: u64 = record["messages"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|m| m["train"] == true)
            .map(|m| m["token_ids"].as_array().map_or(0, Vec::len) as u64)
            .sum();
        let version = record["policy_version"].as_u64().unwrap_or(0);
        let rollout = record["rollout"].as_str().unwrap_or(line).to_string();
        let entry = groups.entry(key.clone()).or_insert_with(|| {
            order.push(key.clone());
            (0, u64::MAX, HashSet::new())
        });
        entry.0 += tokens;
        entry.1 = entry.1.min(version);
        entry.2.insert(rollout);
    }
    order
        .into_iter()
        .map(|key| {
            let (tokens, oldest, rollouts) = &groups[&key];
            if rollouts.len() < 2 {
                return Err(format!("group {key} holds one rollout (group_of_one)"));
            }
            Ok(Queued {
                key,
                tokens: *tokens,
                oldest: *oldest,
            })
        })
        .collect()
}

/// Trains every step the queue can fill, publishes and commits as the
/// document says, and ends the run when it is finishing and empty or at its
/// step limit.
fn advance(
    streams: &Streams,
    run: &mut Live,
    durable: &mut Durable,
    script: &StreamScript,
    store: &std::path::Path,
) {
    if script.paused.load(std::sync::atomic::Ordering::SeqCst) {
        return;
    }
    let step_tokens = number(&run.doc, "step_tokens", 1);
    let publish_every = number(&run.doc, "publish_every", 1).max(1);
    let max_staleness = run.doc["max_staleness"].as_u64();
    let max_steps = run.doc["max_steps"].as_u64();
    let mut next = run.frames.last().map_or(1, |f| f.0 + 1);
    let mut push = |run: &mut Live, frame: Value| {
        run.frames.push((next, frame));
        next += 1;
    };
    loop {
        let queued: u64 = run.queue.iter().map(|g| g.tokens).sum();
        let at_limit = max_steps.is_some_and(|m| durable.step >= m);
        if run.queue.is_empty() || at_limit || (queued < step_tokens && !run.finishing) {
            break;
        }
        let (mut tokens, mut taken, mut dropped, mut ages) =
            (0, Vec::new(), Vec::new(), Vec::new());
        while tokens < step_tokens {
            let Some(group) = run.queue.pop_front() else {
                break;
            };
            let age = durable.version.saturating_sub(group.oldest);
            if max_staleness.is_some_and(|m| age > m) {
                dropped.push(group.key);
                continue;
            }
            tokens += group.tokens;
            ages.push(age);
            taken.push(group.key);
        }
        if !dropped.is_empty() {
            run.dropped_stale += dropped.len() as u64;
            for key in &dropped {
                run.held.remove(key);
            }
            push(
                run,
                json!({"type": "dropped", "reason": "stale", "groups": dropped}),
            );
        }
        if taken.is_empty() {
            continue;
        }
        durable.step += 1;
        let step = durable.step;
        streams
            .trained
            .lock()
            .unwrap()
            .extend(taken.iter().cloned());
        let staleness_max = ages.iter().copied().max().unwrap_or(0);
        #[allow(clippy::cast_precision_loss)]
        let staleness_mean = ages.iter().sum::<u64>() as f64 / ages.len() as f64;
        push(
            run,
            json!({"type": "applied", "step": step, "groups": taken}),
        );
        let queue_tokens: u64 = run.queue.iter().map(|g| g.tokens).sum();
        push(
            run,
            json!({"type": "step", "step": step, "pg": {
                "staleness_mean": staleness_mean, "staleness_max": staleness_max,
                "policy_tokens": tokens, "learner_wait_seconds": 0.0,
                "queue_groups": run.queue.len(), "queue_tokens": queue_tokens,
                "time_in_queue_seconds": 0.0, "dropped_stale": run.dropped_stale,
            }}),
        );
        run.applied.extend(taken);
        if step.is_multiple_of(publish_every) {
            durable.version += 1;
            let bytes =
                format!("scripted streamed adapter, version {}", durable.version).into_bytes();
            let digest = Digest::of(&bytes);
            let _ = std::fs::write(store.join(digest.hex()), &bytes);
            push(
                run,
                json!({"type": "published", "version": durable.version, "digest": digest.hex(), "step": step}),
            );
        }
        if step.is_multiple_of(run.checkpoint_every) {
            commit(run, durable, &mut push);
        }
        if script.restart_at_step == Some(step) && !durable.restarted {
            // The server goes down: nothing after the last checkpoint survives.
            durable.restarted = true;
            run.gone = true;
            return;
        }
    }
    let at_limit = max_steps.is_some_and(|m| durable.step >= m);
    if (run.finishing && run.queue.is_empty()) || at_limit {
        commit(run, durable, &mut push);
        push(run, json!({"type": "done", "status": "succeeded"}));
        run.done = true;
    }
}

fn commit(run: &mut Live, durable: &mut Durable, push: &mut impl FnMut(&mut Live, Value)) {
    if run.applied.is_empty() {
        return;
    }
    let groups = std::mem::take(&mut run.applied);
    durable.committed.extend(groups.iter().cloned());
    let step = durable.step;
    durable.committed_step = step;
    push(
        run,
        json!({"type": "committed", "step": step, "groups": groups}),
    );
}

/// Answers a streaming route (`records`, `finish`, `events`, `result`);
/// `false` when the live run is not a stream.
pub(crate) fn route(
    stream: &mut TcpStream,
    streams: &Streams,
    script: &StreamScript,
    store: &std::path::Path,
    request_line: &str,
    last_event_id: Option<u64>,
    body: &[u8],
) -> bool {
    let mut live = streams.live.lock().unwrap();
    if live.is_none() {
        return false;
    }
    let mut durable_map = streams.durable.lock().unwrap();
    if request_line.contains("/events") {
        let Some(run) = live.as_mut().filter(|run| !run.gone) else {
            refuse(stream, "404 Not Found", "no such job", Value::Null);
            return true;
        };
        let durable = durable_map.entry(run.resume_key.clone()).or_default();
        advance(streams, run, durable, script, store);
        if run.gone {
            refuse(stream, "404 Not Found", "no such job", Value::Null);
            return true;
        }
        let after = last_event_id.unwrap_or(0);
        let mut sse = String::new();
        for (id, frame) in run.frames.iter().filter(|(id, _)| *id > after) {
            let name = if frame["type"] == "done" {
                "done"
            } else {
                "frame"
            };
            sse.push_str(&format!("id: {id}\nevent: {name}\ndata: {frame}\n\n"));
        }
        let _ = respond(stream, "200 OK", "text/event-stream", &sse);
        return true;
    }
    if request_line.contains("/result") {
        let Some(run) = live.as_ref() else {
            return false;
        };
        let durable = durable_map
            .get(&run.resume_key)
            .cloned()
            .unwrap_or_default();
        let outputs = json!({"status": "completed", "version": durable.version, "steps": durable.step,
                             "committed": durable.committed.len()});
        let _ = respond(
            stream,
            "200 OK",
            "application/json",
            &json!({"outputs": outputs, "blobs": {}}).to_string(),
        );
        return true;
    }
    let finish = request_line.contains("/finish");
    if !finish && !request_line.contains("/records") {
        return false;
    }
    let Some(run) = addressed(stream, &mut live, &durable_map, request_line) else {
        return true;
    };
    let durable = durable_map.entry(run.resume_key.clone()).or_default();
    if finish {
        *streams.finished.lock().unwrap() += 1;
        run.finishing = true;
        advance(streams, run, durable, script, store);
        let _ = respond(
            stream,
            "200 OK",
            "application/json",
            &json!({"status": "finishing"}).to_string(),
        );
        return true;
    }
    let groups = match groups_of(body) {
        Ok(groups) => groups,
        Err(why) => {
            refuse(stream, "400 Bad Request", &why, Value::Null);
            return true;
        }
    };
    streams
        .offered
        .lock()
        .unwrap()
        .extend(groups.iter().map(|g| g.key.clone()));
    let step_tokens = number(&run.doc, "step_tokens", 1);
    let queue_tokens = number(&run.doc, "queue_tokens", step_tokens);
    let (mut accepted, mut duplicates, mut fresh) = (Vec::new(), Vec::new(), Vec::new());
    for group in groups {
        if run.held.contains(&group.key) || durable.committed.contains(&group.key) {
            duplicates.push(group.key);
        } else {
            fresh.push(group);
        }
    }
    let queued: u64 = run.queue.iter().map(|g| g.tokens).sum();
    let offered: u64 = fresh.iter().map(|g| g.tokens).sum();
    let epoch = durable.epoch;
    if !fresh.is_empty() && queued >= step_tokens && queued + offered > queue_tokens {
        let detail = json!({"allowance": queue_tokens.saturating_sub(queued), "queued_tokens": queued,
                            "offered_tokens": offered, "epoch": epoch});
        refuse(stream, "429 Too Many Requests", "the queue is full", detail);
        return true;
    }
    for group in fresh {
        run.held.insert(group.key.clone());
        accepted.push(group.key.clone());
        run.queue.push_back(group);
    }
    let queued: u64 = run.queue.iter().map(|g| g.tokens).sum();
    let answer = json!({
        "accepted": accepted, "duplicates": duplicates, "zero_advantage": 0,
        "queued_groups": run.queue.len(), "queued_tokens": queued,
        "allowance": queue_tokens.saturating_sub(queued), "epoch": epoch, "version": durable.version,
    });
    let _ = respond(stream, "200 OK", "application/json", &answer.to_string());
    true
}
