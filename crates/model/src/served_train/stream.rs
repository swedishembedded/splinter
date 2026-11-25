// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements online reinforcement learning against a
// training service that serves each policy version it trains, for its
// clients. If your team needs expertise in streaming rollouts into a running
// policy-gradient job, you can procure our services by sending an email to
// info@swedishembedded.com.

//! A streaming policy-gradient job of a brain service: the training run that
//! takes complete rollout groups while it runs and serves every version it
//! publishes.
//!
//! * **Create** (`POST /jobs`, action `train`, objective `policy_gradient`
//!   with a `stream` document): the job publishes its adapter under a fixed
//!   id every `publish_every` steps, trains a step once `step_tokens` policy
//!   tokens are queued, and holds at most `queue_tokens`. The first frame of
//!   its events names the run's `epoch` and published `version`.
//! * **Offer** (`POST /jobs/{id}/records?epoch=N`, a JSONL body of complete
//!   groups): [`Offered::Taken`] with the keys accepted and the ones the run
//!   already holds, [`Offered::Full`] (429) or [`Offered::NotReady`] (503, or
//!   a server out of reach for up to five minutes, which may be restarting)
//!   with when to ask again; an invalid body is [`StreamError::Refused`].
//! * **Events** (`GET /jobs/{id}/events`, resumed with `Last-Event-ID`): read
//!   on a thread of their own and drained by [`StreamJob::events`]: groups
//!   applied, versions published (once routable), groups dropped as stale,
//!   groups committed (in a durable checkpoint), step telemetry, the end.
//! * **Restart**: a job the server no longer knows (404, 410), an epoch it
//!   refuses (409) or a `stream` frame of another epoch means the run was
//!   restarted and lost what it had not committed; that is
//!   [`StreamError::Restarted`], and the client starts the same run again
//!   (same `resume_key`: the epoch rises, committed keys come back as
//!   duplicates) and offers everything not committed.
//! * **Finish** (`POST /jobs/{id}/finish`): no more groups; the run trains
//!   what is queued, commits it and ends.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver};
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use serde_json::{json, Map, Value};

use super::job::{message_of, EventParser};

mod place;
mod waits;
use crate::endpoint::BrainEndpoints;
use place::with_sources;
pub use place::JobPlace;

/// How long a short call (create, offer, finish) may take.
const CALL: Duration = Duration::from_secs(120);

/// How long the client waits for a new job's first frame.
const FIRST_FRAME: Duration = Duration::from_secs(300);

/// How long the event reader waits before it reopens a stream that ended.
const REOPEN: Duration = Duration::from_millis(200);

/// How long a 429 or 503 without `Retry-After` is waited out.
const DEFAULT_RETRY: Duration = Duration::from_secs(1);

/// How long a new job is asked for again while the server is busy (a run's last job still letting go of it, a
/// device serving) or cannot be reached (restarting).
const START_WAIT: Duration = Duration::from_secs(300);

/// What the job's `stream` document holds.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct StreamDoc {
    /// The id every published version is served under
    /// (`<vendor>/<repo>:<owner>:<name>:<tag>`).
    pub publish_adapter: String,
    /// Optimiser steps between two published versions.
    pub publish_every: u32,
    /// Policy tokens a step trains.
    pub step_tokens: u64,
    /// Policy tokens the queue holds at most; at least `step_tokens`.
    pub queue_tokens: u64,
    /// Versions a record may lag the published one before it is dropped.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_staleness: Option<u64>,
    /// Steps after which the run ends.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_steps: Option<u32>,
    /// The sampler's top-K the score-centred estimator centres over.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// The adapter the run starts from (`{"digest": ...}` in the artifact
    /// store).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_adapter: Option<Value>,
}

/// A streaming run to create.
#[derive(Clone, Debug, PartialEq)]
pub struct StreamSpec {
    /// The base, by brain's model id.
    pub base: String,
    /// The run's other `params` (the policy-gradient settings, `max_block`,
    /// `estimator`, ...); `objective`, `loss_agg`, `stream`, `resume_key`
    /// and `checkpoint_every` are set from the fields below.
    pub params: Map<String, Value>,
    /// The run's durable identity: asked again under it, a restarted run
    /// continues from its last checkpoint in the next epoch.
    pub resume_key: Option<String>,
    /// Steps between two durable checkpoints (each one commits the groups
    /// applied so far).
    pub checkpoint_every: Option<u32>,
    /// The stream document.
    pub stream: StreamDoc,
}

impl StreamSpec {
    /// The `params` of the create call.
    #[must_use]
    pub fn create_params(&self) -> Value {
        let mut params = self.params.clone();
        params.insert("base".into(), json!(self.base));
        params.insert("objective".into(), json!("policy_gradient"));
        params.insert("loss_agg".into(), json!("token-sum"));
        params.insert(
            "stream".into(),
            json!(serde_json::to_string(&self.stream).unwrap_or_default()),
        );
        if let Some(key) = &self.resume_key {
            params.insert("resume_key".into(), json!(key));
        }
        if let Some(every) = self.checkpoint_every {
            params.insert("checkpoint_every".into(), json!(every));
        }
        Value::Object(params)
    }
}

/// How the job answered an offer of groups.
#[derive(Clone, Debug, PartialEq)]
pub enum Offered {
    /// Queued: the keys newly accepted, the keys the run already held (this
    /// epoch's or committed in an earlier one), the room left and the
    /// published version.
    Taken {
        /// Keys newly accepted.
        accepted: Vec<String>,
        /// Keys the run already holds.
        duplicates: Vec<String>,
        /// Policy tokens the queue may still take.
        allowance: u64,
        /// The published version.
        version: u64,
    },
    /// The queue is full (429): `allowance` tokens of room, ask again after
    /// `retry_after`.
    Full {
        /// Policy tokens the queue may still take.
        allowance: u64,
        /// When to ask again.
        retry_after: Duration,
    },
    /// The job is not ready yet (503), or its server cannot be reached for now: ask again after `retry_after`.
    NotReady {
        /// When to ask again.
        retry_after: Duration,
    },
}

/// What the job reports.
#[derive(Clone, Debug, PartialEq)]
pub enum StreamEvent {
    /// An optimiser step's update holding these groups is taken.
    Applied {
        /// The step.
        step: u64,
        /// The groups.
        groups: Vec<String>,
    },
    /// `version` is published and routable.
    Published {
        /// The version.
        version: u64,
        /// The adapter's digest in the artifact store.
        digest: String,
        /// The step it holds.
        step: u64,
        /// Which weights the version is, independently of the server (the
        /// run, epoch, update, base and adapter digests), when the server
        /// names it.
        lineage: Option<splinter_core::revision::PolicyLineage>,
    },
    /// These groups were too far behind and dropped unapplied.
    Dropped {
        /// The groups.
        groups: Vec<String>,
    },
    /// These groups' updates are in a durable checkpoint.
    Committed {
        /// The step of the checkpoint.
        step: u64,
        /// The groups.
        groups: Vec<String>,
    },
    /// A step's telemetry (its `pg` object: staleness, policy tokens, the
    /// learner's wait, the queue).
    Step {
        /// The frame's `pg` object.
        pg: Value,
    },
    /// The job ended: `succeeded`, `failed` (with why) or `cancelled`.
    Done {
        /// Its status.
        status: String,
        /// Why it failed.
        error: Option<String>,
    },
}

/// Why a streaming job could not be used.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum StreamError {
    /// The run was restarted (or ended) under the client: what it had not
    /// committed is lost and must be offered again to the run started anew.
    #[error("the streaming run was restarted: {0}")]
    Restarted(String),
    /// The job refused what it was sent (an invalid record or group).
    #[error("the streaming run refused the records: {0}")]
    Refused(String),
    /// The job could not be created or failed.
    #[error("the streaming run failed: {0}")]
    Failed(String),
    /// The endpoint could not be reached or answered outside the API.
    #[error("the streaming run's endpoint: {0}")]
    Transport(String),
}

/// One round of asking the endpoints for a job.
enum Placed {
    /// An endpoint placed it (or the run was refused for good).
    Job(Result<StreamJob, StreamError>),
    /// Every endpoint was busy or out of reach: ask again after `wait`.
    Busy {
        runtime: tokio::runtime::Runtime,
        client: reqwest::Client,
        wait: Duration,
        refusals: Vec<String>,
    },
}

/// A running streaming job.
pub struct StreamJob {
    runtime: tokio::runtime::Runtime,
    client: reqwest::Client,
    url: String,
    key: String,
    id: String,
    epoch: u64,
    version: u64,
    publish_adapter: String,
    tokenizer_digest: Option<String>,
    frames: Receiver<Value>,
    /// Frames read while the policy was awaited, reported first by `events`.
    held: std::collections::VecDeque<Value>,
    stop: Arc<AtomicBool>,
    reader: Option<std::thread::JoinHandle<()>>,
    /// Since when the endpoint could not be reached, while it cannot.
    unreachable_since: Option<std::time::Instant>,
}

/// The HTTP client of one runtime: a client is used on the runtime that
/// built it.
fn http_client() -> Result<reqwest::Client, StreamError> {
    sven_sdk::http::model_client_builder()
        .no_proxy()
        .connect_timeout(Duration::from_secs(15))
        .build()
        .map_err(|e| StreamError::Failed(e.to_string()))
}

fn retry_after(response: &reqwest::Response) -> Duration {
    response
        .headers()
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map_or(DEFAULT_RETRY, Duration::from_secs)
}

fn keys(value: &Value) -> Vec<String> {
    value
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|k| k.as_str().map(str::to_string))
        .collect()
}

impl StreamJob {
    /// Creates the run on the first of `brain`'s endpoints that places it
    /// and waits for its first frame.
    ///
    /// # Errors
    /// No endpoint placed it, it was refused, or its first frame never came.
    ///
    /// An endpoint that is busy (429, 503) or cannot be reached is asked again, with the wait it names, for up to
    /// five minutes: a run's last job may still be letting go of the run, or the server may be restarting.
    pub fn start(brain: &BrainEndpoints, spec: &StreamSpec) -> Result<Self, StreamError> {
        let mut runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| StreamError::Failed(format!("starting a runtime: {e}")))?;
        let mut client = runtime.block_on(async { http_client() })?;
        let body =
            json!({"model": "brain/train", "action": "train", "params": spec.create_params()});
        let deadline = std::time::Instant::now() + START_WAIT;
        loop {
            match Self::place_once(runtime, client, brain, &body) {
                Placed::Job(job) => return job,
                Placed::Busy {
                    runtime: r,
                    client: c,
                    wait,
                    refusals,
                } => {
                    if std::time::Instant::now() + wait > deadline {
                        return Err(StreamError::Failed(format!(
                            "no endpoint placed the streaming run within {}s: {}",
                            START_WAIT.as_secs(),
                            refusals.join("; ")
                        )));
                    }
                    eprintln!("splinter: no endpoint took the streaming run yet ({}); asking again in {}s", refusals.join("; "), wait.as_secs_f32());
                    std::thread::sleep(wait);
                    (runtime, client) = (r, c);
                }
            }
        }
    }

    /// One round over `brain`'s endpoints: the job the first one placed, or, when every one was busy or out of
    /// reach, the wait before the next round.
    fn place_once(
        runtime: tokio::runtime::Runtime,
        client: reqwest::Client,
        brain: &BrainEndpoints,
        body: &Value,
    ) -> Placed {
        let mut refusals = Vec::new();
        let mut wait: Option<Duration> = None;
        for url in brain.urls() {
            let sent = runtime.block_on(async {
                client
                    .post(format!("{url}/jobs"))
                    .bearer_auth(brain.key())
                    .timeout(CALL)
                    .json(&body)
                    .send()
                    .await
            });
            let response = match sent {
                Ok(response) => response,
                Err(e) => {
                    refusals.push(format!("{url}: {e}"));
                    if e.is_connect() || e.is_timeout() {
                        wait = Some(wait.unwrap_or(DEFAULT_RETRY).min(DEFAULT_RETRY * 2));
                    }
                    continue;
                }
            };
            let status = response.status();
            let busy = matches!(status.as_u16(), 429 | 503).then(|| retry_after(&response));
            let text = runtime.block_on(response.text()).unwrap_or_default();
            if let Some(after) = busy {
                wait = Some(wait.map_or(after, |w| w.min(after)));
            }
            if status != reqwest::StatusCode::ACCEPTED {
                refusals.push(format!(
                    "{url} (HTTP {}): {}",
                    status.as_u16(),
                    message_of(&text)
                ));
                continue;
            }
            let id = serde_json::from_str::<Value>(&text)
                .ok()
                .and_then(|v| v["id"].as_str().map(str::to_string))
                .ok_or_else(|| StreamError::Transport(format!("{url} created a job with no id")));
            return Placed::Job(id.and_then(|id| {
                Self::follow(runtime, client, url.clone(), brain.key().to_string(), id)
            }));
        }
        if let Some(wait) = wait {
            return Placed::Busy {
                runtime,
                client,
                wait,
                refusals,
            };
        }
        Placed::Job(Err(StreamError::Failed(format!(
            "no endpoint placed the streaming run: {}",
            refusals.join("; ")
        ))))
    }

    fn follow(
        runtime: tokio::runtime::Runtime,
        client: reqwest::Client,
        url: String,
        key: String,
        id: String,
    ) -> Result<Self, StreamError> {
        let (sender, frames) = channel();
        let stop = Arc::new(AtomicBool::new(false));
        let reader = {
            let (url, key, id, stop) = (url.clone(), key.clone(), id.clone(), stop.clone());
            std::thread::Builder::new()
                .name("stream-events".into())
                .spawn(move || read_events(&url, &key, &id, &stop, &sender))
                .map_err(|e| StreamError::Failed(format!("starting the event reader: {e}")))?
        };
        let mut job = Self {
            runtime,
            client,
            url,
            key,
            id,
            epoch: 0,
            version: 0,
            publish_adapter: String::new(),
            tokenizer_digest: None,
            frames,
            held: std::collections::VecDeque::new(),
            stop,
            reader: Some(reader),
            unreachable_since: None,
        };
        // A job reports its progress (`{"step", "total"}`) and its stages
        // (`{"type": "stage"}`) before it streams: the run starts at its
        // stream frame (or ends at `done`).
        let deadline = std::time::Instant::now() + FIRST_FRAME;
        let first = loop {
            let left = deadline.saturating_duration_since(std::time::Instant::now());
            let frame = job
                .frames
                .recv_timeout(left)
                .map_err(|_| StreamError::Failed("the run sent no stream frame".into()))?;
            if frame["type"] == "stream" || frame["type"] == "done" {
                break frame;
            }
        };
        if first["type"] == "done" {
            return Err(StreamError::Failed(format!(
                "the run ended before it streamed: {}",
                first["error"]
            )));
        }
        job.epoch = first["epoch"].as_u64().unwrap_or(0);
        job.version = first["version"].as_u64().unwrap_or(0);
        job.publish_adapter = first["publish_adapter"]
            .as_str()
            .unwrap_or_default()
            .to_string();
        job.tokenizer_digest = first["tokenizer_digest"].as_str().map(str::to_owned);
        Ok(job)
    }

    /// `sha256:<hex>` of the base's tokenizer, as the run's `stream` frame
    /// names it; `None` when the server does not.
    #[must_use]
    pub fn tokenizer_digest(&self) -> Option<&str> {
        self.tokenizer_digest.as_deref()
    }

    /// The run's epoch: one more for every job that continued it.
    #[must_use]
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// The version published when the job started, or last offered back.
    #[must_use]
    pub fn version(&self) -> u64 {
        self.version
    }

    /// The id the run serves its versions under.
    #[must_use]
    pub fn publish_adapter(&self) -> &str {
        &self.publish_adapter
    }

    fn refusal(&self, status: u16, text: &str) -> StreamError {
        let why = format!("HTTP {status}: {}", message_of(text));
        match status {
            400 => StreamError::Refused(why),
            404 | 409 | 410 => StreamError::Restarted(why),
            _ => StreamError::Transport(format!("{}: {why}", self.url)),
        }
    }

    /// Offers `records`, JSON lines of complete groups.
    ///
    /// # Errors
    /// The records are refused, the run restarted, or the endpoint failed.
    pub fn offer(&mut self, records: &str) -> Result<Offered, StreamError> {
        let url = format!("{}/jobs/{}/records?epoch={}", self.url, self.id, self.epoch);
        let sent = self.runtime.block_on(async {
            self.client
                .post(url)
                .bearer_auth(&self.key)
                .timeout(CALL)
                .header(reqwest::header::CONTENT_TYPE, "application/x-ndjson")
                .body(records.to_string())
                .send()
                .await
        });
        let response = match sent {
            Ok(response) => {
                self.unreachable_since = None;
                response
            }
            // No answer: the server may be restarting, or it closed a connection the client reused. An offer is
            // idempotent (the run takes a group once, by its key), so it is made again until the server has been
            // out of reach for long; once it is back, the job's events say whether the run went with it.
            Err(e) => {
                let since = *self
                    .unreachable_since
                    .get_or_insert_with(std::time::Instant::now);
                if since.elapsed() > START_WAIT {
                    return Err(StreamError::Transport(format!(
                        "{}: no answer for {}s: {}",
                        self.url,
                        START_WAIT.as_secs(),
                        with_sources(&e)
                    )));
                }
                return Ok(Offered::NotReady {
                    retry_after: DEFAULT_RETRY,
                });
            }
        };
        let status = response.status().as_u16();
        let wait = retry_after(&response);
        let text = self.runtime.block_on(response.text()).unwrap_or_default();
        let body: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
        match status {
            200 => {
                if body["epoch"].as_u64().is_some_and(|e| e != self.epoch) {
                    return Err(StreamError::Restarted(format!(
                        "the run answered in epoch {}, not {}",
                        body["epoch"], self.epoch
                    )));
                }
                if let Some(version) = body["version"].as_u64() {
                    self.version = version;
                }
                Ok(Offered::Taken {
                    accepted: keys(&body["accepted"]),
                    duplicates: keys(&body["duplicates"]),
                    allowance: body["allowance"].as_u64().unwrap_or(0),
                    version: self.version,
                })
            }
            429 => Ok(Offered::Full {
                allowance: body["detail"]["allowance"].as_u64().unwrap_or(0),
                retry_after: wait,
            }),
            503 => Ok(Offered::NotReady { retry_after: wait }),
            other => Err(self.refusal(other, &text)),
        }
    }

    /// What the run reported since the last call, in order; never blocks.
    ///
    /// # Errors
    /// The run restarted (a `stream` frame of another epoch).
    pub fn events(&mut self) -> Result<Vec<StreamEvent>, StreamError> {
        let mut events = Vec::new();
        let held: Vec<Value> = self.held.drain(..).collect();
        for frame in held.into_iter().chain(self.frames.try_iter()) {
            let groups = || keys(&frame["groups"]);
            let number = |key: &str| frame[key].as_u64().unwrap_or(0);
            match frame["type"].as_str().unwrap_or_default() {
                "stream" if number("epoch") != self.epoch => {
                    return Err(StreamError::Restarted(format!(
                        "the run streams epoch {}, not {}",
                        frame["epoch"], self.epoch
                    )))
                }
                "applied" => events.push(StreamEvent::Applied {
                    step: number("step"),
                    groups: groups(),
                }),
                "published" => {
                    self.version = self.version.max(number("version"));
                    let lineage = match frame.get("lineage").filter(|l| !l.is_null()) {
                        Some(lineage) => {
                            Some(serde_json::from_value(lineage.clone()).map_err(|e| {
                                StreamError::Failed(format!("a published frame's lineage: {e}"))
                            })?)
                        }
                        None => None,
                    };
                    events.push(StreamEvent::Published {
                        version: number("version"),
                        digest: frame["digest"].as_str().unwrap_or_default().to_string(),
                        step: number("step"),
                        lineage,
                    });
                }
                "dropped" => events.push(StreamEvent::Dropped { groups: groups() }),
                "committed" => events.push(StreamEvent::Committed {
                    step: number("step"),
                    groups: groups(),
                }),
                "step" => events.push(StreamEvent::Step {
                    pg: frame["pg"].clone(),
                }),
                "done" => events.push(StreamEvent::Done {
                    status: frame["status"].as_str().unwrap_or_default().to_string(),
                    error: frame["error"]["message"]
                        .as_str()
                        .or(frame["error"].as_str())
                        .map(str::to_string),
                }),
                "forgotten" => {
                    return Err(StreamError::Restarted(
                        "the server no longer knows the job".into(),
                    ))
                }
                _ => {}
            }
        }
        Ok(events)
    }
}

impl Drop for StreamJob {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }
}

/// Reads the job's events into `sender` until the job is done, the server
/// forgets it, or `stop` is set; a stream that ends is reopened after the
/// last event seen. A frame is its JSON; the end is `{"type": "done", ...}`;
/// a job the server does not know is `{"type": "forgotten"}`.
fn read_events(
    url: &str,
    key: &str,
    id: &str,
    stop: &AtomicBool,
    sender: &std::sync::mpsc::Sender<Value>,
) {
    let Ok(runtime) = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    else {
        return;
    };
    let Ok(client) = runtime.block_on(async { http_client() }) else {
        return;
    };
    let mut last: Option<u64> = None;
    while !stop.load(Ordering::SeqCst) {
        let ended = runtime.block_on(async {
            let mut request = client
                .get(format!("{url}/jobs/{id}/events"))
                .bearer_auth(key);
            if let Some(seen) = last {
                request = request.header("last-event-id", seen.to_string());
            }
            let Ok(mut response) = request.send().await else {
                return false;
            };
            if response.status() == reqwest::StatusCode::NOT_FOUND
                || response.status() == reqwest::StatusCode::GONE
            {
                let _ = sender.send(json!({"type": "forgotten"}));
                return true;
            }
            let mut parser = EventParser::default();
            loop {
                if stop.load(Ordering::SeqCst) {
                    return true;
                }
                let chunk = match tokio::time::timeout(REOPEN * 5, response.chunk()).await {
                    Err(_) => continue,
                    Ok(Ok(Some(chunk))) => chunk,
                    Ok(_) => return false,
                };
                for event in parser.feed(&chunk) {
                    if let Some(id) = event.id {
                        last = Some(id);
                    }
                    let data: Value = serde_json::from_str(&event.data).unwrap_or(Value::Null);
                    let frame = if event.name == "done" {
                        let mut done = data;
                        done["type"] = json!("done");
                        done
                    } else {
                        data
                    };
                    let is_done = frame["type"] == "done";
                    if sender.send(frame).is_err() || is_done {
                        return true;
                    }
                }
            }
        });
        if ended {
            return;
        }
        std::thread::sleep(REOPEN);
    }
}
