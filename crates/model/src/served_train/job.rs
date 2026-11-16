// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems that run their training as
// a job on a shared model service instead of in their own process, for its
// clients. If your team needs expertise in driving long training jobs over a
// service API with progress, cancellation and recovery, you can procure our
// services by sending an email to info@swedishembedded.com.

//! One brain job over HTTP: create it (`POST /jobs`), follow it as server-sent
//! events (`GET /jobs/{id}/events`), fetch its result, cancel it (`DELETE`).
//!
//! * **Refusal is a wait.** brain places a training run on the cards serving
//!   shares and refuses it (HTTP 429) when every card that could hold it is
//!   busy; the run is then asked of the next endpoint, and of all of them
//!   again after [`REFUSED_WAIT`], until it is placed or the caller cancels.
//!   Any other refusal (a request that could never run) ends the job with
//!   brain's reason.
//! * **The stream is resumed, not trusted.** When the connection ends before
//!   the job's `done` event, it is reopened with `Last-Event-ID`, so no step is
//!   missed and none is reported twice, up to [`RECONNECTS`] times in a row.
//! * **Cancelling is prompt.** The caller's token is polled every
//!   [`POLL`]; when it fires the job is cancelled on the server (it stops at
//!   its next optimiser step) and the call returns.
//!
//! The work happens on a thread of its own with its own runtime, so the
//! caller may be inside an async runtime; the step frames come back over a
//! channel and are handed to the caller's closure on the caller's thread.

use std::sync::mpsc::{channel, Sender};
use std::time::Duration;

use serde_json::Value;

/// How often the caller's cancel token is read.
const POLL: Duration = Duration::from_millis(250);

/// How long every endpoint is left alone after all of them refused a run for
/// want of room.
pub const REFUSED_WAIT: Duration = Duration::from_secs(30);

/// How many times in a row a broken event stream is reopened before the job
/// is given up on.
pub const RECONNECTS: u32 = 20;

/// How long a connection may be silent: brain sends a comment every 15
/// seconds.
const SILENCE: Duration = Duration::from_secs(120);

/// How long a short call (status, result, cancel) may take.
const SHORT_CALL: Duration = Duration::from_secs(60);

/// Why a job did not produce a result.
#[derive(Debug, PartialEq, Eq)]
pub enum JobError {
    /// The caller's cancel fired, or the server cancelled the job.
    Cancelled,
    /// The job ran or was refused for a reason brain gave.
    Failed(String),
    /// An endpoint could not be reached or answered something else than the
    /// API: its URL and what happened.
    Transport(String, String),
}

/// One server-sent event.
#[derive(Debug, PartialEq, Eq)]
pub struct Event {
    /// The `id:` field: the frame's sequence number.
    pub id: Option<u64>,
    /// The `event:` field.
    pub name: String,
    /// The `data:` field.
    pub data: String,
}

/// Splits a stream of bytes into [`Event`]s. Comments (keep-alives) and
/// fields it does not know are dropped; a line ends with `\n` or `\r\n`.
#[derive(Default)]
pub struct EventParser {
    buffer: String,
}

impl EventParser {
    /// Adds `chunk` and returns the events it completed.
    pub fn feed(&mut self, chunk: &[u8]) -> Vec<Event> {
        self.buffer.push_str(&String::from_utf8_lossy(chunk));
        self.buffer = self.buffer.replace("\r\n", "\n");
        let mut events = Vec::new();
        while let Some(end) = self.buffer.find("\n\n") {
            let block: String = self.buffer.drain(..end + 2).collect();
            let (mut id, mut name, mut data) = (None, String::new(), Vec::new());
            for line in block.lines() {
                if let Some(value) = line.strip_prefix("id:") {
                    id = value.trim().parse().ok();
                } else if let Some(value) = line.strip_prefix("event:") {
                    name = value.trim().to_string();
                } else if let Some(value) = line.strip_prefix("data:") {
                    data.push(value.strip_prefix(' ').unwrap_or(value).to_string());
                }
            }
            if !name.is_empty() {
                events.push(Event {
                    id,
                    name,
                    data: data.join("\n"),
                });
            }
        }
        events
    }
}

/// The cancel token a job watches.
pub trait Cancel: Sync {
    /// Whether the caller wants the job stopped.
    fn cancelled(&self) -> bool;
}

impl Cancel for sven_sdk::CancelToken {
    fn cancelled(&self) -> bool {
        self.is_cancelled()
    }
}

/// Runs the `brain/train` job `action` with `params` on the first of `urls` that places
/// it, calling `on_frame` with each structured frame (a training step) as it
/// arrives, and returns the job's report (`outputs` of its result).
pub fn run(
    urls: &[String],
    key: &str,
    refused_wait: Duration,
    first: usize,
    (action, params): (&str, &Value),
    cancel: Option<&dyn Cancel>,
    on_frame: &mut dyn FnMut(&Value),
) -> Result<Value, JobError> {
    let (frames, received) = channel::<Value>();
    std::thread::scope(|scope| {
        let worker = scope.spawn(move || {
            drive(
                urls,
                key,
                refused_wait,
                first,
                (action, params),
                cancel,
                &frames,
            )
        });
        for frame in received {
            on_frame(&frame);
        }
        worker
            .join()
            .unwrap_or_else(|_| Err(JobError::Failed("the job client panicked".into())))
    })
}

fn drive(
    urls: &[String],
    key: &str,
    refused_wait: Duration,
    first: usize,
    (action, params): (&str, &Value),
    cancel: Option<&dyn Cancel>,
    frames: &Sender<Value>,
) -> Result<Value, JobError> {
    // The provider sven's own clients install; a second install (or sven's
    // first) leaves the one in place.
    let _ = rustls::crypto::ring::default_provider().install_default();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| JobError::Failed(format!("starting a runtime: {e}")))?;
    runtime.block_on(async {
        let client = reqwest::Client::builder()
            .no_proxy()
            .connect_timeout(Duration::from_secs(15))
            .build()
            .map_err(|e| JobError::Failed(e.to_string()))?;
        let api = Api {
            client,
            key,
            cancel,
        };
        let (url, id) = api
            .create(urls, refused_wait, first, action, params)
            .await?;
        match api.follow(&url, &id, frames).await {
            Ok(()) => api.result(&url, &id).await,
            Err(JobError::Cancelled) if cancel.is_some_and(Cancel::cancelled) => {
                api.cancel(&url, &id).await;
                Err(JobError::Cancelled)
            }
            Err(other) => Err(other),
        }
    })
}

struct Api<'a> {
    client: reqwest::Client,
    key: &'a str,
    cancel: Option<&'a dyn Cancel>,
}

impl Api<'_> {
    fn cancelled(&self) -> bool {
        self.cancel.is_some_and(Cancel::cancelled)
    }

    /// `future`, abandoned with [`JobError::Cancelled`] as soon as the caller
    /// cancels.
    async fn unless_cancelled<T>(
        &self,
        future: impl std::future::Future<Output = T>,
    ) -> Result<T, JobError> {
        tokio::pin!(future);
        loop {
            tokio::select! {
                value = &mut future => return Ok(value),
                () = tokio::time::sleep(POLL) => {
                    if self.cancelled() {
                        return Err(JobError::Cancelled);
                    }
                }
            }
        }
    }

    /// Creates the job on the first endpoint that places it; returns that
    /// endpoint and the job's id.
    async fn create(
        &self,
        urls: &[String],
        refused_wait: Duration,
        first: usize,
        action: &str,
        params: &Value,
    ) -> Result<(String, String), JobError> {
        let body = serde_json::json!({
            "model": "brain/train",
            "action": action,
            "params": params,
        });
        loop {
            let mut unreachable = Vec::new();
            for offset in 0..urls.len() {
                let url = &urls[(first + offset) % urls.len()];
                // brain answers once the run is placed or refused, which can
                // take its whole admission deadline: no total timeout, the
                // caller's cancel instead.
                let sent = self
                    .unless_cancelled(
                        self.client
                            .post(format!("{url}/jobs"))
                            .bearer_auth(self.key)
                            .json(&body)
                            .send(),
                    )
                    .await?;
                let response = match sent {
                    Ok(response) => response,
                    Err(e) => {
                        unreachable.push((url.clone(), e.to_string()));
                        continue;
                    }
                };
                let status = response.status();
                let text = response.text().await.unwrap_or_default();
                if status == reqwest::StatusCode::ACCEPTED {
                    let id = serde_json::from_str::<Value>(&text)
                        .ok()
                        .and_then(|v| v["id"].as_str().map(str::to_string))
                        .ok_or_else(|| {
                            JobError::Transport(url.clone(), "created a job with no id".into())
                        })?;
                    return Ok((url.clone(), id));
                }
                if status != reqwest::StatusCode::TOO_MANY_REQUESTS {
                    return Err(JobError::Failed(format!(
                        "{url} refused the job (HTTP {}): {}",
                        status.as_u16(),
                        message_of(&text)
                    )));
                }
            }
            if unreachable.len() == urls.len() {
                let (url, reason) = unreachable.swap_remove(0);
                return Err(JobError::Transport(url, reason));
            }
            // Every endpoint that answered said it had no room: wait for
            // serving to go quiet.
            self.unless_cancelled(tokio::time::sleep(refused_wait))
                .await?;
        }
    }

    /// Follows the job's events until its `done` event; each structured
    /// frame goes to `frames`. Returns when the job succeeded; its failure or
    /// cancellation is the error.
    async fn follow(&self, url: &str, id: &str, frames: &Sender<Value>) -> Result<(), JobError> {
        let (mut last, mut broken) = (None::<u64>, 0u32);
        loop {
            match self
                .read_events(url, id, &mut last, &mut broken, frames)
                .await?
            {
                Step::Ended(Ended::Succeeded) => return Ok(()),
                Step::Ended(Ended::Cancelled) => return Err(JobError::Cancelled),
                Step::Ended(Ended::Failed(why)) => return Err(JobError::Failed(why)),
                Step::Forgotten => {
                    return Err(JobError::Failed(format!(
                        "{url} no longer knows job {id} (it restarted, or its record expired)"
                    )))
                }
                Step::Reopen(reason) => {
                    broken += 1;
                    if broken > RECONNECTS {
                        return Err(JobError::Transport(url.to_string(), reason));
                    }
                }
            }
            self.unless_cancelled(tokio::time::sleep(Duration::from_secs(2)))
                .await?;
        }
    }

    /// One connection to the job's event stream, resumed after `last`.
    async fn read_events(
        &self,
        url: &str,
        id: &str,
        last: &mut Option<u64>,
        broken: &mut u32,
        frames: &Sender<Value>,
    ) -> Result<Step, JobError> {
        // A partial event of an earlier connection is not the start of this one's.
        let mut parser = EventParser::default();
        let mut request = self
            .client
            .get(format!("{url}/jobs/{id}/events"))
            .bearer_auth(self.key);
        if let Some(seen) = last {
            request = request.header("last-event-id", seen.to_string());
        }
        let mut response = match self.unless_cancelled(request.send()).await? {
            Ok(response) => response,
            Err(e) => return Ok(Step::Reopen(e.to_string())),
        };
        match response.status() {
            status if status.is_success() => {}
            reqwest::StatusCode::NOT_FOUND => return Ok(Step::Forgotten),
            status => return Ok(Step::Reopen(format!("HTTP {}", status.as_u16()))),
        }
        loop {
            let read = self
                .unless_cancelled(tokio::time::timeout(SILENCE, response.chunk()))
                .await?;
            let chunk = match read {
                Err(_elapsed) => return Ok(Step::Reopen("the stream went silent".into())),
                Ok(Err(e)) => return Ok(Step::Reopen(e.to_string())),
                // The server ended the stream without a `done` event.
                Ok(Ok(None)) => return Ok(Step::Reopen("the stream ended early".into())),
                Ok(Ok(Some(chunk))) => chunk,
            };
            *broken = 0;
            for event in parser.feed(&chunk) {
                *last = event.id.or(*last);
                if let Some(ended) = handle(&event, frames) {
                    return Ok(Step::Ended(ended));
                }
            }
        }
    }

    async fn result(&self, url: &str, id: &str) -> Result<Value, JobError> {
        let response = self
            .client
            .get(format!("{url}/jobs/{id}/result"))
            .bearer_auth(self.key)
            .timeout(SHORT_CALL)
            .send()
            .await
            .map_err(|e| JobError::Transport(url.to_string(), e.to_string()))?;
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(JobError::Failed(format!(
                "{url} gave no result for job {id} (HTTP {}): {}",
                status.as_u16(),
                message_of(&text)
            )));
        }
        let mut result: Value = serde_json::from_str(&text)
            .map_err(|e| JobError::Transport(url.to_string(), format!("the result: {e}")))?;
        Ok(result["outputs"].take())
    }

    /// Asks brain to stop the job; it stops at its next optimiser step.
    async fn cancel(&self, url: &str, id: &str) {
        let _ = self
            .client
            .delete(format!("{url}/jobs/{id}"))
            .bearer_auth(self.key)
            .timeout(SHORT_CALL)
            .send()
            .await;
    }
}

/// How a job ended.
#[derive(Debug, PartialEq, Eq)]
enum Ended {
    Succeeded,
    Cancelled,
    /// With the reason brain gave.
    Failed(String),
}

/// What reading the event stream came to.
enum Step {
    Ended(Ended),
    /// The server does not know the job any more.
    Forgotten,
    /// The connection broke or ended early, for this reason: open it again.
    Reopen(String),
}

/// What one event means: `None` while the job goes on.
fn handle(event: &Event, frames: &Sender<Value>) -> Option<Ended> {
    let data: Value = serde_json::from_str(&event.data).unwrap_or(Value::Null);
    match event.name.as_str() {
        "frame" => {
            let _ = frames.send(data);
            None
        }
        "done" => Some(match data["status"].as_str() {
            Some("succeeded") => Ended::Succeeded,
            Some("cancelled") => Ended::Cancelled,
            Some("failed") => Ended::Failed(
                data["error"]["message"]
                    .as_str()
                    .unwrap_or("it failed")
                    .to_string(),
            ),
            other => Ended::Failed(format!("it ended with the unknown status {other:?}")),
        }),
        _ => None,
    }
}

/// The message of brain's error body, else the body.
fn message_of(body: &str) -> String {
    serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| v["error"]["message"].as_str().map(str::to_string))
        .unwrap_or_else(|| body.chars().take(300).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_are_assembled_across_chunks_and_keep_alives_are_ignored() {
        let mut parser = EventParser::default();
        assert!(parser.feed(b": keep-alive\n\nid: 1\nevent: fra").is_empty());
        let events = parser.feed(b"me\ndata: {\"step\":1}\n\nid: 2\r\nevent: done\r\ndata: {\"status\":\"succeeded\"}\r\n\r\n");
        assert_eq!(
            events,
            vec![
                Event {
                    id: Some(1),
                    name: "frame".into(),
                    data: "{\"step\":1}".into()
                },
                Event {
                    id: Some(2),
                    name: "done".into(),
                    data: "{\"status\":\"succeeded\"}".into()
                },
            ]
        );
    }

    #[test]
    fn a_done_event_decides_how_the_job_ended() {
        let (frames, _kept) = channel();
        let done = |status: &str, error: &str| {
            handle(
                &Event {
                    id: Some(9),
                    name: "done".into(),
                    data: format!(
                        "{{\"status\":\"{status}\",\"error\":{{\"message\":\"{error}\"}}}}"
                    ),
                },
                &frames,
            )
        };
        assert_eq!(done("succeeded", ""), Some(Ended::Succeeded));
        assert_eq!(done("cancelled", ""), Some(Ended::Cancelled));
        assert_eq!(
            done("failed", "no room"),
            Some(Ended::Failed("no room".into()))
        );
    }
}
