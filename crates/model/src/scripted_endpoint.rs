// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems that reach their model
// only through a serving API, for its clients. If your team needs expertise
// in testing agents against OpenAI-compatible inference services, you can
// procure our services by sending an email to info@swedishembedded.com.

//! A scripted stand-in for a `brain serve --openai` endpoint, for specs that
//! exercise the client side of the API without a model or a device.
//!
//! It lists the models it is given (`GET /v1/models`, with the adapter digest
//! a brain card carries), answers chat completions as a stream, refuses a
//! model it does not list as brain does, and records what it saw: the model,
//! the seed and sampling of each request, and the most requests ever in
//! flight together. A request takes [`ScriptedEndpoint::with_delay`] to
//! answer, so requests asked together overlap and the peak shows it. An
//! endpoint may follow a release manifest, listing the adapter the manifest
//! names, as `brain serve --adapter-manifest` does.

// Test support: a listener that cannot be bound is the failure report.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// The key a scripted endpoint accepts.
pub const SCRIPTED_KEY: &str = "sk-scripted";

/// One model the endpoint lists: a base, or an adapter named by its digest.
#[derive(Clone, Debug)]
pub struct Card {
    /// The id requests name it by.
    pub id: String,
    /// `sha256:<hex>` of the adapter file, for an adapter card.
    pub adapter_digest: Option<String>,
}

impl Card {
    /// A base model.
    #[must_use]
    pub fn base(id: &str) -> Self {
        Self {
            id: id.into(),
            adapter_digest: None,
        }
    }

    /// An adapter served as `id`, whose file hashes to `digest`.
    #[must_use]
    pub fn adapter(id: &str, digest: &str) -> Self {
        Self {
            id: id.into(),
            adapter_digest: Some(digest.into()),
        }
    }
}

/// One chat request as the endpoint saw it.
#[derive(Clone, Debug, Default)]
pub struct Seen {
    /// The model it named.
    pub model: String,
    /// The text of its last user message.
    pub last_user: String,
    /// Its `seed`, if it carried one.
    pub seed: Option<u64>,
    /// Its `temperature`, if it carried one.
    pub temperature: Option<f64>,
    /// Its `top_k`, if it carried one.
    pub top_k: Option<u64>,
    /// The bearer token it presented.
    pub bearer: String,
}

type Reply = dyn Fn(&Seen) -> String + Send + Sync;

struct Shared {
    models: Mutex<Vec<Card>>,
    manifest: Mutex<Option<(PathBuf, String)>>,
    delay: Mutex<Duration>,
    reply: Mutex<Arc<Reply>>,
    in_flight: AtomicUsize,
    peak: AtomicUsize,
    seen: Mutex<Vec<Seen>>,
    stop: AtomicBool,
}

/// A running scripted endpoint, stopped when dropped.
pub struct ScriptedEndpoint {
    port: u16,
    shared: Arc<Shared>,
}

impl ScriptedEndpoint {
    /// An endpoint listing `models` on a free loopback port.
    #[must_use]
    pub fn start(models: Vec<Card>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("a loopback port");
        listener.set_nonblocking(true).unwrap();
        let port = listener.local_addr().unwrap().port();
        let shared = Arc::new(Shared {
            models: Mutex::new(models),
            manifest: Mutex::new(None),
            delay: Mutex::new(Duration::ZERO),
            reply: Mutex::new(Arc::new(|_: &Seen| "an answer".to_string())),
            in_flight: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            seen: Mutex::new(Vec::new()),
            stop: AtomicBool::new(false),
        });
        let serving = Arc::clone(&shared);
        std::thread::spawn(move || {
            while !serving.stop.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        let shared = Arc::clone(&serving);
                        std::thread::spawn(move || handle(stream, &shared));
                    }
                    Err(_) => std::thread::sleep(Duration::from_millis(2)),
                }
            }
        });
        Self { port, shared }
    }

    /// Every request takes `delay` to answer.
    #[must_use]
    pub fn with_delay(self, delay: Duration) -> Self {
        *self.shared.delay.lock().unwrap() = delay;
        self
    }

    /// Answers each request with what `reply` makes of it.
    #[must_use]
    pub fn with_reply(self, reply: impl Fn(&Seen) -> String + Send + Sync + 'static) -> Self {
        *self.shared.reply.lock().unwrap() = Arc::new(reply);
        self
    }

    /// Follows the release manifest at `path`, as `brain serve
    /// --adapter-manifest` does: while the file exists, the adapter whose
    /// digest it states is listed as `id`.
    #[must_use]
    pub fn following(self, path: PathBuf, id: &str) -> Self {
        *self.shared.manifest.lock().unwrap() = Some((path, id.to_string()));
        self
    }

    /// The base URL a client names: `http://127.0.0.1:<port>/v1`.
    #[must_use]
    pub fn url(&self) -> String {
        format!("http://127.0.0.1:{}/v1", self.port)
    }

    /// The most chat requests that were in flight together.
    #[must_use]
    pub fn peak_in_flight(&self) -> usize {
        self.shared.peak.load(Ordering::SeqCst)
    }

    /// Every chat request answered or refused so far, in arrival order.
    #[must_use]
    pub fn seen(&self) -> Vec<Seen> {
        self.shared.seen.lock().unwrap().clone()
    }
}

impl Drop for ScriptedEndpoint {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::SeqCst);
    }
}

/// The cards the endpoint lists now: its models, and the adapter its
/// manifest names.
fn listed(shared: &Shared) -> Vec<Card> {
    let mut cards = shared.models.lock().unwrap().clone();
    if let Some((path, id)) = shared.manifest.lock().unwrap().as_ref() {
        if let Ok(text) = std::fs::read_to_string(path) {
            if let Some(digest) = serde_json::from_str::<serde_json::Value>(&text)
                .ok()
                .and_then(|m| m["digest"].as_str().map(str::to_string))
            {
                cards.push(Card::adapter(id, &digest));
            }
        }
    }
    cards
}

fn handle(mut stream: TcpStream, shared: &Shared) {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).unwrap_or(0) == 0 {
        return;
    }
    let mut length = 0usize;
    let mut bearer = String::new();
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
            break;
        }
        let lower = line.to_ascii_lowercase();
        if let Some(value) = lower.strip_prefix("content-length:") {
            length = value.trim().parse().unwrap_or(0);
        }
        if lower.starts_with("authorization:") {
            bearer = line["authorization:".len()..]
                .trim()
                .trim_start_matches("Bearer ")
                .to_string();
        }
    }
    let mut body = vec![0u8; length];
    reader.read_exact(&mut body).unwrap();
    if bearer != SCRIPTED_KEY {
        let _ = respond(&mut stream, "401 Unauthorized", "application/json", "{}");
        return;
    }
    if request_line.starts_with("GET /v1/models") {
        let data: Vec<serde_json::Value> = listed(shared)
            .iter()
            .map(|card| {
                let mut value = serde_json::json!({
                    "id": card.id, "object": "model", "owned_by": "brain", "context_length": 32768
                });
                if let Some(digest) = &card.adapter_digest {
                    value["adapter"] = serde_json::json!({ "id": card.id, "digest": digest });
                }
                value
            })
            .collect();
        let list = serde_json::json!({ "object": "list", "data": data }).to_string();
        let _ = respond(&mut stream, "200 OK", "application/json", &list);
    } else if request_line.starts_with("POST /v1/chat/completions") {
        chat(&mut stream, shared, &body, bearer);
    } else {
        let _ = respond(&mut stream, "404 Not Found", "application/json", "{}");
    }
}

fn chat(stream: &mut TcpStream, shared: &Shared, body: &[u8], bearer: String) {
    let request: serde_json::Value = serde_json::from_slice(body).unwrap_or_default();
    let last_user = request["messages"]
        .as_array()
        .and_then(|m| m.iter().rev().find(|m| m["role"] == "user"))
        .map(|m| match &m["content"] {
            serde_json::Value::String(text) => text.clone(),
            other => other.to_string(),
        })
        .unwrap_or_default();
    let seen = Seen {
        model: request["model"].as_str().unwrap_or_default().to_string(),
        last_user,
        seed: request["seed"].as_u64(),
        temperature: request["temperature"].as_f64(),
        top_k: request["top_k"].as_u64(),
        bearer,
    };
    shared.seen.lock().unwrap().push(seen.clone());
    let now = shared.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
    shared.peak.fetch_max(now, Ordering::SeqCst);
    let delay = *shared.delay.lock().unwrap();
    std::thread::sleep(delay);
    let known = listed(shared).iter().any(|card| card.id == seen.model);
    if known {
        let reply = Arc::clone(&shared.reply.lock().unwrap());
        let text = reply(&seen);
        if request["stream"].as_bool().unwrap_or(false) {
            let chunks = [
                serde_json::json!({"choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": null}]}),
                serde_json::json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
            ];
            let mut sse = String::new();
            for chunk in chunks {
                sse.push_str(&format!("data: {chunk}\n\n"));
            }
            sse.push_str("data: [DONE]\n\n");
            let _ = respond(stream, "200 OK", "text/event-stream", &sse);
        } else {
            let whole = serde_json::json!({"choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]});
            let _ = respond(stream, "200 OK", "application/json", &whole.to_string());
        }
    } else {
        let refusal = serde_json::json!({"error": {"message": format!("unknown model {}", seen.model), "code": "model_not_found"}});
        let _ = respond(
            stream,
            "404 Not Found",
            "application/json",
            &refusal.to_string(),
        );
    }
    shared.in_flight.fetch_sub(1, Ordering::SeqCst);
}

fn respond(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &str,
) -> std::io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status}\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    )?;
    stream.flush()
}
