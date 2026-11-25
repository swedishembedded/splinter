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
//! names, as `brain serve --adapter-manifest` does, or serve the adapters in
//! a directory, as `--serve-adapters` does. It speaks and hears as the speech
//! doubles do (`speech::scripted`): `/v1/audio/speech` writes the text into
//! the samples of a 16-bit WAV, and `/v1/audio/transcriptions` reads it back.

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
    /// Whether it asked for brain's policy trace (`policy_trace: true`).
    pub policy_trace: bool,
    /// Whether it asked for the sampler's log-probabilities (`logprobs`).
    pub sampler_logprobs: bool,
}

type Reply = dyn Fn(&Seen) -> String + Send + Sync;

struct Shared {
    models: Mutex<Vec<Card>>,
    manifest: Mutex<Option<(PathBuf, String)>>,
    adapters_dir: Mutex<Option<PathBuf>>,
    embeddings: Mutex<Vec<serde_json::Value>>,
    runs: Mutex<Vec<serde_json::Value>>,
    speech: Mutex<Vec<(String, serde_json::Value)>>,
    delay: Mutex<Duration>,
    reply: Mutex<Arc<Reply>>,
    jobs: crate::scripted_jobs::Jobs,
    in_flight: AtomicUsize,
    peak: AtomicUsize,
    seen: Mutex<Vec<Seen>>,
    stop: AtomicBool,
    server: Mutex<serde_json::Value>,
    capabilities: Mutex<serde_json::Value>,
}

/// The `server` object a scripted endpoint reports by default: a brain build
/// that checks the training-record contract this build of Splinter writes.
#[must_use]
pub fn default_server() -> serde_json::Value {
    serde_json::json!({
        "engine": {"version": "6.0.0", "commit": "0123456789abcdef0123456789abcdef01234567", "dirty": false},
        "backend": "cuda",
        "training_records": splinter_core::revision::RecordContract::pinned(),
    })
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
            adapters_dir: Mutex::new(None),
            embeddings: Mutex::new(Vec::new()),
            runs: Mutex::new(Vec::new()),
            speech: Mutex::new(Vec::new()),
            delay: Mutex::new(Duration::ZERO),
            reply: Mutex::new(Arc::new(|_: &Seen| "an answer".to_string())),
            jobs: crate::scripted_jobs::Jobs::default(),
            in_flight: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            seen: Mutex::new(Vec::new()),
            stop: AtomicBool::new(false),
            server: Mutex::new(default_server()),
            capabilities: Mutex::new(serde_json::json!([])),
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

    /// Report `server` as `GET /v1/capabilities`'s `server` object; `null`
    /// leaves the object out, as a brain build older than it does.
    #[must_use]
    pub fn with_server(self, server: serde_json::Value) -> Self {
        *self.shared.server.lock().unwrap() = server;
        self
    }

    /// List `actions` as `GET /v1/capabilities`'s `data` (`[{"model",
    /// "action", "params": [{"name"}]}]`); none by default.
    #[must_use]
    pub fn with_capabilities(self, actions: serde_json::Value) -> Self {
        *self.shared.capabilities.lock().unwrap() = actions;
        self
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

    /// Serves the adapters written to `directory`, as `brain serve
    /// --serve-adapters` does: each visible `<hex>.safetensors` is listed as
    /// its own model, `adapter-<hex prefix>`, with the SHA-256 of its bytes
    /// as its digest. Hidden files are ignored.
    #[must_use]
    pub fn serving_adapters(self, directory: PathBuf) -> Self {
        *self.shared.adapters_dir.lock().unwrap() = Some(directory);
        self
    }

    /// Serves training jobs as `script` says (`POST /v1/jobs` and the rest of
    /// brain's job routes), over the artifact store the script names.
    #[must_use]
    pub fn with_training(self, script: crate::scripted_jobs::JobScript) -> Self {
        *self.shared.jobs.script.lock().unwrap() = Some(script);
        self
    }

    /// What the training service saw.
    #[must_use]
    pub fn jobs(&self) -> &crate::scripted_jobs::Jobs {
        &self.shared.jobs
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

    /// The body of every embeddings request so far, in arrival order. Each
    /// text is answered with the vector `[its length in chars, 1]`, and the
    /// items come back in reverse order with their `index`, as a server that
    /// batches may return them.
    #[must_use]
    pub fn embeddings_asked(&self) -> Vec<serde_json::Value> {
        self.shared.embeddings.lock().unwrap().clone()
    }

    /// Every `POST /v1/run` call so far, in arrival order: a `score` is
    /// answered with each supervised answer's characters as its tokens, at a
    /// log-probability of -0.5 each.
    #[must_use]
    pub fn runs_asked(&self) -> Vec<serde_json::Value> {
        self.shared.runs.lock().unwrap().clone()
    }

    /// Every speech request so far, in arrival order: the route
    /// (`transcriptions` or `speech`) and the body it carried.
    #[must_use]
    pub fn speech_asked(&self) -> Vec<(String, serde_json::Value)> {
        self.shared.speech.lock().unwrap().clone()
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

/// The digest a streaming run's served policy is listed with (the double
/// lists the id, not a version's bytes).
const SERVED_POLICY_DIGEST: &str =
    "sha256:5e00000000000000000000000000000000000000000000000000000000000000";

/// The cards the endpoint lists now: its models, and the adapter its
/// manifest names, and the policy a streaming run serves.
fn listed(shared: &Shared) -> Vec<Card> {
    let mut cards = shared.models.lock().unwrap().clone();
    if let Some(policy) = shared.jobs.stream_runs.served_policy() {
        cards.push(Card::adapter(&policy, SERVED_POLICY_DIGEST));
    }
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
    if let Some(directory) = shared.adapters_dir.lock().unwrap().as_ref() {
        let mut names: Vec<_> = std::fs::read_dir(directory)
            .into_iter()
            .flatten()
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| {
                path.extension().is_some_and(|e| e == "safetensors")
                    && !path
                        .file_name()
                        .is_some_and(|n| n.to_string_lossy().starts_with('.'))
            })
            .collect();
        names.sort();
        for path in names {
            if let Ok(file) = std::fs::File::open(&path) {
                if let Ok(digest) = splinter_core::digest::Digest::sha256_of_reader(file) {
                    cards.push(Card::adapter(
                        &format!("adapter-{}", &digest.hex()[..12]),
                        digest.as_str(),
                    ));
                }
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
    let mut last_event_id = None;
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
        if let Some(value) = lower.strip_prefix("last-event-id:") {
            last_event_id = value.trim().parse().ok();
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
    if crate::scripted_jobs::route(
        &mut stream,
        &shared.jobs,
        &request_line,
        last_event_id,
        &body,
    ) {
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
    } else if request_line.starts_with("GET /v1/capabilities") {
        let server = shared.server.lock().unwrap().clone();
        let actions = shared.capabilities.lock().unwrap().clone();
        let mut list = serde_json::json!({"object": "list", "data": actions});
        if !server.is_null() {
            list["server"] = server;
        }
        let _ = respond(&mut stream, "200 OK", "application/json", &list.to_string());
    } else if request_line.starts_with("GET /v1/devices") {
        // Two cards of 80 GiB, the first with 30 GiB held.
        let gib = 1u64 << 30;
        let card = |i: u64, used: u64| {
            serde_json::json!({"id": format!("gpu{i}"), "kind": "gpu", "memory": {
                "total_bytes": 80 * gib, "reserved_bytes": 0, "used_bytes": used * gib,
                "free_bytes": (80 - used) * gib}})
        };
        let list = serde_json::json!({"object": "list", "host": {"backend": "cuda"},
            "data": [{"id": "cpu", "kind": "cpu"}, card(1, 0), card(0, 30)]});
        let _ = respond(&mut stream, "200 OK", "application/json", &list.to_string());
    } else if request_line.starts_with("POST /v1/run ") {
        let call: serde_json::Value = serde_json::from_slice(&body).unwrap_or_default();
        shared.runs.lock().unwrap().push(call.clone());
        let records = call["blobs"]["records"]["data"]
            .as_str()
            .and_then(|b| {
                base64::Engine::decode(&base64::engine::general_purpose::STANDARD, b).ok()
            })
            .map(|b| String::from_utf8_lossy(&b).into_owned());
        let (Some("score"), Some(records)) = (call["action"].as_str(), records) else {
            let _ = respond(&mut stream, "400 Bad Request", "application/json", "{}");
            return;
        };
        let tokens: usize = records
            .lines()
            .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
            .flat_map(|r| r["messages"].as_array().cloned().unwrap_or_default())
            .filter(|m| m["train"] == true)
            .map(|m| m["content"].as_str().map_or(0, |c| c.chars().count()))
            .sum();
        let outputs = serde_json::json!({
            "tokens": tokens, "logprob": -0.5 * tokens as f64,
            "mean_logprob": if tokens == 0 { 0.0 } else { -0.5 }, "model": call["model"],
        });
        let _ = respond(
            &mut stream,
            "200 OK",
            "application/json",
            &serde_json::json!({"outputs": outputs, "blobs": {}}).to_string(),
        );
    } else if request_line.starts_with("POST /v1/embeddings") {
        let request: serde_json::Value = serde_json::from_slice(&body).unwrap_or_default();
        shared.embeddings.lock().unwrap().push(request.clone());
        let mut data: Vec<serde_json::Value> = request["input"]
            .as_array()
            .into_iter()
            .flatten()
            .enumerate()
            .map(|(index, text)| {
                let chars = text.as_str().map_or(0, |t| t.chars().count());
                serde_json::json!({"object": "embedding", "index": index, "embedding": [chars as f64, 1.0]})
            })
            .collect();
        data.reverse();
        let reply = serde_json::json!({"object": "list", "data": data}).to_string();
        let _ = respond(&mut stream, "200 OK", "application/json", &reply);
    } else if request_line.starts_with("POST /v1/audio/transcriptions ") {
        transcription(&mut stream, shared, &body);
    } else if request_line.starts_with("POST /v1/audio/speech ") {
        speech(&mut stream, shared, &body);
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
        policy_trace: request["policy_trace"] == serde_json::json!(true),
        sampler_logprobs: request["logprobs"] == serde_json::json!(true),
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

/// A refusal as brain words it: `{"error": {"message"}}` under `status`.
fn refuse(stream: &mut TcpStream, status: &str, message: &str) {
    let body = serde_json::json!({"error": {"message": message}}).to_string();
    let _ = respond(stream, status, "application/json", &body);
}

/// `POST /v1/audio/transcriptions` with OpenRouter's JSON body: the text a
/// scripted clip holds, for a model the endpoint lists.
fn transcription(stream: &mut TcpStream, shared: &Shared, body: &[u8]) {
    let request: serde_json::Value = serde_json::from_slice(body).unwrap_or_default();
    shared
        .speech
        .lock()
        .unwrap()
        .push(("transcriptions".into(), request.clone()));
    let model = request["model"].as_str().unwrap_or_default();
    if !listed(shared).iter().any(|card| card.id == model) {
        return refuse(stream, "404 Not Found", &format!("unknown model {model}"));
    }
    let clip = request["input_audio"]["data"]
        .as_str()
        .and_then(|b| base64::Engine::decode(&base64::engine::general_purpose::STANDARD, b).ok())
        .and_then(|bytes| crate::speech::Clip::from_wav(&bytes).ok());
    let Some(clip) = clip else {
        return refuse(
            stream,
            "400 Bad Request",
            "input_audio is not a WAV recording",
        );
    };
    let reply = serde_json::json!({
        "text": crate::speech::scripted::text_of(&clip),
        "usage": {"type": "duration", "seconds": clip.seconds()},
    });
    let _ = respond(stream, "200 OK", "application/json", &reply.to_string());
}

/// `POST /v1/audio/speech`: a scripted clip of `input` as a WAV, for a model
/// the endpoint lists. Like brain it takes `model`, `input`,
/// `response_format`, `seed` (echoed in `x-seed` when given) and
/// `input_references`, and refuses any other field by name.
fn speech(stream: &mut TcpStream, shared: &Shared, body: &[u8]) {
    let request: serde_json::Value = serde_json::from_slice(body).unwrap_or_default();
    shared
        .speech
        .lock()
        .unwrap()
        .push(("speech".into(), request.clone()));
    if let Some(field) = request.as_object().and_then(|o| {
        o.keys().find(|k| {
            !matches!(
                k.as_str(),
                "model" | "input" | "response_format" | "seed" | "input_references"
            )
        })
    }) {
        return refuse(
            stream,
            "400 Bad Request",
            &format!("'{field}' is not supported"),
        );
    }
    if !matches!(request["response_format"].as_str(), None | Some("wav")) {
        return refuse(stream, "400 Bad Request", "only wav is scripted");
    }
    let model = request["model"].as_str().unwrap_or_default();
    if !listed(shared).iter().any(|card| card.id == model) {
        return refuse(stream, "404 Not Found", &format!("unknown model {model}"));
    }
    let clip = crate::speech::scripted::clip_of(request["input"].as_str().unwrap_or_default());
    let wav = clip.to_wav();
    let _ = write!(
        stream,
        "HTTP/1.1 200 OK\r\ncontent-type: audio/wav\r\nx-sample-rate: {}\r\n{}content-length: {}\r\nconnection: close\r\n\r\n",
        clip.sample_rate(),
        request["seed"].as_u64().map(|seed| format!("x-seed: {seed}\r\n")).unwrap_or_default(),
        wav.len()
    )
    .and_then(|()| stream.write_all(&wav))
    .and_then(|()| stream.flush());
}

pub(crate) fn respond(
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
