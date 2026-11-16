// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements local model endpoints that let a real agent
// runtime be driven by a programmatic policy, for its clients. If your team
// needs expertise in capturing exact agent requests as training data, you
// can procure our services by sending an email to info@swedishembedded.com.

//! A small OpenAI-compatible endpoint that answers with a policy, and keeps
//! every request it was sent.
//!
//! The agent runtime talks to it exactly as it talks to a served model, so
//! the requests kept here - the system prompt, the tool definitions and the
//! whole conversation, as the runtime wrote them - are exactly what the
//! model will be sent at inference. A demonstration is therefore the last
//! request of an episode with the policy's last reply appended. The endpoint
//! binds a free port on the loopback interface and stops when dropped.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use serde_json::{json, Value};

/// The policy behind the endpoint: the assistant message (OpenAI shape) that
/// answers one request.
pub type Policy = dyn Fn(&Value) -> anyhow::Result<Value> + Send + Sync;

/// One request and what the policy answered.
#[derive(Clone, Debug)]
pub struct Exchange {
    /// The request body as sent.
    pub request: Value,
    /// The assistant message the policy answered with, or its error.
    pub reply: Result<Value, String>,
}

struct Shared {
    model: String,
    policy: Box<Policy>,
    exchanges: Mutex<Vec<Exchange>>,
}

/// A running endpoint.
pub struct Endpoint {
    url: String,
    shared: Arc<Shared>,
    stop: Arc<AtomicBool>,
    port: u16,
    thread: Option<JoinHandle<()>>,
}

impl Endpoint {
    /// Starts an endpoint serving model `model` with `policy`.
    ///
    /// # Errors
    /// No loopback port can be bound.
    pub fn start(model: &str, policy: Box<Policy>) -> anyhow::Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        let shared = Arc::new(Shared {
            model: model.to_string(),
            policy,
            exchanges: Mutex::new(Vec::new()),
        });
        let stop = Arc::new(AtomicBool::new(false));
        let thread = {
            let (shared, stop) = (Arc::clone(&shared), Arc::clone(&stop));
            std::thread::spawn(move || {
                for stream in listener.incoming() {
                    if stop.load(Ordering::SeqCst) {
                        break;
                    }
                    if let Ok(stream) = stream {
                        let shared = Arc::clone(&shared);
                        std::thread::spawn(move || handle(stream, &shared));
                    }
                }
            })
        };
        Ok(Self {
            url: format!("http://127.0.0.1:{port}/v1"),
            shared,
            stop,
            port,
            thread: Some(thread),
        })
    }

    /// The base URL, `http://127.0.0.1:<port>/v1`.
    #[must_use]
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Every exchange so far, in arrival order.
    #[must_use]
    pub fn exchanges(&self) -> Vec<Exchange> {
        self.shared
            .exchanges
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

impl Drop for Endpoint {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        // Wake the accept loop so it sees the flag.
        let _ = TcpStream::connect(("127.0.0.1", self.port));
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn handle(mut stream: TcpStream, shared: &Shared) {
    let Ok(clone) = stream.try_clone() else {
        return;
    };
    let mut reader = BufReader::new(clone);
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).unwrap_or(0) == 0 {
        return;
    }
    let mut length = 0usize;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
            break;
        }
        if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
            length = value.trim().parse().unwrap_or(0);
        }
    }
    let mut body = vec![0u8; length];
    if reader.read_exact(&mut body).is_err() {
        return;
    }
    if request_line.starts_with("GET /v1/models") {
        let list = json!({ "object": "list", "data": [{ "id": shared.model, "object": "model", "owned_by": "minimind" }] });
        let _ = respond(&mut stream, "200 OK", "application/json", &list.to_string());
    } else if request_line.starts_with("POST /v1/chat/completions") {
        chat(&mut stream, shared, &body);
    } else {
        let _ = respond(&mut stream, "404 Not Found", "application/json", "{}");
    }
}

fn chat(stream: &mut TcpStream, shared: &Shared, body: &[u8]) {
    let request: Value = serde_json::from_slice(body).unwrap_or_default();
    let reply = (shared.policy)(&request).map_err(|e| format!("{e:#}"));
    shared
        .exchanges
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push(Exchange {
            request: request.clone(),
            reply: reply.clone(),
        });
    let message = match reply {
        Ok(message) => message,
        Err(reason) => {
            let error = json!({ "error": { "message": format!("policy: {reason}"), "type": "server_error" } });
            let _ = respond(
                stream,
                "500 Internal Server Error",
                "application/json",
                &error.to_string(),
            );
            return;
        }
    };
    let model = request["model"]
        .as_str()
        .unwrap_or(&shared.model)
        .to_string();
    let finish = if message
        .get("tool_calls")
        .is_some_and(|c| c.as_array().is_some_and(|a| !a.is_empty()))
    {
        "tool_calls"
    } else {
        "stop"
    };
    if request["stream"].as_bool().unwrap_or(false) {
        let mut delta = message.clone();
        if let Some(calls) = delta.get_mut("tool_calls").and_then(Value::as_array_mut) {
            for (i, call) in calls.iter_mut().enumerate() {
                call["index"] = json!(i);
            }
        }
        let chunk = |choice: Value| json!({ "id": "minimind", "object": "chat.completion.chunk", "created": 0, "model": model, "choices": [choice] });
        let mut sse = String::new();
        for c in [
            chunk(json!({ "index": 0, "delta": delta, "finish_reason": null })),
            chunk(json!({ "index": 0, "delta": {}, "finish_reason": finish })),
        ] {
            sse.push_str(&format!("data: {c}\n\n"));
        }
        sse.push_str("data: [DONE]\n\n");
        let _ = respond(stream, "200 OK", "text/event-stream", &sse);
    } else {
        let whole = json!({ "id": "minimind", "object": "chat.completion", "created": 0, "model": model,
            "choices": [{ "index": 0, "message": message, "finish_reason": finish }] });
        let _ = respond(stream, "200 OK", "application/json", &whole.to_string());
    }
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

/// The assistant message `base_url` answers `request` with, asked with
/// `key`: the request is sent unchanged but for streaming, which is turned
/// off.
///
/// # Errors
/// The server cannot be reached, refuses, or answers with no message.
pub fn forward(base_url: &str, key: &str, request: &Value) -> anyhow::Result<Value> {
    use anyhow::Context as _;
    let rest = base_url
        .strip_prefix("http://")
        .context("only a plain http:// model server can be recorded")?;
    let (host, prefix) = rest
        .split_once('/')
        .map_or((rest, String::new()), |(h, p)| (h, format!("/{p}")));
    let mut body = request.clone();
    body["stream"] = json!(false);
    body.as_object_mut().map(|o| o.remove("stream_options"));
    let body = body.to_string();
    let mut stream = TcpStream::connect(host).with_context(|| format!("connecting to {host}"))?;
    write!(
        stream,
        "POST {prefix}/chat/completions HTTP/1.1\r\nhost: {host}\r\ncontent-type: application/json\r\n\
         authorization: Bearer {key}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    )?;
    let mut raw = Vec::new();
    stream
        .read_to_end(&mut raw)
        .with_context(|| format!("reading the answer of {host}"))?;
    let split = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .context("the answer has no header end")?;
    let head = String::from_utf8_lossy(&raw[..split]).to_ascii_lowercase();
    let mut payload = raw[split + 4..].to_vec();
    if head.contains("transfer-encoding: chunked") {
        payload = dechunk(&payload).context("the answer's chunked body is malformed")?;
    }
    let status = head.split_whitespace().nth(1).unwrap_or("");
    let answer: Value = serde_json::from_slice(&payload).with_context(|| {
        format!(
            "{host} answered {status} with no JSON: {}",
            String::from_utf8_lossy(&payload)
        )
    })?;
    if status != "200" {
        anyhow::bail!("{host} answered {status}: {answer}");
    }
    answer["choices"][0]["message"]
        .as_object()
        .map(|m| Value::Object(m.clone()))
        .context("the answer holds no message")
}

fn dechunk(mut body: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    loop {
        let line_end = body.windows(2).position(|w| w == b"\r\n")?;
        let size =
            usize::from_str_radix(String::from_utf8_lossy(&body[..line_end]).trim(), 16).ok()?;
        body = &body[line_end + 2..];
        if size == 0 {
            return Some(out);
        }
        out.extend_from_slice(body.get(..size)?);
        body = body.get(size + 2..)?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_policy_answers_through_the_wire_and_every_exchange_is_kept() {
        let policy = |request: &Value| -> anyhow::Result<Value> {
            let n = request["messages"].as_array().map_or(0, Vec::len);
            Ok(json!({ "role": "assistant", "content": format!("saw {n}") }))
        };
        let endpoint = Endpoint::start("m", Box::new(policy)).unwrap();
        let request = json!({ "model": "m", "stream": true, "messages": [{ "role": "user", "content": "hi" }] });
        let reply = forward(endpoint.url(), "k", &request).unwrap();
        assert_eq!(reply["content"], "saw 1");
        let kept = endpoint.exchanges();
        assert_eq!(kept.len(), 1);
        assert_eq!(
            kept[0].request["stream"],
            json!(false),
            "a forwarded request is not streamed"
        );
        assert_eq!(kept[0].reply.as_ref().unwrap()["content"], "saw 1");
    }

    #[test]
    fn a_failing_policy_is_a_server_error_not_an_answer() {
        let endpoint =
            Endpoint::start("m", Box::new(|_: &Value| anyhow::bail!("no idea"))).unwrap();
        let err = forward(
            endpoint.url(),
            "k",
            &json!({ "model": "m", "messages": [] }),
        )
        .unwrap_err();
        assert!(format!("{err:#}").contains("no idea"), "{err:#}");
        assert!(endpoint.exchanges()[0].reply.is_err());
    }
}
