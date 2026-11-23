// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training records taken from the exact
// requests an agent runtime sends, so a model is fine-tuned on what it will
// see when deployed, for its clients. If your team needs expertise in agent
// fine-tuning data, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Reading an agent's request log (`sven.request-log/1`): one JSON object
//! per line, `{"kind": "request", "seq", "model", "request": {"messages",
//! "tools", ...}}` when a request is sent (its whole conversation as the
//! model was given it) and `{"kind": "response", "seq", "response"}` with
//! what came back.

use std::collections::BTreeMap;

use serde_json::Value;
use splinter_core::harness::{is_media, message_parts, ModelCall, ModelResponse, ModelToolCall};

/// One request of the log.
#[derive(Clone, Debug, PartialEq)]
pub struct LoggedRequest {
    /// The model role it was sent to, when the log names one.
    pub role: Option<String>,
    /// Whether any message of the request carries media (an image).
    pub media: bool,
    /// The request and its response; `None` when the log holds no response
    /// to it, or one that cannot be read.
    pub call: Option<ModelCall>,
}

/// Every request of a log, in the order they were sent.
///
/// # Errors
/// A line that is not JSON (`line <n>: <error>`).
pub fn read_request_log(bytes: &[u8]) -> Result<Vec<LoggedRequest>, String> {
    let mut requests: Vec<(u64, LoggedRequest)> = Vec::new();
    let mut responses: BTreeMap<u64, Value> = BTreeMap::new();
    for (n, line) in bytes.split(|b| *b == b'\n').enumerate() {
        if line.is_empty() {
            continue;
        }
        let mut row: Value =
            serde_json::from_slice(line).map_err(|e| format!("line {}: {e}", n + 1))?;
        let seq = row.get("seq").and_then(Value::as_u64);
        match (row.get("kind").and_then(Value::as_str), seq) {
            (Some("request"), _) => {
                let role = row.get("model").and_then(Value::as_str).map(str::to_string);
                let mut request = row.get_mut("request").map(Value::take).unwrap_or_default();
                let messages = take_array(&mut request, "messages");
                let tools = take_array(&mut request, "tools");
                let media = messages.iter().any(|m| message_parts(m).any(is_media));
                let call = seq.map(|seq| ModelCall {
                    seq,
                    messages,
                    tools,
                    response: ModelResponse {
                        text: String::new(),
                        reasoning: None,
                        tool_calls: Vec::new(),
                    },
                });
                requests.push((seq.unwrap_or(0), LoggedRequest { role, media, call }));
            }
            (Some("response"), Some(seq)) => {
                if let Some(response) = row.get_mut("response") {
                    responses.insert(seq, response.take());
                }
            }
            _ => {}
        }
    }
    for (seq, logged) in &mut requests {
        let response = responses.get(seq).and_then(response_of);
        logged.call = match (logged.call.take(), response) {
            (Some(call), Some(response)) => Some(ModelCall { response, ..call }),
            _ => None,
        };
    }
    Ok(requests.into_iter().map(|(_, logged)| logged).collect())
}

/// Whether the answer to `call` is one an agent records a step for: it said
/// something or called a tool.
#[must_use]
pub fn made_a_step(call: &ModelCall) -> bool {
    !call.response.text.trim().is_empty() || !call.response.tool_calls.is_empty()
}

/// The array under `key` of `request`, moved out of it; empty when absent.
fn take_array(request: &mut Value, key: &str) -> Vec<Value> {
    match request.get_mut(key).map(Value::take) {
        Some(Value::Array(items)) => items,
        _ => Vec::new(),
    }
}

/// A logged response as the model answered it: its text, its reasoning and
/// its tool calls. `None` for a response that is not an object or whose calls
/// do not name an id, a tool and arguments.
fn response_of(response: &Value) -> Option<ModelResponse> {
    let object = response.as_object()?;
    let tool_calls = object
        .get("tool_calls")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|call| {
            let field = |key: &str| call.get(key).and_then(Value::as_str).map(str::to_string);
            Some(ModelToolCall {
                id: field("id")?,
                name: field("name")?,
                arguments: field("arguments")?,
            })
        })
        .collect::<Option<Vec<_>>>()?;
    Some(ModelResponse {
        text: object
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        reasoning: object
            .get("reasoning")
            .and_then(Value::as_str)
            .map(str::to_string),
        tool_calls,
    })
}
