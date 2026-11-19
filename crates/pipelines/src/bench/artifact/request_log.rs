// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements tamper-evident hand-over of agent benchmark
// results into training pipelines, for its clients. If your team needs
// expertise in turning benchmark runs into verified training data, you can
// procure our services by sending an email to info@swedishembedded.com.

//! A harness trial's request log (`model-requests.jsonl`), divided among its
//! conversations: which model role each request was sent to, whether it
//! carried an image, and each request with the response it got.

use std::collections::BTreeMap;

use serde_json::Value;
use splinter_core::harness::{
    is_media, message_parts, ModelCall, ModelResponse, ModelToolCall, RoleSource,
};

use super::REQUESTS;
/// The requests of one conversation as the request log attributes them.
#[derive(Clone, Debug)]
pub(super) struct Attribution {
    /// The one model role all its requests name, if they do.
    pub(super) role: Option<String>,
    pub(super) images: bool,
    /// Which requests of the log they are, in order.
    requests: Vec<usize>,
}

impl Attribution {
    pub(super) fn source(&self) -> RoleSource {
        if self.role.is_some() {
            RoleSource::RequestLog
        } else {
            RoleSource::Unknown
        }
    }
}

/// One request of the log.
struct Logged {
    role: Option<String>,
    /// Whether any message of the request carries media (an image).
    media: bool,
    /// The request and its response; `None` when the log holds no response
    /// to it, or one that cannot be read.
    call: Option<ModelCall>,
}

/// What `model-requests.jsonl` says about each request: the model role it
/// was sent to, whether it carries an image, and the request itself with the
/// response it got.
pub(super) struct RequestLog {
    requests: Vec<Logged>,
    pub(super) any_image: bool,
}

impl RequestLog {
    pub(super) fn parse(bytes: &[u8]) -> Result<Self, String> {
        let mut requests: Vec<(u64, Logged)> = Vec::new();
        let mut responses: BTreeMap<u64, Value> = BTreeMap::new();
        for (n, line) in bytes.split(|b| *b == b'\n').enumerate() {
            if line.is_empty() {
                continue;
            }
            let mut row: Value = serde_json::from_slice(line)
                .map_err(|e| format!("{REQUESTS} line {}: {e}", n + 1))?;
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
                    let logged = Logged { role, media, call };
                    requests.push((seq.unwrap_or(0), logged));
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
        let requests: Vec<Logged> = requests.into_iter().map(|(_, logged)| logged).collect();
        let any_image = requests.iter().any(|r| r.media);
        Ok(Self {
            requests,
            any_image,
        })
    }

    /// Divides the requests among `conversations`, in order, one request per
    /// agent step. Only an exact division counts: the request count must
    /// equal the agent steps of all the conversations, or else the count of
    /// the requests that made a step - those answered with text or a call; an
    /// answer that said nothing, which the agent records no step for and asks
    /// again after, and a request the trial ended before answering, made
    /// none - or the requests cannot be told apart and nothing is attributed.
    pub(super) fn attribute(
        &self,
        conversations: &[&atif::Trajectory],
    ) -> Option<Vec<Attribution>> {
        let steps: Vec<usize> = conversations
            .iter()
            .map(|t| {
                t.steps
                    .iter()
                    .filter(|s| s.source == atif::StepOrigin::Agent)
                    .count()
            })
            .collect();
        let total = steps.iter().sum::<usize>();
        let stepping: Vec<usize> = if self.requests.len() == total {
            (0..total).collect()
        } else {
            (0..self.requests.len())
                .filter(|&n| self.requests[n].call.as_ref().is_some_and(made_a_step))
                .collect()
        };
        if stepping.len() != total {
            return None;
        }
        let mut start = 0;
        Some(
            steps
                .into_iter()
                .map(|n| {
                    let own = &stepping[start..start + n];
                    let first = own.first().and_then(|&r| self.requests[r].role.clone());
                    let one_role = own.iter().all(|&r| self.requests[r].role == first);
                    start += n;
                    Attribution {
                        role: first.filter(|_| one_role),
                        images: own.iter().any(|&r| self.requests[r].media),
                        requests: own.to_vec(),
                    }
                })
                .collect(),
        )
    }

    /// The calls of the requests attributed to a conversation; `None` unless
    /// the log holds a readable response to every one of them.
    pub(super) fn calls_of(&self, attribution: &Attribution) -> Option<Vec<ModelCall>> {
        attribution
            .requests
            .iter()
            .map(|&r| self.requests[r].call.clone())
            .collect()
    }
}

/// Whether the answer to `call` is one an agent records a step for: it said
/// something or called a tool.
fn made_a_step(call: &ModelCall) -> bool {
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
