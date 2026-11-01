// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training data for multi-role agent systems,
// where each model role learns only from the conversations it answered, for
// its clients. If your team needs expertise in fine-tuning the roles of an
// agent harness, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Which harness conversation an experience is, and which model role
//! answered it.
//!
//! A trial that ran a sven harness is several conversations, each answered by
//! a model role. Ingestion admits one experience per conversation and files
//! this record under `extra.splinter_harness` of the experience's
//! trajectory; the dataset views read it back to train a role on its own
//! conversations and to leave out those whose model was shown images.

use atif::Trajectory;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The key of `trajectory.extra` the record is filed under.
pub const KEY: &str = "splinter_harness";

/// The key of `trajectory.extra` the conversation's recorded model calls are
/// filed under ([`RecordedCalls`]).
pub const CALLS_KEY: &str = "splinter_harness_calls";

/// Where a conversation's role came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoleSource {
    /// The export names it.
    Export,
    /// The export does not; the request log attributes the conversation's
    /// requests to one model role, unambiguously.
    RequestLog,
    /// Neither says: the role is unknown, never guessed.
    Unknown,
}

/// What an experience records about the harness conversation it is.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HarnessConversation {
    /// The conversation's name.
    pub conversation: String,
    /// The model role that answered it; `None` when unknown.
    pub role: Option<String>,
    /// Where `role` came from.
    pub role_source: RoleSource,
    /// Whether the model was shown images in this conversation. The
    /// trajectory need not carry them, so a record built from it would teach
    /// answers about pictures the student never saw.
    pub shown_images: bool,
}

impl HarnessConversation {
    /// The record filed in `trajectory`, `None` for an experience that is not
    /// a harness conversation.
    #[must_use]
    pub fn of(trajectory: &Trajectory) -> Option<Self> {
        let record = trajectory.extra.as_ref()?.get(KEY)?;
        serde_json::from_value(record.clone()).ok()
    }

    /// The record as the JSON filed under [`KEY`].
    #[must_use]
    pub fn to_value(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
}

/// One tool call of a model's response, as the request log recorded it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelToolCall {
    /// The call's id.
    pub id: String,
    /// The tool called.
    pub name: String,
    /// The arguments, as the model emitted them.
    pub arguments: String,
}

/// What a model answered to one request.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelResponse {
    /// The answer's text.
    pub text: String,
    /// The reasoning before it, when the model produced any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    /// The tool calls it made.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ModelToolCall>,
}

/// One request a harness conversation sent its model, exactly as sent, and
/// the response it got. Messages and tools are sven's own JSON (the request
/// log's), whose reading belongs to the dataset view.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelCall {
    /// The request's number in the trial's request log.
    pub seq: u64,
    /// The messages of the request.
    pub messages: Vec<Value>,
    /// The tool schemas the request offered.
    pub tools: Vec<Value>,
    /// What the model answered.
    pub response: ModelResponse,
}

/// A request as stored: only what differs from the request before it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct StoredCall {
    seq: u64,
    /// How many leading messages are those of the request before.
    shared: usize,
    /// The messages after them.
    messages: Vec<Value>,
    /// The tools, when they differ from the request before's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Value>>,
    response: ModelResponse,
}

/// The model calls of one harness conversation, filed under [`CALLS_KEY`].
///
/// A conversation's requests grow: each repeats the one before and adds the
/// turns since, and a picture shown once is in every later request. They are
/// stored as the difference to the request before, so a conversation costs
/// what its longest request costs and not the sum of all of them, and read
/// back whole.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RecordedCalls {
    /// The sha256 of the `model-requests.jsonl` the calls were read from.
    pub source_sha256: String,
    calls: Vec<StoredCall>,
}

impl RecordedCalls {
    /// Stores `calls`, which were read from the request log with
    /// `source_sha256`.
    #[must_use]
    pub fn record(source_sha256: &str, calls: &[ModelCall]) -> Self {
        let mut stored = Vec::with_capacity(calls.len());
        let mut before: Option<&ModelCall> = None;
        for call in calls {
            let shared = before.map_or(0, |b| {
                b.messages
                    .iter()
                    .zip(&call.messages)
                    .take_while(|(x, y)| x == y)
                    .count()
            });
            stored.push(StoredCall {
                seq: call.seq,
                shared,
                messages: call.messages[shared..].to_vec(),
                tools: before
                    .is_none_or(|b| b.tools != call.tools)
                    .then(|| call.tools.clone()),
                response: call.response.clone(),
            });
            before = Some(call);
        }
        Self {
            source_sha256: source_sha256.to_ascii_lowercase(),
            calls: stored,
        }
    }

    /// The calls, each with its whole request. `None` when the record is not
    /// consistent (a request shares more messages than the one before held).
    #[must_use]
    pub fn calls(&self) -> Option<Vec<ModelCall>> {
        let mut out: Vec<ModelCall> = Vec::with_capacity(self.calls.len());
        for stored in &self.calls {
            let before = out.last();
            let prefix = before.map_or(Some(&[][..]), |b| b.messages.get(..stored.shared))?;
            let mut messages = prefix.to_vec();
            messages.extend(stored.messages.iter().cloned());
            let tools = match (&stored.tools, before) {
                (Some(tools), _) => tools.clone(),
                (None, Some(b)) => b.tools.clone(),
                (None, None) => Vec::new(),
            };
            out.push(ModelCall {
                seq: stored.seq,
                messages,
                tools,
                response: stored.response.clone(),
            });
        }
        Some(out)
    }

    /// The record filed in `trajectory`, `None` for a conversation that
    /// recorded none (ingested before they were kept, or whose requests could
    /// not be told apart).
    #[must_use]
    pub fn of(trajectory: &Trajectory) -> Option<Self> {
        let record = trajectory.extra.as_ref()?.get(CALLS_KEY)?;
        serde_json::from_value(record.clone()).ok()
    }

    /// The record as the JSON filed under [`CALLS_KEY`].
    #[must_use]
    pub fn to_value(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
}

/// The content parts of a message of sven's request format, in order: the
/// parts of its content, or of the content of the tool result it carries (a
/// tool result is `{"tool_call_id", "content"}` inside the content). A message
/// whose content is a string has none.
pub fn message_parts(message: &Value) -> impl Iterator<Item = &Value> {
    let content = message.get("content");
    let content = content
        .filter(|c| c.is_object())
        .and_then(|c| c.get("content"))
        .or(content);
    content.and_then(Value::as_array).into_iter().flatten()
}

/// Whether `part` is a part that is not text: media a model was shown.
#[must_use]
pub fn is_media(part: &Value) -> bool {
    part.get("type").and_then(Value::as_str) != Some("text")
}

/// The URL of `part` when it is an image: sven writes one as
/// `{"type":"image","image_url":URL}`.
#[must_use]
pub fn image_url(part: &Value) -> Option<&str> {
    (part.get("type").and_then(Value::as_str) == Some("image"))
        .then(|| part.get("image_url").and_then(Value::as_str))
        .flatten()
}
