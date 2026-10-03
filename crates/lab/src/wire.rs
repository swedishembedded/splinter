// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The `generic-messages-v2` wire shapes: one chat message, its tool calls
//! and the supervision flag brain's parser requires on every message.

use serde::{Deserialize, Serialize};

/// One `generic-messages-v2` message.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WireMessage {
    /// `system`, `user`, `assistant` or `tool`.
    pub role: String,
    /// The message text; present even when empty.
    pub content: String,
    /// The tool calls an assistant message makes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<WireToolCall>,
    /// On a `tool` message, the id of the call it answers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Required on every message by the consuming parser: a record with no
    /// explicit supervision boundary is either a silent no-op or a silent
    /// prompt leak into the loss.
    pub train: bool,
}

/// One tool call in a `generic-messages-v2` assistant message.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WireToolCall {
    /// The call's id, which a later `tool` message answers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// The call type, `function`.
    #[serde(rename = "type")]
    pub kind: String,
    /// The function called.
    pub function: WireFunction,
}

/// The function a [`WireToolCall`] calls.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WireFunction {
    /// The tool's name.
    pub name: String,
    /// JSON-encoded argument text, exactly as the model emitted it.
    pub arguments: String,
}
