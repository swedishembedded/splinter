// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that act through tools a caller
// defines, for its clients. If your team needs expertise in measuring and
// tracing what a model does with a tool it has never seen, you can procure
// our services by sending an email to info@swedishembedded.com.

//! One turn of a model that may call tools the caller defines.
//!
//! [`tool_turn`] puts a conversation and a set of tools to a model and returns
//! everything it did in that turn: the words it said, each tool call with its
//! arguments read as JSON, and what the turn cost. The tools are only offered;
//! nothing is executed here, because what a call does belongs to the caller
//! (a decision recorded, a lookup made, a refusal). A grader reads the calls;
//! a chat loop answers them and puts the next turn.

use std::collections::BTreeMap;
use std::sync::Arc;

use futures::StreamExt as _;
use serde_json::Value;
use sven_sdk::model::{CompletionRequest, Message, ModelProvider, ResponseEvent, ToolSchema};

/// One tool call a model made.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolCallMade {
    /// The call's id, to answer it by.
    pub id: String,
    /// The tool named.
    pub name: String,
    /// The arguments as JSON; `Null` when what the model wrote is not JSON.
    pub arguments: Value,
    /// The arguments exactly as the model wrote them.
    pub raw_arguments: String,
}

/// What a model did in one turn.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ToolTurn {
    /// The words it said outside any call.
    pub text: String,
    /// Its calls, in the order of the slots it used.
    pub calls: Vec<ToolCallMade>,
    /// Tokens of the request, as the provider reports them; 0 if it does not.
    pub input_tokens: u32,
    /// Tokens of the reply, as the provider reports them; 0 if it does not.
    pub output_tokens: u32,
    /// Whether the model stopped on the token limit, so that a call may be cut
    /// short.
    pub truncated: bool,
}

/// Puts `messages` and `tools` to `provider` and reads the turn it makes, at
/// most `max_tokens` long.
///
/// # Errors
/// The provider failed to start the reply or failed in the middle of it.
pub async fn tool_turn(
    provider: &Arc<dyn ModelProvider>,
    messages: Vec<Message>,
    tools: Vec<ToolSchema>,
    max_tokens: u32,
) -> anyhow::Result<ToolTurn> {
    let request = CompletionRequest {
        messages,
        tools,
        stream: true,
        max_output_tokens_override: Some(max_tokens),
        ..Default::default()
    };
    let mut stream = provider.complete(request).await?;
    let mut turn = ToolTurn::default();
    // Chunks of one call share a slot; the id and name arrive in the first.
    let mut slots: BTreeMap<u32, (String, String, String)> = BTreeMap::new();
    while let Some(event) = stream.next().await {
        match event? {
            ResponseEvent::TextDelta(text) => turn.text.push_str(&text),
            ResponseEvent::ToolCall {
                index,
                id,
                name,
                arguments,
            } => {
                let slot = slots.entry(index).or_default();
                if !id.is_empty() {
                    slot.0 = id;
                }
                if !name.is_empty() {
                    slot.1 = name;
                }
                slot.2.push_str(&arguments);
            }
            ResponseEvent::Usage {
                input_tokens,
                output_tokens,
                ..
            } => {
                turn.input_tokens = input_tokens;
                turn.output_tokens = output_tokens;
            }
            ResponseEvent::MaxTokens => turn.truncated = true,
            ResponseEvent::Error(e) => anyhow::bail!("the model failed: {e}"),
            ResponseEvent::Done => break,
            _ => {}
        }
    }
    turn.calls = slots
        .into_values()
        .map(|(id, name, raw_arguments)| ToolCallMade {
            id,
            name,
            arguments: serde_json::from_str(&raw_arguments).unwrap_or(Value::Null),
            raw_arguments,
        })
        .collect();
    Ok(turn)
}
