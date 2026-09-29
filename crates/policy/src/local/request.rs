// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements coding agents that run their model
// in-process - no separately versioned serving process to drift from the
// build. If your team needs expertise in local inference integration or
// provider abstractions, you can procure our services by sending an email
// to info@swedishembedded.com.

//! sven's completion request as the invocation brain's chat parser reads:
//! OpenAI-shaped messages and tools, the output budget and sampling.

use anyhow::Context;
use capability::Invocation;
use sven_sdk::model::{CompletionRequest, ContentPart, Message, MessageContent, Role, ToolSchema};

/// Top-k for agent work: see the sampling defaults in the parent module.
const DEFAULT_TOP_K: i64 = 20;

/// Maps sven's request onto the invocation brain's chat parser reads. The
/// message/tool shapes are the OpenAI wire shapes brain's
/// `parse_chat_messages`/`parse_tools` accept - sven's types are serialized
/// into those shapes rather than re-invented here.
pub(super) fn invocation_from(
    req: &CompletionRequest,
    max_new: usize,
    temperature: f64,
) -> anyhow::Result<Invocation> {
    // The request's output budget is a contract: a caller asking for more
    // than the default (explore wants one WHOLE JSON object per section)
    // must get it. complete() clamps the ask to the KV cache first; what
    // survives to here is what the engine will be told.
    let max_new = req
        .max_output_tokens_override
        .map_or(max_new, |n| n as usize);
    let messages: Vec<serde_json::Value> = req.messages.iter().map(message_json).collect();
    let tools: Vec<serde_json::Value> = req.tools.iter().map(tool_json).collect();
    let mut inv = Invocation::new();
    if !messages.is_empty() {
        inv = inv.set(
            "messages",
            serde_json::Value::String(
                serde_json::to_string(&messages).context("serializing messages")?,
            ),
        );
    }
    if !tools.is_empty() {
        inv = inv.set(
            "tools",
            serde_json::Value::String(serde_json::to_string(&tools).context("serializing tools")?),
        );
    }
    inv = inv
        .set("max_new", serde_json::json!(max_new))
        .set("temp", serde_json::json!(temperature))
        .set("top_k", serde_json::json!(DEFAULT_TOP_K))
        // Agent work wants the answer, not a reasoning preamble it cannot
        // use as tool input.
        .set("enable_thinking", serde_json::json!(false));
    Ok(inv)
}

/// One sven message as brain's chat parser reads it. A tool result rides in
/// as `role: "tool"` with its call id; an assistant tool request rides out
/// as `tool_calls`, so the template renders the exchange the model itself
/// produced.
#[must_use]
fn message_json(message: &Message) -> serde_json::Value {
    let role = match message.role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    };
    let mut value = serde_json::json!({
        "role": role,
        "content": content_text(message),
    });
    match &message.content {
        MessageContent::ToolCall {
            tool_call_id,
            function,
        } => {
            value["tool_calls"] = serde_json::json!([{
                "id": tool_call_id,
                "function": {"name": function.name, "arguments": function.arguments},
            }]);
            value["content"] = serde_json::Value::String(String::new());
        }
        MessageContent::ToolResult {
            tool_call_id,
            content,
        } => {
            value["tool_call_id"] = serde_json::json!(tool_call_id);
            if let Some(text) = content.as_text() {
                value["content"] = serde_json::json!(text);
            }
        }
        _ => {}
    }
    value
}

/// The message's text content: plain text verbatim, mixed parts joined. A
/// tool-result part array keeps its text rather than disappearing.
#[must_use]
fn content_text(message: &Message) -> String {
    match &message.content {
        MessageContent::Text(text) => text.clone(),
        MessageContent::ContentParts(parts) => parts
            .iter()
            .filter_map(|p| match p {
                ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

/// One sven tool schema as an OpenAI-shaped function object, the shape
/// brain's tool parser accepts.
#[must_use]
fn tool_json(tool: &ToolSchema) -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use qwen3::chat;
    use sven_sdk::model::{FunctionCall, ToolResultContent};

    fn message(role: Role, content: MessageContent) -> Message {
        Message { role, content }
    }

    /// The provider's default generation cap, as the parent module sets it.
    const DEFAULT_MAX_NEW_TOKENS: usize = super::super::DEFAULT_MAX_NEW_TOKENS;

    /// A request's output budget is a contract with the caller: `explore`
    /// asks for 32k because a section's facts reply must arrive as ONE
    /// complete JSON object - a reply truncated at the provider's own
    /// default mid-string is a parse failure and the section's facts are
    /// lost (observed: "EOF while parsing a string"). The override must
    /// reach the engine, clamped to what the KV cache can hold.
    #[test]
    fn a_requests_output_budget_reaches_the_invocation() {
        let req = CompletionRequest {
            messages: vec![message(Role::User, MessageContent::Text("q".into()))],
            max_output_tokens_override: Some(32_768),
            ..CompletionRequest::default()
        };
        let inv = invocation_from(&req, DEFAULT_MAX_NEW_TOKENS, 0.7).unwrap();
        assert_eq!(
            inv.params.get("max_new"),
            Some(&serde_json::json!(32_768)),
            "the request's output budget must override the provider default"
        );
        // No override: the provider default stands.
        let plain = CompletionRequest {
            messages: vec![message(Role::User, MessageContent::Text("q".into()))],
            ..CompletionRequest::default()
        };
        let inv = invocation_from(&plain, DEFAULT_MAX_NEW_TOKENS, 0.7).unwrap();
        assert_eq!(
            inv.params.get("max_new"),
            Some(&serde_json::json!(DEFAULT_MAX_NEW_TOKENS))
        );
    }

    #[test]
    fn sven_messages_map_onto_the_openai_shapes_brains_parser_reads() {
        let user = message(Role::User, MessageContent::Text("do the thing".into()));
        let assistant = message(
            Role::Assistant,
            MessageContent::ToolCall {
                tool_call_id: "call_1".into(),
                function: FunctionCall {
                    name: "write".into(),
                    arguments: r#"{"path":"a.txt"}"#.into(),
                },
            },
        );
        let tool = message(
            Role::Tool,
            MessageContent::ToolResult {
                tool_call_id: "call_1".into(),
                content: ToolResultContent::Text("wrote 5 bytes".into()),
            },
        );
        let mapped: Vec<serde_json::Value> = [&user, &assistant, &tool]
            .iter()
            .map(|m| message_json(m))
            .collect();
        assert_eq!(mapped[0]["role"], "user");
        assert_eq!(mapped[1]["tool_calls"][0]["function"]["name"], "write");
        assert_eq!(mapped[2]["role"], "tool");
        assert_eq!(mapped[2]["tool_call_id"], "call_1");
        assert_eq!(mapped[2]["content"], "wrote 5 bytes");
        // Round-trip through brain's own parser - the consumer this feeds.
        let raw = serde_json::to_string(&mapped).unwrap();
        let parsed = chat::parse_chat_messages(&raw, None).unwrap();
        assert_eq!(parsed.len(), 3);
        assert_eq!(parsed[1].tool_calls[0].name, "write");
        assert_eq!(parsed[1].tool_calls[0].arguments, r#"{"path":"a.txt"}"#);
        assert_eq!(parsed[2].role, data::qwen_chat::Role::Tool);
        assert_eq!(parsed[2].tool_call_id.as_deref(), Some("call_1"));
    }

    #[test]
    fn sven_tool_schemas_map_onto_openai_function_objects() {
        let tool = ToolSchema {
            name: "read".into(),
            description: "read a file".into(),
            parameters: serde_json::json!({"type":"object","properties":{"path":{"type":"string"}}}),
            is_mcp: false,
        };
        let mapped = tool_json(&tool);
        // The tools param is the whole array, as the invocation builds it.
        let raw = serde_json::to_string(&vec![mapped]).unwrap();
        let parsed = chat::parse_tools(Some(&raw)).unwrap();
        assert_eq!(parsed.len(), 1);
        assert!(parsed[0].contains("read"));
    }

    #[test]
    fn mixed_content_parts_keep_their_text() {
        let message = message(
            Role::User,
            MessageContent::ContentParts(vec![ContentPart::Text {
                text: "see ".into(),
            }]),
        );
        assert_eq!(content_text(&message), "see ");
    }
}
