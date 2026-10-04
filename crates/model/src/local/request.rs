// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements coding agents that run their model
// in-process - no separately versioned serving process to drift from the
// build. If your team needs expertise in local inference integration or
// provider abstractions, you can procure our services by sending an email
// to info@swedishembedded.com.

//! sven's completion request as a [`brain::ChatRequest`]: typed messages
//! and tools, the output budget and sampling.
//!
//! A request's `response_format` has no counterpart: brain's chat pipeline
//! has no constrained decoding, and sven's contract is that a driver which
//! cannot constrain the reply ignores the field while the caller post-parses.

use brain::chat::{ToolCall, ToolSchema};
use brain::{ChatMessage, ChatRequest};
use sven_sdk::model::{CompletionRequest, ContentPart, Message, MessageContent, Role};

/// How a local model samples each reply: the generation cap for a request
/// that names none, the temperature and the top-k cut.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Sampling {
    /// Generation cap when the request names none.
    pub max_new_tokens: u32,
    /// The softmax temperature.
    pub temperature: f32,
    /// Only the `top_k` most likely tokens are sampled from.
    pub top_k: u32,
    /// Whether the model may reason before it answers. Off, a model whose
    /// template opens a reasoning block is asked with the block closed, so
    /// the reply is the answer and comes at once; on, it reasons first and
    /// the reply cap leaves room for that.
    #[serde(default)]
    pub thinking: bool,
}

/// Maps sven's request onto brain's. The output budget is the request's own
/// when it names one - a caller asking for more than the default (explore
/// wants one WHOLE JSON object per section) must get it - and brain bounds
/// it by what the prompt leaves of the context, so a big ask degrades to
/// "every token the engine can still hold" rather than failing.
pub(super) fn chat_request(req: &CompletionRequest, sampling: &Sampling) -> ChatRequest {
    ChatRequest::new(chat_messages(&req.messages))
        .tools(req.tools.iter().map(tool_schema).collect())
        .max_tokens(output_budget(req, sampling))
        .temperature(sampling.temperature)
        .top_k(sampling.top_k)
        .thinking(sampling.thinking)
}

/// The most tokens the reply may take: the request's override, else the
/// provider default.
fn output_budget(req: &CompletionRequest, sampling: &Sampling) -> u32 {
    req.max_output_tokens_override
        .unwrap_or(sampling.max_new_tokens)
}

/// sven's messages as brain's conversation. sven records one assistant turn
/// as its text followed by one message per tool call; those fold back into
/// the single assistant turn the model produced, so the template renders the
/// exchange as the model wrote it - and as training renders it.
fn chat_messages(messages: &[Message]) -> Vec<ChatMessage> {
    let mut out: Vec<ChatMessage> = Vec::with_capacity(messages.len());
    // The assistant turn being assembled: its text and the calls so far.
    let mut assistant: Option<(String, Vec<ToolCall>)> = None;
    let flush = |out: &mut Vec<ChatMessage>, turn: Option<(String, Vec<ToolCall>)>| {
        if let Some((text, calls)) = turn {
            out.push(ChatMessage::assistant(text).with_tool_calls(calls));
        }
    };
    for message in messages {
        match (&message.role, &message.content) {
            (
                Role::Assistant,
                MessageContent::ToolCall {
                    tool_call_id,
                    function,
                },
            ) => {
                let call = ToolCall {
                    id: tool_call_id.clone(),
                    name: function.name.clone(),
                    arguments: function.arguments.clone(),
                };
                assistant
                    .get_or_insert_with(|| (String::new(), Vec::new()))
                    .1
                    .push(call);
            }
            (Role::Assistant, _) => {
                flush(&mut out, assistant.take());
                assistant = Some((content_text(&message.content), Vec::new()));
            }
            (role, content) => {
                flush(&mut out, assistant.take());
                out.push(match (role, content) {
                    (
                        _,
                        MessageContent::ToolResult {
                            tool_call_id,
                            content,
                        },
                    ) => ChatMessage::tool(
                        tool_call_id.clone(),
                        // sven's SDK names no type for a result's parts, so
                        // only a plain-text result carries text here.
                        content.as_text().unwrap_or_default(),
                    ),
                    (Role::System, _) => ChatMessage::system(content_text(content)),
                    // A tool message without a result shape has no call id
                    // to pair it with; it keeps its role.
                    (Role::Tool, _) => ChatMessage::tool(String::new(), content_text(content)),
                    _ => ChatMessage::user(content_text(content)),
                });
            }
        }
    }
    flush(&mut out, assistant);
    out
}

/// A message's text: plain text verbatim, mixed parts' text joined. Parts
/// the text model cannot read (images, audio) are left out.
fn content_text(content: &MessageContent) -> String {
    match content {
        MessageContent::Text(text) => text.clone(),
        MessageContent::ContentParts(parts) => parts
            .iter()
            .filter_map(|p| match p {
                ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect(),
        MessageContent::ToolCall { .. } | MessageContent::ToolResult { .. } => String::new(),
    }
}

/// One sven tool schema as brain's.
fn tool_schema(tool: &sven_sdk::model::ToolSchema) -> ToolSchema {
    ToolSchema::new(&tool.name, &tool.description, tool.parameters.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sven_sdk::model::{FunctionCall, ToolResultContent};

    fn message(role: Role, content: MessageContent) -> Message {
        Message { role, content }
    }

    fn user(text: &str) -> Message {
        message(Role::User, MessageContent::Text(text.into()))
    }

    const SAMPLING: Sampling = super::super::AGENT_SAMPLING;

    /// A request's output budget is a contract with the caller: `explore`
    /// asks for 32k because a section's facts reply must arrive as ONE
    /// complete JSON object - a reply truncated at the provider's own
    /// default mid-string is a parse failure and the section's facts are
    /// lost (observed: "EOF while parsing a string"). The override must
    /// reach the engine; brain bounds it by the context.
    #[test]
    fn a_requests_output_budget_overrides_the_provider_default() {
        let req = CompletionRequest {
            messages: vec![user("q")],
            max_output_tokens_override: Some(32_768),
            ..CompletionRequest::default()
        };
        assert_eq!(
            output_budget(&req, &SAMPLING),
            32_768,
            "the request's output budget must override the provider default"
        );
        // No override: the provider default stands.
        let plain = CompletionRequest {
            messages: vec![user("q")],
            ..CompletionRequest::default()
        };
        assert_eq!(output_budget(&plain, &SAMPLING), SAMPLING.max_new_tokens);
    }

    /// The whole tool exchange reaches the prompt brain renders: the
    /// assistant's text and call as one turn, the result paired with it,
    /// and the tools on offer.
    #[test]
    fn a_tool_exchange_renders_as_the_model_produced_it() {
        let req = CompletionRequest {
            messages: vec![
                message(Role::System, MessageContent::Text("be terse".into())),
                user("do the thing"),
                message(Role::Assistant, MessageContent::Text("writing it".into())),
                message(
                    Role::Assistant,
                    MessageContent::ToolCall {
                        tool_call_id: "call_1".into(),
                        function: FunctionCall {
                            name: "write".into(),
                            arguments: r#"{"path":"a.txt"}"#.into(),
                        },
                    },
                ),
                message(
                    Role::Tool,
                    MessageContent::ToolResult {
                        tool_call_id: "call_1".into(),
                        content: ToolResultContent::Text("wrote 5 bytes".into()),
                    },
                ),
            ],
            tools: vec![sven_sdk::model::ToolSchema {
                name: "read".into(),
                description: "read a file".into(),
                parameters: serde_json::json!({"type":"object","properties":{"path":{"type":"string"}}}),
                is_mcp: false,
            }],
            ..CompletionRequest::default()
        };
        let messages = chat_messages(&req.messages);
        assert_eq!(
            messages.len(),
            4,
            "system, user, one assistant turn, tool result: {messages:?}"
        );
        assert_eq!(
            messages[2],
            ChatMessage::assistant("writing it").with_tool_calls(vec![ToolCall {
                id: "call_1".into(),
                name: "write".into(),
                arguments: r#"{"path":"a.txt"}"#.into(),
            }])
        );
        assert_eq!(messages[3], ChatMessage::tool("call_1", "wrote 5 bytes"));

        // Rendered by brain's own template - the consumer this feeds.
        let prompt = chat_request(&req, &SAMPLING).render_prompt().unwrap();
        for expected in [
            "be terse",
            "do the thing",
            "writing it",
            r#""name": "write""#,
            "a.txt",
            "wrote 5 bytes",
            "read a file",
        ] {
            assert!(
                prompt.contains(expected),
                "{expected:?} missing from the prompt:\n{prompt}"
            );
        }
    }

    #[test]
    fn mixed_content_parts_keep_their_text() {
        let content = MessageContent::ContentParts(vec![
            ContentPart::Text {
                text: "see ".into(),
            },
            ContentPart::Text {
                text: "this".into(),
            },
        ]);
        assert_eq!(content_text(&content), "see this");
    }
}
