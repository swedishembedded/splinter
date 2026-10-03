// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements coding agents that run their model
// in-process - no separately versioned serving process to drift from the
// build. If your team needs expertise in local inference integration or
// provider abstractions, you can procure our services by sending an email
// to info@swedishembedded.com.

//! brain's chat reply as sven's event stream: the deltas while they are
//! generated, then the tool calls, usage and the ending.

use brain::chat::{ChatDelta, FinishReason};
use brain::ChatResponse;
use sven_sdk::model::ResponseEvent;

/// A streamed piece of the reply as the event sven reads. `None` for a kind
/// of delta this provider does not forward.
pub(super) fn delta_event(delta: ChatDelta) -> Option<ResponseEvent> {
    match delta {
        ChatDelta::Text(text) => Some(ResponseEvent::TextDelta(text)),
        ChatDelta::Reasoning(text) => Some(ResponseEvent::ThinkingDelta(text)),
        _ => None,
    }
}

/// The finished reply's non-text events. Visible text and reasoning were
/// streamed as they were generated; the tail carries the tool calls -
/// complete, post-finish - then usage, then how the reply ended.
pub(super) fn events_from(response: ChatResponse) -> Vec<anyhow::Result<ResponseEvent>> {
    let mut events: Vec<anyhow::Result<ResponseEvent>> = Vec::new();
    for (index, call) in (0u32..).zip(response.tool_calls) {
        events.push(Ok(ResponseEvent::ToolCall {
            index,
            id: call.id,
            name: call.name,
            arguments: call.arguments,
        }));
    }
    // Usage is reported only when the engine counted it: a missing count is
    // unmeasured, not zero. This engine has no prompt cache, so its cache
    // counts are a true zero.
    if let (Some(input_tokens), Some(output_tokens)) = (
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
    ) {
        events.push(Ok(ResponseEvent::Usage {
            input_tokens,
            output_tokens,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
            cost_usd: None,
        }));
    }
    match response.finish_reason {
        FinishReason::Stop | FinishReason::StopSequence | FinishReason::ToolCalls => {}
        // A truncated reply says so before it ends, so the agent can tell
        // cut-off tool arguments from finished ones.
        FinishReason::Length => events.push(Ok(ResponseEvent::MaxTokens)),
        // A partial reply is not a finished one: a cancelled turn, or one
        // that missed a tool demand, fails rather than passing for an answer.
        reason @ (FinishReason::Cancelled | FinishReason::ToolChoiceUnmet) => {
            events.push(Ok(ResponseEvent::Error(format!(
                "generation ended without a complete reply: {}",
                reason.as_str()
            ))));
            return events;
        }
    }
    events.push(Ok(ResponseEvent::Done));
    events
}

#[cfg(test)]
mod tests {
    use super::*;
    use brain::chat::{ChatUsage, ToolCall};

    fn response(finish_reason: FinishReason, usage: ChatUsage) -> ChatResponse {
        ChatResponse {
            text: "did it".into(),
            reasoning: String::new(),
            tool_calls: vec![ToolCall {
                id: "c1".into(),
                name: "write".into(),
                arguments: "{}".into(),
            }],
            finish_reason,
            usage,
        }
    }

    fn counted() -> ChatUsage {
        ChatUsage {
            prompt_tokens: Some(120),
            completion_tokens: Some(34),
        }
    }

    #[test]
    fn a_finished_replys_tail_is_tool_calls_usage_and_done() {
        let events = events_from(response(FinishReason::ToolCalls, counted()));
        // The visible text was streamed while it was generated; the tail
        // must not repeat it.
        assert_eq!(events.len(), 3, "tool call, usage, done: {events:?}");
        assert!(matches!(
            &events[0],
            Ok(ResponseEvent::ToolCall { index: 0, id, name, arguments })
                if id == "c1" && name == "write" && arguments == "{}"
        ));
        assert!(matches!(
            &events[1],
            Ok(ResponseEvent::Usage { input_tokens, output_tokens, cost_usd: None, .. })
                if *input_tokens == 120 && *output_tokens == 34
        ));
        assert!(matches!(events[2], Ok(ResponseEvent::Done)));
    }

    /// A reply that carries no token counts reports no usage at all: a
    /// missing count is unmeasured, and a zero would read as free.
    #[test]
    fn a_reply_without_token_counts_reports_no_usage() {
        let mut reply = response(FinishReason::Stop, ChatUsage::default());
        reply.tool_calls.clear();
        let events = events_from(reply);
        assert_eq!(events.len(), 1, "done only: {events:?}");
        assert!(matches!(events[0], Ok(ResponseEvent::Done)));
    }

    /// How a reply ended reaches the agent: a budget cut is announced before
    /// the end, and a cancelled reply is a failure, never a finished answer.
    #[test]
    fn a_truncated_reply_says_so_and_a_cancelled_one_fails() {
        let truncated = events_from(response(FinishReason::Length, counted()));
        assert!(matches!(
            truncated.as_slice(),
            [.., Ok(ResponseEvent::MaxTokens), Ok(ResponseEvent::Done)]
        ));
        let cancelled = events_from(response(FinishReason::Cancelled, counted()));
        assert!(
            matches!(cancelled.last(), Some(Ok(ResponseEvent::Error(what))) if what.contains("cancelled")),
            "{cancelled:?}"
        );
        assert!(!cancelled
            .iter()
            .any(|e| matches!(e, Ok(ResponseEvent::Done))));
    }

    #[test]
    fn deltas_stream_as_text_and_thinking() {
        assert!(matches!(
            delta_event(ChatDelta::Text("a".into())),
            Some(ResponseEvent::TextDelta(t)) if t == "a"
        ));
        assert!(matches!(
            delta_event(ChatDelta::Reasoning("r".into())),
            Some(ResponseEvent::ThinkingDelta(t)) if t == "r"
        ));
    }
}
