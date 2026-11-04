// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that act through tools a caller
// defines, for its clients. If your team needs expertise in measuring and
// tracing what a model does with a tool it has never seen, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec: a model is put a conversation and a set of tools the caller defines,
//! and what comes back is everything it did in that turn - the words it said,
//! each tool call it made with its arguments read as JSON, and what it cost -
//! however the stream chunked it. A reply with no call is a turn with none; a
//! stream that fails is an error, not a turn.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::sync::{Arc, Mutex};

use serde_json::json;
use splinter_agent::tool_turn::tool_turn;
use sven_sdk::model::{
    CompletionRequest, Message, ModelProvider, ResponseEvent, ResponseStream, ToolSchema,
};

/// A provider that streams a fixed list of events and keeps the request.
struct Streaming {
    events: Mutex<Vec<anyhow::Result<ResponseEvent>>>,
    seen: Mutex<Vec<CompletionRequest>>,
}

impl Streaming {
    fn new(events: Vec<anyhow::Result<ResponseEvent>>) -> Arc<Self> {
        Arc::new(Self {
            events: Mutex::new(events),
            seen: Mutex::new(Vec::new()),
        })
    }
}

#[async_trait::async_trait]
impl ModelProvider for Streaming {
    fn name(&self) -> &str {
        "streaming"
    }
    fn model_name(&self) -> &str {
        "streaming-1"
    }
    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        self.seen.lock().unwrap().push(req);
        let events = std::mem::take(&mut *self.events.lock().unwrap());
        Ok(Box::pin(futures::stream::iter(events)))
    }
}

fn decide() -> ToolSchema {
    ToolSchema {
        name: "decide".into(),
        description: "Record the decision you take.".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "choice": {"type": "string", "enum": ["capitalism", "communism"]},
                "motivation": {"type": "string"}
            },
            "required": ["choice", "motivation"]
        }),
        ..Default::default()
    }
}

fn call(index: u32, id: &str, name: &str, arguments: &str) -> anyhow::Result<ResponseEvent> {
    Ok(ResponseEvent::ToolCall {
        index,
        id: id.into(),
        name: name.into(),
        arguments: arguments.into(),
    })
}

#[tokio::test]
async fn a_call_split_across_chunks_comes_back_whole_with_the_words_and_the_cost() {
    let provider = Streaming::new(vec![
        Ok(ResponseEvent::TextDelta("Weighing it, ".into())),
        Ok(ResponseEvent::TextDelta("I decide.".into())),
        call(0, "c1", "decide", r#"{"choice": "capi"#),
        call(0, "", "", r#"talism", "motivation": "freedom"}"#),
        Ok(ResponseEvent::Usage {
            input_tokens: 120,
            output_tokens: 31,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
            cost_usd: None,
        }),
        Ok(ResponseEvent::Done),
    ]);
    let dynamic: Arc<dyn ModelProvider> = provider.clone();
    let turn = tool_turn(
        &dynamic,
        vec![Message::system("You decide."), Message::user("Which?")],
        vec![decide()],
        256,
    )
    .await
    .unwrap();
    assert_eq!(turn.text, "Weighing it, I decide.");
    assert_eq!(turn.calls.len(), 1);
    assert_eq!(turn.calls[0].name, "decide");
    assert_eq!(turn.calls[0].id, "c1");
    assert_eq!(
        turn.calls[0].arguments,
        json!({"choice": "capitalism", "motivation": "freedom"})
    );
    assert_eq!((turn.input_tokens, turn.output_tokens), (120, 31));

    let seen = provider.seen.lock().unwrap();
    assert_eq!(seen[0].tools[0].name, "decide");
    assert_eq!(seen[0].max_output_tokens_override, Some(256));
}

#[tokio::test]
async fn parallel_calls_keep_their_own_arguments_and_a_reply_without_one_has_none() {
    let provider: Arc<dyn ModelProvider> = Streaming::new(vec![
        call(
            1,
            "b",
            "decide",
            r#"{"choice": "communism", "motivation": "m"}"#,
        ),
        call(
            0,
            "a",
            "decide",
            r#"{"choice": "capitalism", "motivation": "n"}"#,
        ),
        Ok(ResponseEvent::Done),
    ]);
    let turn = tool_turn(&provider, vec![Message::user("Which?")], vec![decide()], 64)
        .await
        .unwrap();
    let ids: Vec<&str> = turn.calls.iter().map(|c| c.id.as_str()).collect();
    assert_eq!(
        ids,
        ["a", "b"],
        "calls come back in the model's own slot order"
    );

    let silent: Arc<dyn ModelProvider> = Streaming::new(vec![
        Ok(ResponseEvent::TextDelta("I would rather not say.".into())),
        Ok(ResponseEvent::Done),
    ]);
    let turn = tool_turn(&silent, vec![Message::user("Which?")], vec![decide()], 64)
        .await
        .unwrap();
    assert!(turn.calls.is_empty());
    assert_eq!(turn.text, "I would rather not say.");
}

#[tokio::test]
async fn arguments_that_are_not_json_are_kept_as_text_and_a_failed_stream_is_an_error() {
    let provider: Arc<dyn ModelProvider> = Streaming::new(vec![
        call(0, "x", "decide", r#"{"choice": "capital"#),
        Ok(ResponseEvent::MaxTokens),
        Ok(ResponseEvent::Done),
    ]);
    let turn = tool_turn(&provider, vec![Message::user("Which?")], vec![decide()], 8)
        .await
        .unwrap();
    assert_eq!(turn.calls[0].arguments, serde_json::Value::Null);
    assert_eq!(turn.calls[0].raw_arguments, r#"{"choice": "capital"#);
    assert!(turn.truncated, "the model ran out of tokens");

    let failing: Arc<dyn ModelProvider> =
        Streaming::new(vec![Ok(ResponseEvent::Error("device lost".into()))]);
    let error = tool_turn(&failing, vec![Message::user("Which?")], vec![decide()], 8)
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("device lost"), "{error}");
}
