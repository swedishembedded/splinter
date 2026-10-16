// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that act through tools a caller
// defines, for its clients. If your team needs expertise in measuring and
// tracing what a model does with a tool it has never seen, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec: a person talks to a trained model through tools they define. A tool
//! is a name, a description and typed arguments in a file; a call is checked
//! against it (a missing argument, a value outside the options, an unknown
//! tool is refused with a message the model can read and correct); an answer
//! to a call goes back to the model, which may call again or speak; and every
//! exchange is a trace of what was said, called, answered and spent.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::sync::{Arc, Mutex};

use persona_chat::decoding::Decoding;
use persona_chat::session::Session;
use persona_chat::tools::{decide, read_tools};
use persona_chat::trace;
use serde_json::json;
use splinter_sdk::agent::sven::model::{
    CompletionRequest, ModelProvider, ResponseEvent, ResponseStream,
};

/// A model that plays back one scripted reply per request, and keeps the
/// requests it was sent.
struct Script {
    replies: Mutex<Vec<Vec<ResponseEvent>>>,
    seen: Mutex<Vec<CompletionRequest>>,
}

impl Script {
    fn new(replies: Vec<Vec<ResponseEvent>>) -> Arc<Self> {
        Arc::new(Self {
            replies: Mutex::new(replies),
            seen: Mutex::new(Vec::new()),
        })
    }
}

#[async_trait::async_trait]
impl ModelProvider for Script {
    fn name(&self) -> &str {
        "script"
    }
    fn model_name(&self) -> &str {
        "script-1"
    }
    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        self.seen.lock().unwrap().push(req);
        let mut replies = self.replies.lock().unwrap();
        let events = if replies.is_empty() {
            vec![ResponseEvent::Done]
        } else {
            replies.remove(0)
        };
        Ok(Box::pin(futures::stream::iter(events.into_iter().map(Ok))))
    }
}

fn call(id: &str, name: &str, arguments: serde_json::Value) -> Vec<ResponseEvent> {
    vec![
        ResponseEvent::ToolCall {
            index: 0,
            id: id.into(),
            name: name.into(),
            arguments: arguments.to_string(),
        },
        ResponseEvent::Done,
    ]
}

fn speaks(text: &str) -> Vec<ResponseEvent> {
    vec![ResponseEvent::TextDelta(text.into()), ResponseEvent::Done]
}

/// A conversation offering `decide` as a tool whose answer the model goes on
/// from, so that the rounds after a call can be told.
fn session(script: &Arc<Script>) -> Session {
    let provider: Arc<dyn ModelProvider> = script.clone();
    let mut tool = decide(&["capitalism".into(), "communism".into()]);
    tool.final_call = false;
    Session::new(provider, "You are a steward.", vec![tool])
}

/// A conversation offering `decide` as it is by default: the last word.
fn final_session(script: &Arc<Script>) -> Session {
    let provider: Arc<dyn ModelProvider> = script.clone();
    Session::new(
        provider,
        "You are a steward.",
        vec![decide(&["capitalism".into(), "communism".into()])],
    )
}

#[test]
fn a_tool_file_defines_typed_arguments_and_is_refused_when_it_cannot_be_offered() {
    let tools = read_tools(
        r#"[{"name": "pick_side", "description": "Pick a side.",
             "parameters": [
               {"name": "side", "type": "enum", "values": ["left", "right"]},
               {"name": "weight", "type": "number", "required": false}
             ], "result": "Noted."}]"#,
    )
    .unwrap();
    let schema = tools[0].schema();
    assert_eq!(schema.name, "pick_side");
    assert_eq!(
        schema.parameters["properties"]["side"]["enum"],
        json!(["left", "right"])
    );
    assert_eq!(schema.parameters["required"], json!(["side"]));

    for (bad, why) in [
        (r#"[{"name": "", "description": "x"}]"#, "name"),
        (
            r#"[{"name": "a", "description": "x", "parameters": [{"name": "p", "type": "enum"}]}]"#,
            "values",
        ),
        (
            r#"[{"name": "a", "description": "x"}, {"name": "a", "description": "y"}]"#,
            "twice",
        ),
        (
            r#"[{"name": "a", "description": "x", "parameters": [{"name": "p", "type": "blob"}]}]"#,
            "type",
        ),
    ] {
        let error = read_tools(bad).unwrap_err().to_string();
        assert!(error.contains(why), "{bad}: {error}");
    }
}

#[tokio::test]
async fn a_call_is_answered_and_the_model_speaks_after_it() {
    let script = Script::new(vec![
        call(
            "c1",
            "decide",
            json!({"choice": "capitalism", "motivation": "it rewards effort"}),
        ),
        speaks("I have decided."),
    ]);
    let mut chat = session(&script);
    let exchange = chat.say("Which is the obvious choice?").await.unwrap();
    assert_eq!(
        exchange.rounds.len(),
        2,
        "the call, then the words after it"
    );
    let first = &exchange.rounds[0].calls[0];
    assert_eq!(first.name, "decide");
    assert!(first.valid);
    assert_eq!(first.arguments["choice"], "capitalism");
    assert_eq!(first.result, "Recorded.");
    assert_eq!(exchange.rounds[1].text, "I have decided.");

    // The second request carried the answer to the call.
    let seen = script.seen.lock().unwrap();
    let last = format!("{:?}", seen[1].messages);
    assert!(last.contains("Recorded."), "{last}");
    assert!(last.contains("Which is the obvious choice?"), "{last}");
    // The offered tool is the one defined.
    assert_eq!(seen[0].tools[0].name, "decide");
}

#[tokio::test]
async fn a_call_the_tool_does_not_allow_is_refused_in_words_the_model_can_correct() {
    let script = Script::new(vec![
        call(
            "c1",
            "decide",
            json!({"choice": "monarchy", "motivation": "m"}),
        ),
        call("c2", "decide", json!({"choice": "communism"})),
        call(
            "c3",
            "decide",
            json!({"choice": "communism", "motivation": "shared"}),
        ),
        call("c4", "shrug", json!({})),
        speaks("done"),
    ]);
    let mut chat = session(&script).rounds(5);
    let exchange = chat.say("Which?").await.unwrap();
    let calls: Vec<_> = exchange.rounds.iter().flat_map(|r| &r.calls).collect();
    assert!(
        !calls[0].valid && calls[0].result.contains("monarchy"),
        "{:?}",
        calls[0]
    );
    assert!(
        !calls[1].valid && calls[1].result.contains("motivation"),
        "a missing argument is named: {:?}",
        calls[1]
    );
    assert!(calls[2].valid);
    assert!(
        !calls[3].valid && calls[3].result.contains("shrug"),
        "an unknown tool is named"
    );
    assert_eq!(exchange.rounds.len(), 5);
}

#[tokio::test]
async fn a_model_that_keeps_calling_is_stopped_after_the_rounds_allowed() {
    let script = Script::new(
        (0..10)
            .map(|n| {
                call(
                    &format!("c{n}"),
                    "decide",
                    json!({"choice": "communism", "motivation": "again"}),
                )
            })
            .collect(),
    );
    let mut chat = session(&script).rounds(3);
    let exchange = chat.say("Which?").await.unwrap();
    assert_eq!(exchange.rounds.len(), 3);
    assert!(exchange.stopped, "the exchange says it was cut off");
}

#[tokio::test]
async fn a_conversation_keeps_what_was_said_and_a_trace_records_each_exchange() {
    let script = Script::new(vec![speaks("Ask me anything."), speaks("Left.")]);
    let mut chat = session(&script);
    chat.say("Hello").await.unwrap();
    let exchange = chat.say("Left or right?").await.unwrap();
    let seen = script.seen.lock().unwrap();
    let second = format!("{:?}", seen[1].messages);
    assert!(
        second.contains("Hello") && second.contains("Ask me anything."),
        "{second}"
    );

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("trace.jsonl");
    let log = trace::open(&path).unwrap();
    trace::record(&log, "q2", "script-1", None, &exchange).unwrap();
    let rows = trace::read(&path).unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].id, "q2");
    assert_eq!(rows[0].exchange.prompt, "Left or right?");
    let shown = trace::render(&exchange);
    assert!(
        shown.contains("Left or right?") && shown.contains("Left."),
        "{shown}"
    );
}

#[tokio::test]
async fn a_valid_call_to_a_final_tool_ends_the_exchange_and_an_invalid_one_does_not() {
    let script = Script::new(vec![
        call(
            "c1",
            "decide",
            json!({"choice": "monarchy", "motivation": "m"}),
        ),
        call(
            "c2",
            "decide",
            json!({"choice": "communism", "motivation": "shared"}),
        ),
        call(
            "c3",
            "decide",
            json!({"choice": "communism", "motivation": "again"}),
        ),
    ]);
    let mut chat = final_session(&script).rounds(5);
    let exchange = chat.say("Which?").await.unwrap();
    assert_eq!(
        exchange.rounds.len(),
        2,
        "the refused call is tried again; the accepted one is the end"
    );
    assert!(
        !exchange.stopped,
        "ending on a final call is not being cut off"
    );
    assert_eq!(
        script.seen.lock().unwrap().len(),
        2,
        "no turn is spent after it"
    );
}

#[test]
fn a_tool_is_final_when_its_file_says_so_and_decide_always_is() {
    let tools = read_tools(
        r#"[{"name": "look_up", "description": "Look something up."},
            {"name": "rule", "description": "Rule.", "final": true}]"#,
    )
    .unwrap();
    assert!(!tools[0].final_call && tools[1].final_call);
    assert!(decide(&["a".into()]).final_call);
}

#[tokio::test]
async fn every_trace_line_names_the_model_and_the_seed_its_exchange_was_decoded_with() {
    let script = Script::new(vec![speaks("Left."), speaks("Right.")]);
    let mut chat = session(&script);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("trace.jsonl");
    let log = trace::open(&path).unwrap();
    let session_seed = splinter_sdk::seed::resolve("seed", None);
    for (n, id) in [(1, "a"), (2, "b")] {
        let decoding = Decoding::for_exchange(session_seed, n);
        let exchange = chat.say("Left or right?").await.unwrap();
        trace::record(&log, id, "Qwen/Qwen3-8B+maren", Some(&decoding), &exchange).unwrap();
    }
    let rows = trace::read(&path).unwrap();
    assert!(rows.iter().all(|r| r.model == "Qwen/Qwen3-8B+maren"));
    let seeds: Vec<u64> = rows.iter().map(|r| r.decoding.unwrap().seed).collect();
    assert_eq!(seeds[0], Decoding::for_exchange(session_seed, 1).seed);
    assert_ne!(seeds[0], seeds[1], "an exchange draws its own stream");
    assert_eq!(
        rows[0].decoding.unwrap().temperature,
        Decoding::for_exchange(session_seed, 1).temperature
    );
}
