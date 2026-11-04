// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training data for multi-role agent systems,
// where each model role learns from the requests it was actually sent, for
// its clients. If your team needs expertise in fine-tuning the roles of an
// agent harness, you can procure our services by sending an email to
// info@swedishembedded.com.

//! A conversation's model calls are stored as differences and read back whole.

use serde_json::{json, Value};
use splinter_core::harness::{
    image_url, is_media, message_parts, ModelCall, ModelResponse, RecordedCalls,
};

fn call(seq: u64, messages: Vec<Value>, tools: Vec<Value>) -> ModelCall {
    ModelCall {
        seq,
        messages,
        tools,
        response: ModelResponse {
            text: format!("answer {seq}"),
            reasoning: None,
            tool_calls: Vec::new(),
        },
    }
}

fn say(role: &str, text: &str) -> Value {
    json!({"role": role, "content": text})
}

#[test]
fn growing_requests_round_trip_and_share_what_they_repeat() {
    let tool = |n: &str| json!({"name": n});
    let calls = vec![
        call(
            1,
            vec![say("system", "s"), say("user", "u")],
            vec![tool("a")],
        ),
        call(
            2,
            vec![say("system", "s"), say("user", "u"), say("assistant", "x")],
            vec![tool("a")],
        ),
        // The context was rewritten and a tool was loaded.
        call(
            3,
            vec![say("system", "s"), say("user", "summary")],
            vec![tool("a"), tool("b")],
        ),
    ];
    let recorded = RecordedCalls::record("ABC", &calls);
    assert_eq!(recorded.source_sha256, "abc");
    let stored = recorded.to_value();
    // The second request adds one message to the first's two; the third
    // keeps only the system turn.
    assert_eq!(stored["calls"][1]["shared"], 2);
    assert_eq!(stored["calls"][1]["messages"].as_array().unwrap().len(), 1);
    assert!(stored["calls"][1].get("tools").is_none());
    assert_eq!(stored["calls"][2]["shared"], 1);
    let back: RecordedCalls = serde_json::from_value(stored).unwrap();
    assert_eq!(back.calls().unwrap(), calls);
}

#[test]
fn a_record_sharing_more_than_the_request_before_is_refused() {
    let recorded = RecordedCalls::record("x", &[call(1, vec![say("user", "u")], vec![])]);
    let mut value = recorded.to_value();
    let second = json!({"seq": 2, "shared": 5, "messages": [], "response": {"text": ""}});
    value["calls"].as_array_mut().unwrap().push(second);
    let broken: RecordedCalls = serde_json::from_value(value).unwrap();
    assert!(broken.calls().is_none());
}

#[test]
fn images_are_found_in_a_user_message_and_in_a_tool_result() {
    let png = "data:image/png;base64,AAAA";
    let user = json!({"role": "user", "content": [
        {"type": "text", "text": "look"},
        {"type": "image", "image_url": png},
    ]});
    let tool = json!({"role": "tool", "content": {"tool_call_id": "c", "content": [
        {"type": "text", "text": "Attached"},
        {"type": "image", "image_url": png},
    ]}});
    let plain = json!({"role": "tool", "content": {"tool_call_id": "c", "content": "text"}});
    for message in [&user, &tool] {
        let images: Vec<&str> = message_parts(message).filter_map(image_url).collect();
        assert_eq!(images, [png]);
        assert!(message_parts(message).any(is_media));
    }
    assert_eq!(message_parts(&plain).count(), 0);
    assert_eq!(message_parts(&say("user", "u")).count(), 0);
}
