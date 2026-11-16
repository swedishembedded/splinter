// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training records taken from the exact
// requests an agent runtime sends, so a model is fine-tuned on what it will
// see when deployed, for its clients. If your team needs expertise in agent
// fine-tuning data, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Training records from an episode's recorded model calls (the wire).
//!
//! One record per call, as brain's `generic-messages-v2` chat records: the
//! conversation exactly as the runtime sent it (system prompt, tools and
//! every earlier turn) as context, and the reply that call received as the
//! one supervised message. Only episodes the verifier accepted are turned
//! into records, so every supervised action belongs to a correct episode.

use serde_json::{json, Value};

use crate::ViewError;

/// The record of one call: `request`'s messages as context and `reply`, the
/// assistant message it was answered with, supervised. `episode` names the
/// attempt it came from.
///
/// # Errors
/// The request holds no messages, or the reply is not an assistant message.
pub fn record(
    request: &Value,
    reply: &Value,
    episode: &str,
    call: usize,
) -> Result<Value, ViewError> {
    let refuse = |why: &str| ViewError::Call(format!("{episode} call {call}: {why}"));
    if reply["role"] != "assistant" {
        return Err(refuse("the reply is not an assistant message"));
    }
    let mut messages: Vec<Value> = request["messages"]
        .as_array()
        .ok_or_else(|| refuse("the request holds no messages"))?
        .iter()
        .map(|m| {
            let mut m = m.clone();
            m["train"] = json!(false);
            plain_content(&mut m);
            m
        })
        .collect();
    let mut reply = reply.clone();
    reply["train"] = json!(true);
    plain_content(&mut reply);
    messages.push(reply);
    unique_call_ids(&mut messages);
    Ok(json!({
        "messages": messages,
        "tools": request["tools"].clone(),
        "metadata": { "episode": episode, "call": call },
    }))
}

/// Renumbers the tool calls of a conversation `call_0`, `call_1`, ... in
/// order, and the results that answer them to match: a model that reuses one
/// id for every call leaves results that cannot be paired by id, and brain
/// refuses such a record. A result is paired to the oldest unanswered call
/// with its old id.
fn unique_call_ids(messages: &mut [Value]) {
    let mut next = 0usize;
    let mut open: Vec<(String, String)> = Vec::new();
    for message in messages.iter_mut() {
        if let Some(calls) = message.get_mut("tool_calls").and_then(Value::as_array_mut) {
            for call in calls {
                let old = call["id"].as_str().unwrap_or_default().to_string();
                let new = format!("call_{next}");
                next += 1;
                call["id"] = json!(new);
                open.push((old, new));
            }
        } else if message["role"] == "tool" {
            let old = message["tool_call_id"]
                .as_str()
                .unwrap_or_default()
                .to_string();
            let at = open
                .iter()
                .position(|(o, _)| *o == old)
                .or(if open.is_empty() { None } else { Some(0) });
            if let Some(at) = at {
                message["tool_call_id"] = json!(open.remove(at).1);
            }
        }
    }
}

/// Gives `message` the plain-text `content` a chat record holds: a missing
/// or null content (an assistant turn that only calls tools) is empty, and a
/// list of text parts is joined.
fn plain_content(message: &mut Value) {
    let text = match &message["content"] {
        Value::String(_) => return,
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p["text"].as_str())
            .collect::<String>(),
        _ => String::new(),
    };
    message["content"] = json!(text);
}

/// Every `(request, reply)` call of an accepted episode as JSON lines.
///
/// # Errors
/// As [`record`], for any call.
pub fn episode_lines(calls: &[(Value, Value)], episode: &str) -> Result<String, ViewError> {
    let mut out = String::new();
    for (call, (request, reply)) in calls.iter().enumerate() {
        out.push_str(&record(request, reply, episode, call)?.to_string());
        out.push('\n');
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_reply_is_supervised_and_the_context_is_the_request() {
        let request = json!({ "messages": [
            { "role": "system", "content": "s" },
            { "role": "user", "content": [{ "type": "text", "text": "u" }] },
            { "role": "assistant", "tool_calls": [] },
        ], "tools": [{ "type": "function" }] });
        let reply = json!({ "role": "assistant", "content": null, "tool_calls": [] });
        let r = record(&request, &reply, "demonstrate/x#1", 0).unwrap();
        let trained: Vec<bool> = r["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["train"].as_bool().unwrap())
            .collect();
        assert_eq!(trained, [false, false, false, true]);
        let contents: Vec<&str> = r["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["content"].as_str().unwrap())
            .collect();
        assert_eq!(
            contents,
            ["s", "u", "", ""],
            "every message has plain-text content"
        );
        assert_eq!(r["tools"], json!([{ "type": "function" }]));
        assert!(record(&request, &json!({ "role": "user", "content": "x" }), "e", 0).is_err());
        let reused = json!({ "messages": [
            { "role": "user", "content": "u" },
            { "role": "assistant", "content": "", "tool_calls": [{ "id": "c", "type": "function", "function": { "name": "a", "arguments": "{}" } }] },
            { "role": "tool", "tool_call_id": "c", "content": "1" },
            { "role": "assistant", "content": "", "tool_calls": [{ "id": "c", "type": "function", "function": { "name": "b", "arguments": "{}" } }] },
            { "role": "tool", "tool_call_id": "c", "content": "2" },
        ] });
        let r = record(&reused, &reply, "e", 0).unwrap();
        let m = r["messages"].as_array().unwrap();
        assert_eq!(
            (
                m[1]["tool_calls"][0]["id"].as_str(),
                m[2]["tool_call_id"].as_str()
            ),
            (Some("call_0"), Some("call_0"))
        );
        assert_eq!(
            (
                m[3]["tool_calls"][0]["id"].as_str(),
                m[4]["tool_call_id"].as_str()
            ),
            (Some("call_1"), Some("call_1"))
        );
        assert!(record(&json!({}), &reply, "e", 0).is_err());
    }
}
