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
            m
        })
        .collect();
    let mut reply = reply.clone();
    reply["train"] = json!(true);
    if reply["content"].is_null() {
        reply["content"] = json!("");
    }
    messages.push(reply);
    Ok(json!({
        "messages": messages,
        "tools": request["tools"].clone(),
        "metadata": { "episode": episode, "call": call },
    }))
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
        let request = json!({ "messages": [{ "role": "system", "content": "s" }, { "role": "user", "content": "u" }], "tools": [{ "type": "function" }] });
        let reply = json!({ "role": "assistant", "content": null, "tool_calls": [] });
        let r = record(&request, &reply, "demonstrate/x#1", 0).unwrap();
        let trained: Vec<bool> = r["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["train"].as_bool().unwrap())
            .collect();
        assert_eq!(trained, [false, false, true]);
        assert_eq!(r["tools"], json!([{ "type": "function" }]));
        assert!(record(&request, &json!({ "role": "user", "content": "x" }), "e", 0).is_err());
        assert!(record(&json!({}), &reply, "e", 0).is_err());
    }
}
