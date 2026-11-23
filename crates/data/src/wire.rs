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
//!
//! A record is a function of what the policy saw and did, never of when or
//! how fast: the same episode written twice is the same bytes, so a dataset's
//! digest, and every key derived from it, is reproducible. Three things on
//! the wire vary between runs of one episode and are made canonical here: a
//! shell result's measured duration ([`stable_tool_result`]), the ids a
//! server gives tool calls (renumbered `call_0`, `call_1`, ... in order), and
//! the attempt an episode ran as (the caller names the episode, not the
//! attempt).

use serde_json::{json, Value};

use crate::ViewError;

/// The record of one call: `request`'s messages as context and `reply`, the
/// assistant message it was answered with, supervised. `episode` names the
/// episode it came from (not its attempt, so a rerun writes the same bytes).
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
        // A request that offered no tools (a runtime's own follow-up call)
        // offered an empty list: brain reads `tools` as a list.
        "tools": request.get("tools").filter(|t| !t.is_null()).cloned().unwrap_or_else(|| json!([])),
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

/// Makes `message` the message a chat record holds: its role, plain-text
/// `content` (a missing or null content, an assistant turn that only calls
/// tools, is empty; a list of text parts is joined), its tool calls as
/// `{id, type, function: {name, arguments}}` with the arguments as JSON
/// text, and the call a tool result answers. Fields a server adds to a reply
/// (`refusal`, `annotations`, ...) are dropped: brain refuses a record
/// that carries one.
fn plain_content(message: &mut Value) {
    let content = match &message["content"] {
        Value::String(text) => text.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p["text"].as_str())
            .collect::<String>(),
        _ => String::new(),
    };
    let mut clean = json!({ "role": message["role"].clone(), "content": content });
    if let Some(calls) = message["tool_calls"].as_array() {
        let calls: Vec<Value> = calls
            .iter()
            .map(|c| {
                let arguments = match &c["function"]["arguments"] {
                    Value::String(text) => text.clone(),
                    other => other.to_string(),
                };
                json!({ "id": c["id"].clone(), "type": "function",
                        "function": { "name": c["function"]["name"].clone(), "arguments": arguments } })
            })
            .collect();
        clean["tool_calls"] = json!(calls);
    }
    if let Some(id) = message.get("tool_call_id") {
        clean["tool_call_id"] = id.clone();
    }
    if message["role"] == "tool" {
        clean["content"] = json!(stable_tool_result(&content));
    }
    if let Some(train) = message.get("train") {
        clean["train"] = train.clone();
    }
    *message = clean;
}

/// The marker of a shell result's status line, as the agent runtime renders
/// it: `Exit: <code> | Duration: <seconds>s`, optionally followed by more
/// `| ...` fields.
const DURATION_FIELD: &str = " | Duration: ";

/// The duration a canonical record states: what a workspace command of a few
/// milliseconds is shown as, so a record reads like the results the model is
/// served at inference.
// perf-number: the canonical value of a masked field, not a measurement.
const CANONICAL_DURATION: &str = "0.0s";

/// `text`, a tool result, with every wall-clock measurement replaced by
/// [`CANONICAL_DURATION`]: the duration in a shell result's
/// `Exit: <code> | Duration: <seconds>s` status line is the only part of a
/// result that differs between two runs of one command. Everything else is
/// kept byte for byte.
#[must_use]
pub fn stable_tool_result(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for (i, line) in text.split('\n').enumerate() {
        if i > 0 {
            out.push('\n');
        }
        out.push_str(&stable_status_line(line));
    }
    out
}

/// One line of a tool result, its duration made canonical when it is a
/// shell status line.
fn stable_status_line(line: &str) -> std::borrow::Cow<'_, str> {
    let Some(at) = line.find(DURATION_FIELD) else {
        return line.into();
    };
    if !line.starts_with("Exit: ") {
        return line.into();
    }
    let value_start = at + DURATION_FIELD.len();
    let rest = &line[value_start..];
    let value_len = rest
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .unwrap_or(rest.len());
    if value_len == 0 || !rest[value_len..].starts_with('s') {
        return line.into();
    }
    format!(
        "{}{CANONICAL_DURATION}{}",
        &line[..value_start],
        &rest[value_len + 1..]
    )
    .into()
}

/// True when every tool call in `record` has arguments that are a JSON
/// object: a chat template renders a call's arguments as an object, so a
/// call a model wrote with broken arguments (which the runtime answered with
/// an error) cannot be shown to a trainer.
#[must_use]
pub fn renderable(record: &Value) -> bool {
    record["messages"]
        .as_array()
        .into_iter()
        .flatten()
        .all(|m| {
            m["tool_calls"].as_array().into_iter().flatten().all(|c| {
                c["function"]["arguments"]
                    .as_str()
                    .and_then(|a| serde_json::from_str::<Value>(a).ok())
                    .is_some_and(|a| a.is_object())
            })
        })
}

/// Every `(request, reply)` call of an episode as JSON lines; a call whose
/// conversation holds a tool call with arguments that are not a JSON object
/// is left out (see [`renderable`]).
///
/// # Errors
/// As [`record`], for any call.
pub fn episode_lines(calls: &[(Value, Value)], episode: &str) -> Result<String, ViewError> {
    let mut out = String::new();
    for (call, (request, reply)) in calls.iter().enumerate() {
        let record = record(request, reply, episode, call)?;
        if renderable(&record) {
            out.push_str(&record.to_string());
            out.push('\n');
        }
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
        let served = json!({ "role": "assistant", "content": "ok", "refusal": null });
        let kept = record(&request, &served, "e", 0).unwrap();
        assert!(
            kept["messages"]
                .as_array()
                .unwrap()
                .iter()
                .all(|m| m.get("refusal").is_none()),
            "a server's extra fields are dropped"
        );
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
        let toolless = record(
            &json!({ "messages": [{ "role": "user", "content": "u" }] }),
            &reply,
            "e",
            0,
        )
        .unwrap();
        assert_eq!(
            toolless["tools"],
            json!([]),
            "a request without tools records an empty list"
        );
    }

    #[test]
    fn two_runs_of_one_episode_write_the_same_bytes() {
        // The same episode as two runs saw it: the shell took another time
        // and the server named its tool calls differently.
        let run = |duration: &str, id: &str| {
            let call = json!({ "role": "assistant", "content": null, "tool_calls": [
                { "id": id, "type": "function", "function": { "name": "shell", "arguments": "{\"shell_command\":\"./calc combine 1 2\"}" } }] });
            let request = |more: Vec<Value>| {
                let mut messages = vec![json!({ "role": "user", "content": "combine" })];
                messages.extend(more);
                json!({ "messages": messages, "tools": [] })
            };
            let result = json!({ "role": "tool", "tool_call_id": id,
                "content": format!("3\n\nExit: 0 | Duration: {duration} | Buffer: buf_0001") });
            let done = json!({ "role": "assistant", "content": "3" });
            episode_lines(
                &[
                    (request(vec![]), call.clone()),
                    (request(vec![call, result]), done),
                ],
                "demonstrate/train-00000-combine",
            )
            .unwrap()
        };
        let a = run("0.0s", "call_9f2a_0");
        assert_eq!(
            a,
            run("1.7s", "call_c7dc_0"),
            "the same bytes, so the same digest"
        );
        assert!(
            a.contains("Exit: 0 | Duration: 0.0s | Buffer: buf_0001"),
            "the status line keeps its shape: {a}"
        );
        assert!(!a.contains("#"), "no attempt number: {a}");
        assert_eq!(
            stable_tool_result("Exit: 2 | Duration: 12.5s"),
            "Exit: 2 | Duration: 0.0s"
        );
        let broken = json!({ "role": "assistant", "content": null, "tool_calls": [
            { "id": "c", "type": "function", "function": { "name": "read_file", "arguments": "{\"path\" \"a\"}" } }] });
        let request = json!({ "messages": [{ "role": "user", "content": "u" }], "tools": [] });
        let after = json!({ "messages": [{ "role": "user", "content": "u" }, broken.clone(),
            { "role": "tool", "tool_call_id": "c", "content": "error: invalid arguments" }], "tools": [] });
        let done = json!({ "role": "assistant", "content": "ok" });
        let lines = episode_lines(&[(request, broken), (after, done)], "e").unwrap();
        assert_eq!(
            lines, "",
            "a call with broken arguments, and every record showing it, is left out"
        );
        for kept in [
            "L1:Duration: 4s",
            "Exit: 0 | Duration: soon",
            "wrote 4 bytes",
        ] {
            assert_eq!(stable_tool_result(kept), kept, "not a status line");
        }
    }
}
