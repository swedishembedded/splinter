// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training records taken from the exact
// requests an agent runtime sends, so a model is fine-tuned on what it will
// see when deployed, for its clients. If your team needs expertise in agent
// fine-tuning data, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Reading an agent's request log (`sven.request-log/1`): one JSON object
//! per line, `{"kind": "request", "seq", "model", "request": {"messages",
//! "tools", ...}}` when a request is sent (its whole conversation as the
//! model was given it) and `{"kind": "response", "seq", "response"}` with
//! what came back.

use std::collections::BTreeMap;

use serde_json::Value;
use splinter_core::harness::{is_media, message_parts, ModelCall, ModelResponse, ModelToolCall};

/// One request of the log.
#[derive(Clone, Debug, PartialEq)]
pub struct LoggedRequest {
    /// The model role it was sent to, when the log names one.
    pub role: Option<String>,
    /// Whether any message of the request carries media (an image).
    pub media: bool,
    /// The request and its response; `None` when the log holds no response
    /// to it, or one that cannot be read.
    pub call: Option<ModelCall>,
}

/// Every request of a log, in the order they were sent.
///
/// A response is paired to its request by `seq`, so a log that cannot be
/// paired unambiguously is refused rather than read as fewer or other calls:
/// a request or response with no `seq`, a `seq` used by two requests or by two
/// responses, and a response to no request. A request the log holds no
/// response to (the trial ended before the model answered), or one that
/// cannot be read, is a request with no call.
///
/// # Errors
/// A line that is not JSON, or a log that cannot be paired
/// (`line <n>: <reason>`).
pub fn read_request_log(bytes: &[u8]) -> Result<Vec<LoggedRequest>, String> {
    let mut requests: Vec<(u64, LoggedRequest)> = Vec::new();
    // The line each seq was first seen on, to name the repeat.
    let mut request_lines: BTreeMap<u64, usize> = BTreeMap::new();
    let mut responses: BTreeMap<u64, (usize, Value)> = BTreeMap::new();
    for (n, line) in bytes.split(|b| *b == b'\n').enumerate() {
        let n = n + 1;
        if line.is_empty() {
            continue;
        }
        let mut row: Value = serde_json::from_slice(line).map_err(|e| format!("line {n}: {e}"))?;
        let kind = row.get("kind").and_then(Value::as_str).map(str::to_string);
        let kind = match kind.as_deref() {
            Some(kind @ ("request" | "response")) => kind,
            _ => continue,
        };
        let seq = row
            .get("seq")
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("line {n}: a {kind} has no seq to pair it by"))?;
        if kind == "request" {
            if let Some(first) = request_lines.insert(seq, n) {
                return Err(format!(
                    "line {n}: request seq {seq} repeats the request on line {first}"
                ));
            }
            let role = row.get("model").and_then(Value::as_str).map(str::to_string);
            let mut request = row.get_mut("request").map(Value::take).unwrap_or_default();
            let messages = take_array(&mut request, "messages");
            let tools = take_array(&mut request, "tools");
            let media = messages.iter().any(|m| message_parts(m).any(is_media));
            let call = ModelCall {
                seq,
                messages,
                tools,
                response: ModelResponse {
                    text: String::new(),
                    reasoning: None,
                    tool_calls: Vec::new(),
                },
            };
            requests.push((
                seq,
                LoggedRequest {
                    role,
                    media,
                    call: Some(call),
                },
            ));
        } else if let Some(response) = row.get_mut("response") {
            if let Some((first, _)) = responses.insert(seq, (n, response.take())) {
                return Err(format!(
                    "line {n}: response seq {seq} repeats the response on line {first}"
                ));
            }
        }
    }
    if let Some((seq, (line, _))) = responses
        .iter()
        .find(|(seq, _)| !request_lines.contains_key(seq))
    {
        return Err(format!(
            "line {line}: response seq {seq} answers no request"
        ));
    }
    for (seq, logged) in &mut requests {
        let response = responses.get(seq).and_then(|(_, r)| response_of(r));
        logged.call = match (logged.call.take(), response) {
            (Some(call), Some(response)) => Some(ModelCall { response, ..call }),
            _ => None,
        };
    }
    Ok(requests.into_iter().map(|(_, logged)| logged).collect())
}

/// Whether the answer to `call` is one an agent records a step for: it said
/// something or called a tool.
#[must_use]
pub fn made_a_step(call: &ModelCall) -> bool {
    !call.response.text.trim().is_empty() || !call.response.tool_calls.is_empty()
}

/// The array under `key` of `request`, moved out of it; empty when absent.
fn take_array(request: &mut Value, key: &str) -> Vec<Value> {
    match request.get_mut(key).map(Value::take) {
        Some(Value::Array(items)) => items,
        _ => Vec::new(),
    }
}

/// A logged response as the model answered it: its text, its reasoning and
/// its tool calls. `None` for a response that is not an object or whose calls
/// do not name an id, a tool and arguments.
fn response_of(response: &Value) -> Option<ModelResponse> {
    let object = response.as_object()?;
    let tool_calls = object
        .get("tool_calls")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|call| {
            let field = |key: &str| call.get(key).and_then(Value::as_str).map(str::to_string);
            Some(ModelToolCall {
                id: field("id")?,
                name: field("name")?,
                arguments: field("arguments")?,
            })
        })
        .collect::<Option<Vec<_>>>()?;
    Some(ModelResponse {
        text: object
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        reasoning: object
            .get("reasoning")
            .and_then(Value::as_str)
            .map(str::to_string),
        tool_calls,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(seq: u64, text: &str) -> String {
        format!(
            r#"{{"kind":"request","seq":{seq},"model":"m","request":{{"messages":[{{"role":"user","content":"{text}"}}],"tools":[]}}}}"#
        )
    }

    fn response(seq: u64, text: &str) -> String {
        format!(r#"{{"kind":"response","seq":{seq},"response":{{"text":"{text}"}}}}"#)
    }

    fn log(lines: &[String]) -> Vec<u8> {
        lines.join("\n").into_bytes()
    }

    #[test]
    fn a_response_is_paired_to_its_request_by_seq_whatever_the_line_order() {
        let read = read_request_log(&log(&[
            request(1, "first"),
            request(2, "second"),
            response(2, "b"),
            response(1, "a"),
        ]))
        .unwrap();
        let answers: Vec<&str> = read
            .iter()
            .map(|r| r.call.as_ref().unwrap().response.text.as_str())
            .collect();
        assert_eq!(answers, ["a", "b"], "in the order the requests were sent");
    }

    #[test]
    fn a_request_nobody_answered_is_a_request_with_no_call() {
        let read =
            read_request_log(&log(&[request(1, "q"), request(2, "q"), response(1, "a")])).unwrap();
        assert_eq!(read.len(), 2);
        assert!(read[0].call.is_some() && read[1].call.is_none());
    }

    #[test]
    fn a_log_that_cannot_be_paired_is_refused_not_read_as_other_calls() {
        let no_seq = r#"{"kind":"request","request":{"messages":[]}}"#.to_string();
        for (lines, wanted) in [
            (vec![no_seq], "line 1: a request has no seq"),
            (
                vec![request(1, "q"), request(1, "again")],
                "line 2: request seq 1 repeats the request on line 1",
            ),
            (
                vec![request(1, "q"), response(1, "a"), response(1, "b")],
                "line 3: response seq 1 repeats the response on line 2",
            ),
            (
                vec![request(1, "q"), response(2, "a")],
                "line 2: response seq 2 answers no request",
            ),
        ] {
            let error = read_request_log(&log(&lines)).unwrap_err();
            assert!(error.contains(wanted), "{error}");
        }
    }
}
