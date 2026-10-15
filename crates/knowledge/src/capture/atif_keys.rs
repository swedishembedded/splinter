// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements lossless trajectory intake that refuses what
// it cannot represent instead of dropping it, for its clients. If your team
// needs expertise in training-data provenance, you can procure our services
// by sending an email to info@swedishembedded.com.

//! The structural keys of an ATIF v1.7 trajectory.
//!
//! The typed model ignores a key it does not know, so a field a newer
//! producer added would vanish between the file and the stored source with no
//! trace. [`unknown_key`] reads the raw JSON first and names the path of the
//! first such key. The producer's own room for metadata (`extra`), the tool
//! definitions and a call's arguments are free-form by the format and are
//! not walked.

use serde_json::Value;

const TRAJECTORY: &[&str] = &[
    "schema_version",
    "session_id",
    "trajectory_id",
    "agent",
    "notes",
    "final_metrics",
    "continued_trajectory_ref",
    "extra",
    "subagent_trajectories",
    "steps",
];
const AGENT: &[&str] = &["name", "version", "model_name", "tool_definitions", "extra"];
const STEP: &[&str] = &[
    "step_id",
    "timestamp",
    "source",
    "model_name",
    "reasoning_effort",
    "message",
    "reasoning_content",
    "tool_calls",
    "observation",
    "metrics",
    "extra",
    "llm_call_count",
    "is_copied_context",
];
const TOOL_CALL: &[&str] = &["tool_call_id", "function_name", "arguments", "extra"];
const OBSERVATION: &[&str] = &["results"];
const RESULT: &[&str] = &[
    "source_call_id",
    "content",
    "subagent_trajectory_ref",
    "extra",
];
const SUBAGENT_REF: &[&str] = &["trajectory_id", "trajectory_path", "session_id", "extra"];
const METRICS: &[&str] = &[
    "prompt_tokens",
    "completion_tokens",
    "cached_tokens",
    "cost_usd",
    "prompt_token_ids",
    "completion_token_ids",
    "logprobs",
    "extra",
];
const FINAL_METRICS: &[&str] = &[
    "total_prompt_tokens",
    "total_completion_tokens",
    "total_cached_tokens",
    "total_cost_usd",
    "total_steps",
    "extra",
];
const TEXT_SEGMENT: &[&str] = &["type", "text"];
const IMAGE_SEGMENT: &[&str] = &["type", "source"];
const IMAGE_SOURCE: &[&str] = &["media_type", "path"];

/// The JSON path of the first structural key of the trajectory `value` that
/// ATIF v1.7 does not define, or `None` when every key is known.
#[must_use]
pub fn unknown_key(value: &Value) -> Option<String> {
    trajectory(value, "")
}

/// The first key of `value`, an object, outside `known`, as a path under
/// `at`. Anything that is not an object is left to the typed parse.
fn stray(value: &Value, known: &[&str], at: &str) -> Option<String> {
    value
        .as_object()?
        .keys()
        .find(|key| !known.contains(&key.as_str()))
        .map(|key| join(at, key))
}

fn join(at: &str, key: &str) -> String {
    if at.is_empty() {
        key.to_string()
    } else {
        format!("{at}.{key}")
    }
}

fn each<'a>(value: &'a Value, key: &str, at: &str) -> impl Iterator<Item = (String, &'a Value)> {
    let base = join(at, key);
    value
        .get(key)
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
        .map(move |(i, item)| (format!("{base}[{i}]"), item))
}

fn child<'a>(value: &'a Value, key: &str, at: &str) -> Option<(String, &'a Value)> {
    value.get(key).map(|v| (join(at, key), v))
}

fn trajectory(value: &Value, at: &str) -> Option<String> {
    stray(value, TRAJECTORY, at)
        .or_else(|| child(value, "agent", at).and_then(|(p, v)| stray(v, AGENT, &p)))
        .or_else(|| {
            child(value, "final_metrics", at).and_then(|(p, v)| stray(v, FINAL_METRICS, &p))
        })
        .or_else(|| each(value, "steps", at).find_map(|(p, v)| step(v, &p)))
        .or_else(|| each(value, "subagent_trajectories", at).find_map(|(p, v)| trajectory(v, &p)))
}

fn step(value: &Value, at: &str) -> Option<String> {
    stray(value, STEP, at)
        .or_else(|| body(value.get("message")?, &join(at, "message")))
        .or_else(|| each(value, "tool_calls", at).find_map(|(p, v)| stray(v, TOOL_CALL, &p)))
        .or_else(|| child(value, "metrics", at).and_then(|(p, v)| stray(v, METRICS, &p)))
        .or_else(|| child(value, "observation", at).and_then(|(p, v)| observation(v, &p)))
}

fn observation(value: &Value, at: &str) -> Option<String> {
    stray(value, OBSERVATION, at).or_else(|| {
        each(value, "results", at).find_map(|(p, result)| {
            stray(result, RESULT, &p)
                .or_else(|| body(result.get("content")?, &join(&p, "content")))
                .or_else(|| {
                    each(result, "subagent_trajectory_ref", &p)
                        .find_map(|(r, v)| stray(v, SUBAGENT_REF, &r))
                })
        })
    })
}

/// A message or result content: text, or segments of text and images.
fn body(value: &Value, at: &str) -> Option<String> {
    let segments = value.as_array()?;
    segments.iter().enumerate().find_map(|(i, segment)| {
        let at = format!("{at}[{i}]");
        match segment.get("type").and_then(Value::as_str) {
            Some("image") => stray(segment, IMAGE_SEGMENT, &at).or_else(|| {
                child(segment, "source", &at).and_then(|(p, v)| stray(v, IMAGE_SOURCE, &p))
            }),
            _ => stray(segment, TEXT_SEGMENT, &at),
        }
    })
}
