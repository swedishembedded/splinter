// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements audits and gates that read what a tool-using
// agent really did, for its clients. If your team needs expertise in judging
// agent trajectories, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The tool calls of a trajectory, each with the output it returned.
//!
//! A step's calls run in parallel and finish in any order, so a result is
//! paired with its call by the call id it names. This is the one place that
//! pairing is decided: the exposure audit and every gate that reads what an
//! agent did go through it.

use std::collections::HashSet;

use atif::{ContentSegment, MessageBody, ObservationEntry, StepOrigin, TraceStep, Trajectory};
use serde_json::Value;

/// One tool call of an agent step and what it returned.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolCall {
    /// The trajectory step that made it.
    pub step: u64,
    /// The tool's name.
    pub tool: String,
    /// The arguments the model sent.
    pub arguments: Value,
    /// What the call returned, as text; empty when no result was recorded.
    pub output: String,
}

/// The tool calls of the agent steps of `trajectory`, in order.
#[must_use]
pub fn tool_calls(trajectory: &Trajectory) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    for step in trajectory
        .steps
        .iter()
        .filter(|step| step.source == StepOrigin::Agent)
    {
        let outputs = outputs_by_call(step);
        for (call, output) in step.tool_calls.iter().flatten().zip(outputs) {
            calls.push(ToolCall {
                step: step.step_id,
                tool: call.function_name.clone(),
                arguments: call.arguments.clone(),
                output,
            });
        }
    }
    calls
}

/// The text of a message body: its text segments joined, images left out.
#[must_use]
pub fn body_text(body: &MessageBody) -> String {
    match body {
        MessageBody::Text(text) => text.clone(),
        MessageBody::Segments(segments) => segments
            .iter()
            .filter_map(|segment| match segment {
                ContentSegment::Text { text } => Some(text.as_str()),
                ContentSegment::Image { .. } => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

/// The text of one observation result.
fn result_text(result: &ObservationEntry) -> String {
    result.content.as_ref().map(body_text).unwrap_or_default()
}

/// The output each call of `step` returned. Results arrive in the order the
/// calls finished, so they are paired by the call id each names (a call
/// whose result is missing returned nothing). A step whose results do not
/// name its calls (an id is shared or unknown) is paired by position when
/// the counts agree; else the whole step's output stands for every call,
/// which is the most that can be said of it.
#[must_use]
pub fn outputs_by_call(step: &TraceStep) -> Vec<String> {
    let calls = step.tool_calls.as_deref().unwrap_or_default();
    let results: Vec<&ObservationEntry> =
        step.observation.iter().flat_map(|o| &o.results).collect();
    let ids: HashSet<&str> = calls.iter().map(|c| c.tool_call_id.as_str()).collect();
    let named_by_id = ids.len() == calls.len()
        && results.iter().all(|r| {
            r.source_call_id
                .as_deref()
                .is_some_and(|id| ids.contains(id))
        });
    if named_by_id {
        return calls
            .iter()
            .map(|call| {
                results
                    .iter()
                    .filter(|r| r.source_call_id.as_deref() == Some(call.tool_call_id.as_str()))
                    .map(|r| result_text(r))
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .collect();
    }
    let texts: Vec<String> = results.into_iter().map(result_text).collect();
    if texts.len() == calls.len() {
        texts
    } else {
        vec![texts.join("\n"); calls.len()]
    }
}
