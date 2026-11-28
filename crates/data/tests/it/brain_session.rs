// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training on whole agent sessions of the
// tokens a policy sampled, for its clients. If your team needs expertise in
// turning multi-call agent episodes into exact and cheap training samples,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: a two-call session a CPU brain served (Qwen3-0.6B, policy trace on,
//! the first answer sent back with its `sampled` record) projects to exact
//! calls whose server records name their traced ids, and merges into one
//! record. Brain's `check_dataset` accepted records made this way and its
//! `score` reproduced every call's behaviour log-probabilities under f32 KV
//! (splinter knowledge 252); this test pins the projection of what the
//! server answered.

#![allow(clippy::unwrap_used)]

use atif::{
    AgentProfile, ObservationEntry, StepMetrics, StepObservation, StepOrigin, ToolInvocation,
    TraceStep, Trajectory,
};
use serde_json::{json, Value};
use splinter_core::harness::{ModelCall, ModelResponse, ModelToolCall};
use splinter_data::calls::{project_calls, CallProjection, Sampler};
use splinter_data::session::SessionRecord;
use splinter_data::ProxyReason;

/// The request log's form of chat `messages`: a call is a message of its own.
fn to_log(messages: &[Value]) -> Vec<Value> {
    let mut out = Vec::new();
    for m in messages {
        if m["role"] == "assistant" {
            out.push(json!({"role": "assistant", "content": m["content"].as_str().unwrap_or("")}));
            for c in m["tool_calls"].as_array().into_iter().flatten() {
                out.push(json!({"role": "assistant", "content": {"tool_call_id": c["id"], "function": {"name": c["function"]["name"], "arguments": c["function"]["arguments"]}}}));
            }
        } else if m["role"] == "tool" {
            out.push(json!({"role": "tool", "content": {"tool_call_id": m["tool_call_id"], "content": m["content"]}}));
        } else {
            out.push(m.clone());
        }
    }
    out
}

/// The session's trajectory and request log, each step's `sampled` record
/// as `record` makes it of the one the server sent, projected.
fn projected(record: impl Fn(&Value) -> String) -> CallProjection {
    let session: Value =
        serde_json::from_slice(include_bytes!("../fixtures/brain-session.json")).unwrap();
    let tools = vec![
        json!({"name": "shell", "description": "run a shell command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}),
    ];
    let opening = session["requests"][0][1]["content"].as_str().unwrap();
    let mut steps = vec![TraceStep::new(1, StepOrigin::User, opening)];
    let mut calls = Vec::new();
    for (k, r) in session["responses"].as_array().unwrap().iter().enumerate() {
        let m = &r["choices"][0]["message"];
        let t = &r["policy_trace"];
        let ids = |key: &str| serde_json::from_value::<Vec<i64>>(t[key].clone()).unwrap();
        let text = m["content"].as_str().unwrap_or("");
        let mut step = TraceStep::new(k as u64 + 2, StepOrigin::Agent, text);
        let mut tool_calls = Vec::new();
        for c in m["tool_calls"].as_array().into_iter().flatten() {
            let id = c["id"].as_str().unwrap();
            let arguments = c["function"]["arguments"].as_str().unwrap();
            step.tool_calls = Some(vec![ToolInvocation::new(id, "shell")
                .with_arguments(serde_json::from_str(arguments).unwrap())]);
            step.observation = Some(StepObservation::single(ObservationEntry::for_call(
                id,
                "a.txt\nb.txt",
            )));
            tool_calls.push(ModelToolCall {
                id: id.into(),
                name: "shell".into(),
                arguments: arguments.into(),
            });
        }
        step.metrics = Some(StepMetrics {
            prompt_token_ids: Some(ids("prompt_token_ids")),
            completion_token_ids: Some(ids("generation_token_ids")),
            logprobs: Some(serde_json::from_value(t["generation_log_probs"].clone()).unwrap()),
            extra: Some(json!({"sven": {
                "logprob_kind": r["brain"]["logprob_kind"],
                "logprob_temperature": t["log_prob_temperature"],
                "tokenizer_digest": t["tokenizer_digest"],
                "policy_version": r["brain"]["policy_version"].to_string(),
            }})),
            ..StepMetrics::default()
        });
        // sven keeps the record beside the step's metrics.
        step.extra = Some(json!({"sven": {"sampled": record(&m["sampled"])}}));
        steps.push(step);
        let messages: Vec<Value> = serde_json::from_value(session["requests"][k].clone()).unwrap();
        calls.push(ModelCall {
            seq: k as u64 + 1,
            messages: to_log(&messages),
            tools: tools.clone(),
            response: ModelResponse {
                text: text.into(),
                reasoning: None,
                tool_calls,
            },
        });
    }
    let mut trajectory = Trajectory::new("ATIF-v1.7", AgentProfile::new("brain-session", "1"));
    trajectory.steps = steps;
    trajectory.final_metrics = Some(atif::FinalMetrics {
        extra: Some(json!({"sven": {"policy_version_span": [0, 0]}})),
        ..Default::default()
    });
    project_calls(&trajectory, &calls, &Sampler::untruncated(1.0).unwrap())
}

#[test]
fn a_served_session_projects_exactly_and_merges_into_one_record() {
    let projection = projected(Value::to_string);
    assert!(projection.exact(), "{:?}", projection.proxy);
    let [merged] = SessionRecord::segments(&projection.records, None)
        .unwrap()
        .try_into()
        .unwrap();
    let turns: Vec<_> = merged.sampled.turns.iter().flatten().collect();
    assert_eq!(turns.len(), 2, "both answers supervised in the one record");
    assert!(merged.tokens() < merged.per_call_tokens);

    // A record naming another prompt than the trace (its digest changed)
    // leaves the call untrained.
    let tampered = projected(|record| {
        let mut record = record.clone();
        record["context_hash"] = json!(format!("sha256:{}", "0".repeat(64)));
        record.to_string()
    });
    assert_eq!(tampered.proxy.get(&ProxyReason::SampledDiffers), Some(&2));
}
