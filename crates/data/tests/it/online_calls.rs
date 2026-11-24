// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements online reinforcement learning from the exact
// tokens an agent's policy sampled, for its clients. If your team needs
// expertise in training agents on their own rollouts while they run, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: a rollout is trained one record per model call. Each record is the
//! request as sent with its answer the one supervised turn, carrying exactly
//! what the runtime recorded of that request: its prompt and generated ids,
//! the policy's log-probabilities, the sampler's top-K when asked, its policy
//! version and the sampler. A step that holds several requests is read from
//! its per-request list, never from its run-together fields. A rollout is
//! exact only when every call is; a missing trace, the sampler's own
//! log-probabilities, another temperature, a malformed top-K or calls that
//! do not pair make it inexact, counted by reason.

#![allow(clippy::unwrap_used)]

use crate::common;

use atif::{StepMetrics, TraceStep};
use common::*;
use serde_json::{json, Value};
use splinter_core::harness::{ModelCall, ModelResponse, ModelToolCall};
use splinter_data::calls::{project_calls, Sampler};
use splinter_data::{dataset_line, Objective, ProxyReason};

const DIGEST: &str = "sha256:c7ae3118aa233013ee286be9c64eed92aeb1d113ebd470a1a3e646b3d78192a3";

fn sampler() -> Sampler {
    Sampler::untruncated(0.8).unwrap()
}

/// What sven records of one request: its ids, the policy's figures and its
/// identity.
fn request(prompt: &[i64], generated: &[i64], version: &str) -> Value {
    json!({
        "prompt_token_ids": prompt,
        "completion_token_ids": generated,
        "logprobs": generated.iter().map(|_| -0.5).collect::<Vec<_>>(),
        "logprob_kind": "policy",
        "logprob_temperature": 0.8,
        "tokenizer_digest": DIGEST,
        "policy_version": version,
    })
}

/// The metrics of a step of one request: the request's ids in the step's
/// own fields, its identity in `extra.sven`.
fn single(request: &Value) -> StepMetrics {
    fn ids<T: serde::de::DeserializeOwned>(request: &Value, key: &str) -> Option<T> {
        request
            .get(key)
            .map(|v| serde_json::from_value(v.clone()).unwrap())
    }
    let mut facts = request.clone();
    for key in ["prompt_token_ids", "completion_token_ids", "logprobs"] {
        facts.as_object_mut().unwrap().remove(key);
    }
    StepMetrics {
        prompt_token_ids: ids(request, "prompt_token_ids"),
        completion_token_ids: ids(request, "completion_token_ids"),
        logprobs: ids(request, "logprobs"),
        extra: Some(json!({ "sven": facts })),
        ..StepMetrics::default()
    }
}

/// The metrics of a step of several requests: the step's fields run them
/// together (the first prompt, every generated id), the list is each one's.
fn several(requests: &[Value]) -> StepMetrics {
    let all = |key: &str| -> Vec<Value> {
        requests
            .iter()
            .flat_map(|r| r[key].as_array().unwrap().clone())
            .collect()
    };
    StepMetrics {
        prompt_token_ids: serde_json::from_value(requests[0]["prompt_token_ids"].clone()).unwrap(),
        completion_token_ids: serde_json::from_value(json!(all("completion_token_ids"))).unwrap(),
        logprobs: serde_json::from_value(json!(all("logprobs"))).unwrap(),
        extra: Some(json!({ "sven": { "requests": requests, "policy_version": "3" } })),
        ..StepMetrics::default()
    }
}

fn step(n: u64, text: &str, metrics: StepMetrics) -> TraceStep {
    let mut s = say(n, text);
    s.metrics = Some(metrics);
    s
}

fn shell() -> Value {
    json!({"name": "shell", "description": "Run a command.", "parameters": {"type": "object"}})
}

/// The request log's two calls: the opening, which calls `shell`, and the
/// request after its result, answered with text.
fn calls() -> Vec<ModelCall> {
    let opening = vec![
        json!({"role": "system", "content": "Be brief."}),
        json!({"role": "user", "content": "List the files."}),
    ];
    let mut later = opening.clone();
    later.extend([
        json!({"role": "assistant", "content": ""}),
        json!({"role": "assistant", "content": {"tool_call_id": "c0", "function": {
            "name": "shell", "arguments": "{\"command\":\"ls\"}"}}}),
        json!({"role": "tool", "content": {"tool_call_id": "c0", "content": "a.txt"}}),
    ]);
    vec![
        ModelCall {
            seq: 1,
            messages: opening,
            tools: vec![shell()],
            response: ModelResponse {
                text: String::new(),
                reasoning: None,
                tool_calls: vec![ModelToolCall {
                    id: "c0".into(),
                    name: "shell".into(),
                    arguments: "{\"command\":\"ls\"}".into(),
                }],
            },
        },
        ModelCall {
            seq: 2,
            messages: later,
            tools: vec![shell()],
            response: ModelResponse {
                text: "One file: a.txt.".into(),
                reasoning: None,
                tool_calls: Vec::new(),
            },
        },
    ]
}

fn with_span(mut trajectory: atif::Trajectory, span: [u64; 2]) -> atif::Trajectory {
    let totals = atif::FinalMetrics {
        extra: Some(json!({ "sven": { "policy_version_span": span } })),
        ..Default::default()
    };
    trajectory.final_metrics = Some(totals);
    trajectory
}

#[test]
fn each_call_is_a_record_of_its_request_and_answer_with_exactly_its_tokens() {
    let first = request(&[1, 2], &[7, 8], "4");
    let second = request(&[1, 2, 7, 8, 9], &[5], "4");
    let trajectory = with_span(
        trajectory(vec![
            user(1, "List the files."),
            step(2, "", single(&first)),
            step(3, "One file: a.txt.", single(&second)),
        ]),
        [4, 4],
    );
    let projection = project_calls(&trajectory, &calls(), &sampler());
    assert!(projection.exact(), "{projection:#?}");
    assert_eq!(
        (projection.span, projection.tokens, projection.truncated),
        (Some((4, 4)), 3, false)
    );
    let [one, two] = &projection.records[..] else {
        panic!("two records")
    };
    let roles = |r: &splinter_data::calls::CallRecord| -> Vec<(String, bool)> {
        r.messages
            .iter()
            .map(|m| (m.role.clone(), m.train))
            .collect()
    };
    let s = |r: &str, t| (r.to_string(), t);
    assert_eq!(
        roles(one),
        [s("system", false), s("user", false), s("assistant", true)]
    );
    assert_eq!(
        roles(two),
        [
            s("system", false),
            s("user", false),
            s("assistant", false),
            s("tool", false),
            s("assistant", true)
        ],
        "the second request's whole context, its answer the only supervised turn"
    );
    assert_eq!(two.messages[4].content, "One file: a.txt.");
    let turn = two.sampled.turns[0].as_ref().unwrap();
    assert_eq!(
        (&turn.prompt_token_ids[..], &turn.token_ids[..]),
        (&[1, 2, 7, 8, 9][..], &[5][..])
    );
    assert_eq!(two.policy_version, 4);

    // The learner's adapter adds the scored group: brain's v2 rollout line.
    let line: Value = serde_json::from_str(
        &dataset_line(
            Objective::PolicyGradient,
            &one.record(1.0, "online:7", "online:7/0"),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(line["reward"], 1.0);
    assert_eq!(line["rollout"], "online:7/0");
    assert_eq!(line["group"], "online:7");
    assert_eq!(line["policy_version"], 4);
    assert_eq!(
        line["sampler"],
        json!({"temperature": 0.8}),
        "untruncated: no cut declared"
    );
    assert_eq!(line["tokenizer_digest"], DIGEST);
    assert_eq!(line["tools"][0]["function"]["name"], "shell");
    let answer = &line["messages"][2];
    assert_eq!(
        (answer["train"].clone(), answer["token_ids"].clone()),
        (json!(true), json!([7, 8]))
    );
    assert_eq!(answer["behavior_logprobs"], json!([-0.5, -0.5]));
    assert_eq!(answer["prompt_token_ids"], json!([1, 2]));
    assert!(answer.get("behavior_top_ids").is_none());
    assert!(line.get("truncated").is_none());
}

#[test]
fn a_step_of_several_requests_is_read_per_request_and_a_refit_spans_both_versions() {
    let mut first = request(&[1, 2], &[7, 8], "3");
    first["generation_top_ids"] = json!([[7, 3], [8, 4]]);
    first["generation_top_log_probs"] = json!([[-0.5, -1.5], [-0.5, -2.0]]);
    let mut second = request(&[1, 2, 7, 8, 9], &[5], "4");
    second["generation_top_ids"] = json!([[5, 6]]);
    second["generation_top_log_probs"] = json!([[-0.5, -1.0]]);
    let trajectory = with_span(
        trajectory(vec![
            user(1, "List the files."),
            step(2, "One file: a.txt.", several(&[first, second])),
        ]),
        [3, 4],
    );
    let projection = project_calls(&trajectory, &calls(), &sampler());
    assert!(projection.exact(), "{projection:#?}");
    assert_eq!(projection.span, Some((3, 4)));
    let versions: Vec<u64> = projection
        .records
        .iter()
        .map(|r| r.policy_version)
        .collect();
    assert_eq!(
        versions,
        [3, 4],
        "each call names the version that decoded it"
    );
    let turn = projection.records[1].sampled.turns[0].as_ref().unwrap();
    assert_eq!(
        turn.token_ids,
        [5],
        "the request's own ids, not the step's run together"
    );
    assert_eq!(turn.behavior_top_ids.as_deref(), Some(&[vec![5, 6]][..]));
    let line: Value = serde_json::from_str(
        &dataset_line(
            Objective::PolicyGradient,
            &projection.records[0].record(0.0, "g", "g/1"),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        line["messages"][2]["behavior_top_ids"],
        json!([[7, 3], [8, 4]])
    );
    assert_eq!(
        line["messages"][2]["behavior_top_logprobs"],
        json!([[-0.5, -1.5], [-0.5, -2.0]])
    );
}

#[test]
fn a_call_without_its_exact_trace_makes_the_rollout_inexact_counted_by_reason() {
    let exact = || request(&[1, 2], &[7, 8], "4");
    let later = || request(&[1, 2, 7, 8, 9], &[5], "4");
    let case = |first: Value, second: Value| {
        let t = trajectory(vec![
            user(1, "List the files."),
            step(2, "", single(&first)),
            step(3, "One file: a.txt.", single(&second)),
        ]);
        project_calls(&t, &calls(), &sampler())
    };
    let without = |mut r: Value, key: &str| {
        r.as_object_mut().unwrap().remove(key);
        r
    };
    let set = |mut r: Value, key: &str, value: Value| {
        r[key] = value;
        r
    };
    for (second, reason) in [
        (without(later(), "logprobs"), ProxyReason::NoTrace),
        (
            set(later(), "logprob_kind", json!("sampler")),
            ProxyReason::SamplerLogprobs,
        ),
        (
            set(later(), "logprob_temperature", json!(1.0)),
            ProxyReason::TemperatureMismatch,
        ),
        (
            without(later(), "policy_version"),
            ProxyReason::NoPolicyVersion,
        ),
        (
            set(later(), "generation_top_ids", json!([[5, 5]])),
            ProxyReason::TopMismatch,
        ),
        (
            set(later(), "tokenizer_digest", json!("sha256:00")),
            ProxyReason::MixedTokenizer,
        ),
    ] {
        let projection = case(exact(), second);
        assert!(!projection.exact(), "{reason:?}");
        assert_eq!(
            projection.proxy.get(&reason),
            Some(&1),
            "{reason:?}: {projection:#?}"
        );
        assert_eq!(
            projection.records.len(),
            1,
            "the exact call alone is projected"
        );
    }

    // Calls and recorded requests that are not one to one pair nothing.
    let t = trajectory(vec![
        user(1, "List the files."),
        step(2, "", single(&exact())),
    ]);
    let projection = project_calls(&t, &calls(), &sampler());
    assert!(!projection.exact());
    assert!(projection.records.is_empty());
    assert_eq!(projection.proxy.get(&ProxyReason::Unpaired), Some(&2));

    // A request cut at its length limit marks the whole rollout truncated.
    let projection = case(exact(), set(later(), "truncated", json!(true)));
    assert!(projection.truncated && projection.records.iter().all(|r| r.truncated));
}
