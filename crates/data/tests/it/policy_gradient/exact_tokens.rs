// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reinforcement learning from the verified
// rewards of agent rollouts for its clients. If your team needs expertise in
// training agents against verifiers that do not trust them, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec: a rewarded action whose agent step recorded what the policy sampled
//! (the ids it was given, the ids it generated, their log-probabilities under
//! the policy's own distribution) is written with exactly those, so nothing is
//! re-rendered and the trainer weighs the tokens that were sampled. An action
//! without a usable trace is still written, without per-token fields, and the
//! projection counts why. Ids decode to the unscrubbed text, so a conversation
//! the scrub would change carries none.

use super::*;
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicUsize, Ordering};

use atif::StepMetrics;
use splinter_core::harness::{ModelCall, ModelResponse, ModelToolCall};
use splinter_data::ProxyReason;

const DIGEST: &str = "sha256:aa";

/// What sven records on an agent step: ids given, ids generated, and one
/// log-probability per generated id, of the kind `kind`.
fn trace(prompt: &[i64], completion: &[i64], logprobs: &[f64], kind: &str) -> StepMetrics {
    StepMetrics {
        prompt_token_ids: Some(prompt.to_vec()),
        completion_token_ids: Some(completion.to_vec()),
        logprobs: Some(logprobs.to_vec()),
        extra: Some(json!({"sven": {"logprob_kind": kind, "tokenizer_digest": DIGEST}})),
        ..StepMetrics::default()
    }
}

/// Two trials (a pass and a fail, so the group has a spread) whose agent
/// steps carry the traces `call_step` and `answer` make.
fn traced(call_step: StepMetrics, answer: StepMetrics, tool_result: Option<&str>) -> Corpus {
    let mut corpus = Corpus::new();
    for trial in [Trial::pass("t1"), Trial::fail("t2", &[])] {
        let (call_step, answer) = (call_step.clone(), answer.clone());
        let calls = requests(tool_result.unwrap_or("part.py"));
        add_with(&mut corpus, &trial, None, false, Some(&calls), |steps| {
            steps[2].metrics = Some(call_step);
            steps[3].metrics = Some(answer);
            if let Some(result) = tool_result {
                let entry = &mut steps[2].observation.as_mut().unwrap().results[0];
                entry.content = Some(result.into());
            }
        });
    }
    corpus
}

/// The two requests the harness recorded the model was sent: the opening
/// (system, user) answered by the call, and the one after its result
/// answered by the final text.
fn requests(result: &str) -> Vec<ModelCall> {
    let opening = vec![
        json!({"role": "system", "content": "You are a careful engineer."}),
        json!({"role": "user", "content": "Design a bracket."}),
    ];
    let mut later = opening.clone();
    later.extend([
        json!({"role": "assistant", "content": "", "reasoning": "look"}),
        json!({"role": "assistant", "content": {"tool_call_id": "c0", "function": {
            "name": "shell", "arguments": "{\"command\":\"ls\"}"}}}),
        json!({"role": "tool", "content": {"tool_call_id": "c0", "content": result}}),
    ]);
    let shell =
        json!({"name": "shell", "description": "Run a command.", "parameters": {"type": "object"}});
    let response = |text: &str, calls: Vec<ModelToolCall>| ModelResponse {
        text: text.into(),
        reasoning: (!calls.is_empty()).then(|| "look".into()),
        tool_calls: calls,
    };
    vec![
        ModelCall {
            seq: 1,
            messages: opening,
            tools: vec![shell.clone()],
            response: response(
                "",
                vec![ModelToolCall {
                    id: "c0".into(),
                    name: "shell".into(),
                    arguments: "{\"command\": \"ls\"}".into(),
                }],
            ),
        },
        ModelCall {
            seq: 2,
            messages: later,
            tools: vec![shell],
            response: response("Done.", Vec::new()),
        },
    ]
}

fn good() -> (StepMetrics, StepMetrics) {
    (
        trace(&[1, 2, 3], &[10, 11], &[-0.5, -0.25], "policy"),
        trace(
            &[1, 2, 3, 10, 11, 4],
            &[12, 13, 14],
            &[-1.0, -2.0, -0.125],
            "policy",
        ),
    )
}

fn lines(projection: &Projection) -> Vec<Value> {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "pg-exact-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let path = dir.join("d.jsonl");
    write_dataset(&path, projection, &Unchecked).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    text.lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

fn supervised(line: &Value) -> Vec<&Value> {
    line["messages"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|m| m["train"] == true)
        .collect()
}

fn assert_exact(line: &Value, supervised_ids: &[(&[i64], &[i64], &[f64])]) {
    assert_eq!(line["tokenizer_digest"], DIGEST);
    let got = supervised(line);
    assert_eq!(got.len(), supervised_ids.len());
    for (message, (prompt, completion, logprobs)) in got.iter().zip(supervised_ids) {
        assert_eq!(message["prompt_token_ids"], json!(prompt));
        assert_eq!(message["token_ids"], json!(completion));
        assert_eq!(message["behavior_logprobs"], json!(logprobs));
    }
    // Context is never given token fields.
    for message in line["messages"].as_array().unwrap() {
        if message["train"] == false {
            assert!(message.get("token_ids").is_none(), "{message}");
        }
    }
}

#[test]
fn a_policy_trace_is_written_one_to_one_with_the_action_it_belongs_to() {
    let (call_step, answer) = good();
    let corpus = traced(call_step, answer, None);

    let per_action = view().project(&corpus).unwrap();
    let written = lines(&per_action);
    assert_eq!(written.len(), 4);
    for pair in written.chunks(2) {
        assert_exact(&pair[0], &[(&[1, 2, 3], &[10, 11], &[-0.5, -0.25])]);
        assert_exact(
            &pair[1],
            &[(&[1, 2, 3, 10, 11, 4], &[12, 13, 14], &[-1.0, -2.0, -0.125])],
        );
    }
    assert_eq!(per_action.tokens.exact, 4);
    assert!(per_action.tokens.proxy.is_empty());

    let packed = view().packed(true).project(&corpus).unwrap();
    let written = lines(&packed);
    assert_eq!(written.len(), 2);
    for line in &written {
        assert_exact(
            line,
            &[
                (&[1, 2, 3], &[10, 11], &[-0.5, -0.25]),
                (&[1, 2, 3, 10, 11, 4], &[12, 13, 14], &[-1.0, -2.0, -0.125]),
            ],
        );
    }
    assert_eq!(packed.tokens.exact, 4);

    // Brain's contract (its fixtures, copied unchanged): every field written
    // on a record or a message is one its valid rollout records carry.
    let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/training-records/v2/chat_rollout.jsonl");
    let (mut record_keys, mut message_keys) = (BTreeSet::new(), BTreeSet::new());
    for line in std::fs::read_to_string(fixtures).unwrap().lines() {
        let line: Value = serde_json::from_str(line).unwrap();
        record_keys.extend(line.as_object().unwrap().keys().cloned());
        for message in line["messages"].as_array().unwrap() {
            message_keys.extend(message.as_object().unwrap().keys().cloned());
        }
    }
    record_keys.extend(["tools", "metadata"].map(String::from));
    message_keys.extend(["tool_calls", "tool_call_id", "reasoning_content"].map(String::from));
    for line in lines(&per_action).iter().chain(&written) {
        let keys: BTreeSet<String> = line.as_object().unwrap().keys().cloned().collect();
        assert!(
            keys.is_subset(&record_keys),
            "{keys:?} not in {record_keys:?}"
        );
        for message in line["messages"].as_array().unwrap() {
            let keys: BTreeSet<String> = message.as_object().unwrap().keys().cloned().collect();
            assert!(
                keys.is_subset(&message_keys),
                "{keys:?} not in {message_keys:?}"
            );
        }
    }
}

#[test]
fn a_trace_that_is_not_the_policys_own_or_does_not_add_up_is_a_proxy_and_says_why() {
    let (_, answer) = good();
    // The call step's figures are the sampler's (after top-k/top-p); the
    // answer's carry a log-probability fewer than its ids.
    let sampler = trace(&[1, 2, 3], &[10, 11], &[-0.5, -0.25], "sampler");
    let short = trace(&[1], &[12, 13, 14], &[-1.0, -2.0], "policy");
    let sampled = traced(sampler, answer.clone(), None);
    let counted = traced(good().0, short, None);
    let untraced = traced(StepMetrics::default(), answer, None);

    for (corpus, reason, exact) in [
        (sampled, ProxyReason::SamplerLogprobs, 2),
        (counted, ProxyReason::CountMismatch, 2),
        (untraced, ProxyReason::NoTrace, 2),
    ] {
        let projection = view().project(&corpus).unwrap();
        let written = lines(&projection);
        // Every action is still a record; the one without a usable trace
        // carries no per-token field at all.
        assert_eq!(written.len(), 4, "{reason:?}");
        let bare = written
            .iter()
            .filter(|l| supervised(l).iter().all(|m| m.get("token_ids").is_none()))
            .count();
        assert_eq!(bare, 2, "{reason:?}");
        assert_eq!(projection.tokens.exact, exact, "{reason:?}");
        assert_eq!(projection.tokens.proxy.get(&reason), Some(&2), "{reason:?}");
        for line in &written {
            for message in supervised(line) {
                let fields = ["token_ids", "behavior_logprobs", "prompt_token_ids"]
                    .iter()
                    .filter(|k| message.get(**k).is_some())
                    .count();
                assert!(fields == 0 || fields == 3, "partial fields: {message}");
            }
        }
    }
}

#[test]
fn a_trace_with_no_prompt_or_an_impossible_logprob_is_never_labelled_the_policys_sample() {
    let (call_step, answer) = good();
    // A log-probability is a finite number at most zero; the context the
    // answer was sampled after is at least one id.
    for (bad, reason) in [
        (
            trace(&[1, 2, 3], &[10, 11], &[-0.5, f64::NAN], "policy"),
            ProxyReason::InvalidLogprobs,
        ),
        (
            trace(&[1, 2, 3], &[10, 11], &[-0.5, f64::NEG_INFINITY], "policy"),
            ProxyReason::InvalidLogprobs,
        ),
        (
            trace(&[1, 2, 3], &[10, 11], &[-0.5, 0.25], "policy"),
            ProxyReason::InvalidLogprobs,
        ),
        (
            trace(&[], &[10, 11], &[-0.5, -0.25], "policy"),
            ProxyReason::NoTrace,
        ),
    ] {
        let projection = view().project(&traced(bad, answer.clone(), None)).unwrap();
        assert_eq!(projection.tokens.proxy.get(&reason), Some(&2), "{reason:?}");
        assert_eq!(projection.tokens.exact, 2, "{reason:?}");
    }
    let projection = view().project(&traced(call_step, answer, None)).unwrap();
    assert_eq!(projection.tokens.exact, 4);
}

#[test]
fn a_conversation_the_scrub_would_change_carries_no_ids() {
    let (call_step, answer) = good();
    // A listing of a home directory is anonymised in what a student is shown,
    // so the ids (which decode to the original) would teach the name.
    let corpus = traced(call_step, answer, Some("\x2fhome/marsch/part.py"));
    let projection = view().project(&corpus).unwrap();
    for line in lines(&projection) {
        assert!(line.get("tokenizer_digest").is_none());
        for message in supervised(&line) {
            assert!(message.get("token_ids").is_none(), "{message}");
        }
    }
    assert_eq!(projection.tokens.exact, 0);
    assert_eq!(
        projection.tokens.proxy.get(&ProxyReason::Scrubbed),
        Some(&4)
    );
}

#[test]
fn a_conversation_rebuilt_from_its_trajectory_carries_no_ids() {
    // Without the requests the model was sent, the context is the task's
    // instruction under the strip policy, not the prompt the ids follow.
    let (call_step, answer) = good();
    let mut corpus = Corpus::new();
    for trial in [Trial::pass("t1"), Trial::fail("t2", &[])] {
        let (call_step, answer) = (call_step.clone(), answer.clone());
        add_with(&mut corpus, &trial, None, false, None, |steps| {
            steps[2].metrics = Some(call_step);
            steps[3].metrics = Some(answer);
        });
    }
    let projection = view().project(&corpus).unwrap();
    for line in lines(&projection) {
        for message in supervised(&line) {
            assert!(message.get("token_ids").is_none(), "{message}");
        }
    }
    assert_eq!(projection.tokens.exact, 0);
    assert_eq!(
        projection
            .tokens
            .proxy
            .get(&ProxyReason::ContextReconstructed),
        Some(&4)
    );
}

#[test]
fn a_context_read_from_requests_keeps_the_text_the_model_was_sent() {
    // As sven sends an answer back: the whitespace the model wrote before its
    // call, and the call's arguments as it wrote them. A chat template renders
    // both as given (Qwen3's writes a newline before a call only after text),
    // so the record's context holds them as sent, or its prompt ids would
    // name another context.
    let (call_step, answer) = good();
    let mut corpus = Corpus::new();
    for trial in [Trial::pass("t1"), Trial::fail("t2", &[])] {
        let mut calls = requests("part.py");
        let written = "{\"command\":  \"ls\"}";
        calls[0].response.text = "\n\n".into();
        calls[0].response.tool_calls[0].arguments = written.into();
        calls[1].messages[2]["content"] = json!("\n\n");
        calls[1].messages[3]["content"]["function"]["arguments"] = json!(written);
        let (call_step, answer) = (call_step.clone(), answer.clone());
        add_with(&mut corpus, &trial, None, false, Some(&calls), |steps| {
            steps[2].metrics = Some(call_step);
            steps[3].metrics = Some(answer);
        });
    }
    let projection = view().project(&corpus).unwrap();
    assert_eq!(projection.tokens.exact, 4, "{:?}", projection.tokens);
    let written = lines(&projection);
    let history = written
        .iter()
        .flat_map(|l| l["messages"].as_array().unwrap())
        .find(|m| m["role"] == "assistant" && m["train"] == false)
        .expect("the second action's context holds the first answer");
    assert_eq!(history["content"], "\n\n");
    assert_eq!(
        history["tool_calls"][0]["function"]["arguments"],
        "{\"command\":  \"ls\"}"
    );
}

#[test]
fn a_build_that_trains_only_sampled_tokens_leaves_every_proxy_action_out_and_says_so() {
    // A proxy action's text is re-rendered and weighed against the start of
    // the run, not against what the policy sampled: tokens it may never have
    // drawn, whose importance ratio is unbounded (knowledge 188). Asked to
    // train only sampled tokens, the view writes no such record, counts each,
    // and keeps every trial in its group's baseline.
    let (_, answer) = good();
    let sampler = trace(&[1, 2, 3], &[10, 11], &[-0.5, -0.25], "sampler");
    let mixed = traced(sampler, answer.clone(), None);
    let projection = view().sampled_only(true).project(&mixed).unwrap();
    let written = lines(&projection);
    assert_eq!(written.len(), 2, "only the two traced answers are trained");
    for line in &written {
        assert!(!supervised(line).is_empty());
        for message in supervised(line) {
            assert!(message.get("token_ids").is_some(), "{message}");
        }
    }
    assert_eq!(projection.excluded.get(&Exclusion::ProxyTokens), Some(&2));

    // A conversation whose every action is a proxy trains nothing, yet both
    // trials still set the group's baseline.
    let (call_step, answer) = good();
    let scrubbed = traced(call_step, answer, Some("\x2fhome/marsch/part.py"));
    let projection = view().sampled_only(true).project(&scrubbed).unwrap();
    assert_eq!(projection.excluded.get(&Exclusion::ProxyTokens), Some(&4));
    let outcomes = outcomes(&projection);
    assert_eq!(outcomes.len(), 2);
    assert!(outcomes.iter().all(|o| o.baseline_only));

    // Left unset, proxies are written as before.
    let (call_step, answer) = good();
    let scrubbed = traced(call_step, answer, Some("\x2fhome/marsch/part.py"));
    assert_eq!(lines(&view().project(&scrubbed).unwrap()).len(), 4);
}
