// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training on whole agent sessions of the
// tokens a policy sampled, for its clients. If your team needs expertise in
// turning multi-call agent episodes into exact and cheap training samples,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: an agent session whose every call is exact and whose requests are
//! append-only (each call's prompt ids are the previous call's prompt ids,
//! its generated ids and the turns between) is one training record: the
//! last request and its answer, every answer supervised in place with
//! exactly its call's ids and log-probabilities, the oldest policy version,
//! the sampler and tokenizer once. Any session that is not is refused with
//! a typed reason naming the call (and the first differing position), and
//! its calls stay one record each. The append-only test is brain's rule
//! `session_not_append_only`: a template that drops an earlier answer's
//! reasoning from the next request is not append-only.

#![allow(clippy::unwrap_used)]

use crate::common;
use crate::online_calls::{calls, request, sampler, single, step, with_span, DIGEST};

use common::*;
use proptest::prelude::*;
use serde_json::{json, Value};
use splinter_core::chat::WireMessage;
use splinter_data::calls::{project_calls, CallRecord, Sampler};
use splinter_data::session::{first_divergence, FallbackReason, SessionRecord, Unmergeable};
use splinter_data::{dataset_line, Objective, SampledTurn, SampledTurns};

fn message(role: &str, content: &str, train: bool) -> WireMessage {
    WireMessage {
        role: role.into(),
        content: content.into(),
        images: Vec::new(),
        tool_calls: Vec::new(),
        tool_call_id: None,
        reasoning_content: None,
        train,
        loss_weight: None,
    }
}

/// A session of calls whose `k`-th prompt is `prompts[k]` and answer
/// `answers[k]`: call `k`'s request is the user's question and every earlier
/// answer followed by an observation, its answer supervised.
fn session(prompts: &[Vec<u32>], answers: &[Vec<u32>]) -> Vec<CallRecord> {
    (0..prompts.len())
        .map(|k| {
            let mut messages = vec![message("user", "go", false)];
            for j in 0..k {
                messages.push(message("assistant", &format!("answer {j}"), false));
                messages.push(message("user", &format!("observation {j}"), false));
            }
            messages.push(message("assistant", &format!("answer {k}"), true));
            CallRecord {
                messages,
                tools: Vec::new(),
                sampled: SampledTurns {
                    tokenizer_digest: DIGEST.into(),
                    turns: vec![Some(SampledTurn {
                        prompt_token_ids: prompts[k].clone(),
                        token_ids: answers[k].clone(),
                        behavior_logprobs: answers[k].iter().map(|_| -0.25).collect(),
                        behavior_top_ids: None,
                        behavior_top_logprobs: None,
                    })],
                },
                policy_version: 10 - k as u64,
                sampler: sampler(),
                truncated: false,
            }
        })
        .collect()
}

/// Prompts that make the session append-only: each the previous prompt, its
/// answer and the observation between.
fn append_only(first: &[u32], answers: &[Vec<u32>], between: &[Vec<u32>]) -> Vec<Vec<u32>> {
    let mut prompts = vec![first.to_vec()];
    for k in 1..answers.len() {
        let mut next = prompts[k - 1].clone();
        next.extend(&answers[k - 1]);
        next.extend(&between[k - 1]);
        prompts.push(next);
    }
    prompts
}

fn ids() -> impl Strategy<Value = Vec<u32>> {
    proptest::collection::vec(0u32..50_000, 1..12)
}

/// A session of 1 to 8 calls: the first prompt, each answer and each
/// observation between two calls.
fn sessions() -> impl Strategy<Value = (Vec<u32>, Vec<Vec<u32>>, Vec<Vec<u32>>)> {
    (1usize..8).prop_flat_map(|calls| {
        (
            ids(),
            proptest::collection::vec(ids(), calls),
            proptest::collection::vec(ids(), calls),
        )
    })
}

proptest! {
    /// An append-only session merges, and its merged record supervises
    /// exactly the per-call answers, each with its call's own ids and
    /// log-probabilities, at the index its call answered at.
    #[test]
    fn a_merged_session_holds_exactly_its_calls_answers((first, answers, between) in sessions()) {
        let calls = session(&append_only(&first, &answers, &between), &answers);
        let merged = SessionRecord::of_calls(&calls).unwrap();
        let supervised: Vec<usize> = merged.messages.iter().enumerate().filter(|(_, m)| m.train).map(|(i, _)| i).collect();
        let answered: Vec<usize> = calls.iter().map(|c| c.messages.len() - 1).collect();
        prop_assert_eq!(supervised, answered);
        for (call, turn) in calls.iter().zip(&merged.sampled.turns) {
            prop_assert_eq!(&call.sampled.turns[0], turn);
            let at = call.messages.len() - 1;
            prop_assert_eq!(&call.messages[at], &merged.messages[at]);
        }
        prop_assert_eq!(merged.policy_version, 10 - (calls.len() as u64 - 1));
        let per_call: u64 = calls.iter().map(|c| {
            let t = c.sampled.turns[0].as_ref().unwrap();
            (t.prompt_token_ids.len() + t.token_ids.len()) as u64
        }).sum();
        prop_assert_eq!(merged.per_call_tokens, per_call);
        prop_assert!(merged.tokens() <= per_call);
    }

    /// Changing any id of a later prompt within the session so far, or
    /// cutting it short, is found at that call and position.
    #[test]
    fn a_changed_or_short_prompt_is_found_where_it_differs(
        (first, answers, between) in sessions().prop_filter("two calls", |s| s.1.len() > 1),
        call_pick in any::<prop::sample::Index>(),
        position_pick in any::<prop::sample::Index>(),
        cut in any::<bool>(),
    ) {
        let mut prompts = append_only(&first, &answers, &between);
        let call = 1 + call_pick.index(prompts.len() - 1);
        let so_far = prompts[call - 1].len() + answers[call - 1].len();
        // A prompt cut to nothing names no ids at all: that is a missing trace.
        let position = if cut { 1 + position_pick.index(so_far - 1) } else { position_pick.index(so_far) };
        if cut {
            prompts[call].truncate(position);
        } else {
            prompts[call][position] = prompts[call][position].wrapping_add(1);
        }
        let calls = session(&prompts, &answers);
        let turns: Vec<&SampledTurn> = calls.iter().map(|c| c.sampled.turns[0].as_ref().unwrap()).collect();
        prop_assert_eq!(first_divergence(&turns), Some((call, position)));
        prop_assert_eq!(SessionRecord::of_calls(&calls), Err(Unmergeable::NotAppendOnly { call, position }));
    }
}

/// Qwen3's habit: the next request holds an earlier answer without the
/// reasoning the model sampled, so the session is not append-only from the
/// first id of that answer, and the session stays one record per call.
#[test]
fn a_template_that_drops_earlier_reasoning_is_not_append_only() {
    let think = [900, 901, 902];
    let answer = |text: u32| -> Vec<u32> { think.iter().copied().chain([text, 999]).collect() };
    let answers = vec![answer(1), answer(2), answer(3)];
    let mut prompts = vec![vec![10, 11]];
    for k in 1..3 {
        let mut next = prompts[k - 1].clone();
        next.extend(&answers[k - 1][think.len()..]);
        next.push(50 + k as u32);
        prompts.push(next);
    }
    let calls = session(&prompts, &answers);
    let why = SessionRecord::of_calls(&calls).unwrap_err();
    assert_eq!(
        why,
        Unmergeable::NotAppendOnly {
            call: 1,
            position: 2
        }
    );
    assert!(why.to_string().contains("position 2"), "{why}");
}

/// Each typed reason a session is not merged: never a merged record that
/// brain would refuse or train in a context the sampler did not see.
#[test]
fn a_session_that_cannot_merge_says_why_and_where() {
    let answers = vec![vec![7, 8], vec![5], vec![6]];
    let prompts = append_only(&[1, 2], &answers, &[vec![9], vec![4]]);
    let base = || session(&prompts, &answers);

    let mut calls = base();
    calls[2].sampler = Sampler::untruncated(1.0).unwrap();
    assert_eq!(
        SessionRecord::of_calls(&calls),
        Err(Unmergeable::SamplerDiffers { call: 2 })
    );

    let mut calls = base();
    calls[1].sampled.tokenizer_digest = "sha256:00".into();
    assert_eq!(
        SessionRecord::of_calls(&calls),
        Err(Unmergeable::TokenizerDiffers { call: 1 })
    );

    let mut calls = base();
    calls[1].sampled.turns[0]
        .as_mut()
        .unwrap()
        .behavior_logprobs
        .pop();
    assert_eq!(
        SessionRecord::of_calls(&calls),
        Err(Unmergeable::MissingIds { call: 1 })
    );

    let mut calls = base();
    calls[2].sampled.turns = vec![None];
    assert_eq!(
        SessionRecord::of_calls(&calls),
        Err(Unmergeable::MissingIds { call: 2 })
    );

    let mut calls = base();
    calls[0].tools = vec![json!({"name": "shell"})];
    assert_eq!(
        SessionRecord::of_calls(&calls),
        Err(Unmergeable::ToolsDiffer { call: 0 })
    );

    // The second request re-wrote the first answer: the ids agree, the
    // messages do not.
    let mut calls = base();
    calls[1].messages[1].content = "answer 0, re-written".into();
    assert_eq!(
        SessionRecord::of_calls(&calls),
        Err(Unmergeable::MessagesDiverge {
            call: 1,
            message: 1
        })
    );

    let merged = SessionRecord::of_calls(&base()).unwrap();
    let tokens = merged.tokens();
    assert_eq!(tokens, 8, "the last prompt (7 ids) and its answer");
    assert_eq!(SessionRecord::of_calls(&[]), Err(Unmergeable::NotExact));
}

/// From what the runtime recorded: the projection of a two-call rollout
/// merges into one rollout line carrying both answers' exact fields, and an
/// inexact projection never merges.
#[test]
fn a_recorded_append_only_rollout_is_one_brain_rollout_line() {
    let first = request(&[1, 2], &[7, 8], "4");
    let second = request(&[1, 2, 7, 8, 9], &[5], "3");
    let trajectory = with_span(
        trajectory(vec![
            user(1, "List the files."),
            step(2, "", single(&first)),
            step(3, "One file: a.txt.", single(&second)),
        ]),
        [3, 4],
    );
    let projection = project_calls(&trajectory, &calls(), &sampler());
    let [merged] = SessionRecord::segments(&projection.records, None)
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!((merged.tokens(), merged.per_call_tokens), (6, 10));
    let line: Value = serde_json::from_str(
        &dataset_line(
            Objective::PolicyGradient,
            &merged.record(1.0, "online:7", "online:7/0"),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        (&line["reward"], &line["group"], &line["policy_version"]),
        (&json!(1.0), &json!("online:7"), &json!(3)),
        "the oldest call's version"
    );
    assert_eq!(line["tokenizer_digest"], DIGEST);
    assert_eq!(line["sampler"], json!({"temperature": 0.8}));
    let trained: Vec<(usize, &Value, &Value)> = line["messages"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .filter(|(_, m)| m["train"] == json!(true))
        .map(|(i, m)| (i, &m["prompt_token_ids"], &m["token_ids"]))
        .collect();
    assert_eq!(
        trained,
        [
            (2, &json!([1, 2]), &json!([7, 8])),
            (4, &json!([1, 2, 7, 8, 9]), &json!([5]))
        ]
    );

    let inexact = project_calls(
        &trajectory,
        &calls()[..1],
        &Sampler::untruncated(0.8).unwrap(),
    );
    assert!(
        !inexact.exact(),
        "an inexact projection is written per call"
    );
}

/// A session compacted between calls is several append-only segments: each
/// is one merged record, all carrying the session's one reward and rollout,
/// and together they supervise every call's answer exactly once.
#[test]
fn a_compacted_session_yields_one_merged_record_per_segment_with_one_reward() {
    let answers = vec![vec![7, 8], vec![5], vec![6], vec![3, 4]];
    let mut prompts = append_only(&[1, 2], &answers, &[vec![9], vec![4], vec![2]]);
    // Compaction before call 2: its prompt is notes, not the session so far.
    prompts[2] = vec![1, 77, 78];
    prompts[3] = [prompts[2].clone(), answers[2].clone(), vec![2]].concat();
    let mut calls = session(&prompts, &answers);
    calls[2].messages = vec![
        message("user", "notes", false),
        message("assistant", "answer 2", true),
    ];
    calls[3].messages = vec![
        message("user", "notes", false),
        message("assistant", "answer 2", false),
        message("user", "observation 2", false),
        message("assistant", "answer 3", true),
    ];
    let segments = SessionRecord::segments(&calls, None).unwrap();
    assert_eq!(segments.len(), 2, "one record per segment");
    assert_eq!(
        segments
            .iter()
            .map(|s| s.sampled.turns.len())
            .collect::<Vec<_>>(),
        [2, 2]
    );
    let records: Vec<_> = segments.iter().map(|s| s.record(1.0, "g", "r7")).collect();
    for record in &records {
        let line = serde_json::to_value(record).unwrap();
        let text = line.to_string();
        assert!(
            text.contains("\"reward\":1.0") && text.contains("\"rollout\":\"r7\""),
            "{text}"
        );
    }
    // A block too small for a segment splits it at a call.
    let small = SessionRecord::segments(&calls, Some(6)).unwrap();
    assert!(
        small.len() > 2
            && small
                .iter()
                .all(|s| s.tokens() <= 6 || s.sampled.turns.len() == 1)
    );
}

/// Only two things keep a session from merged records: a call with no sampled
/// ids (untraced) and calls drawn under different samplers.
#[test]
fn a_session_falls_back_only_when_untraced_or_of_mixed_samplers() {
    let answers = vec![vec![7, 8], vec![5]];
    let prompts = append_only(&[1, 2], &answers, &[vec![9]]);
    let mut calls = session(&prompts, &answers);
    calls[1].sampled.turns = vec![None];
    assert_eq!(
        SessionRecord::segments(&calls, None),
        Err(FallbackReason::Untraced)
    );
    let mut calls = session(&prompts, &answers);
    calls[1].sampler = Sampler::untruncated(1.0).unwrap();
    assert_eq!(
        SessionRecord::segments(&calls, None),
        Err(FallbackReason::MixedSampler)
    );
    assert_eq!(FallbackReason::Untraced.name(), "untraced");
    assert_eq!(FallbackReason::MixedSampler.name(), "mixed_sampler");
}

/// The ids a reply's `sampled` record carries (unsigned LEB128 varints,
/// base64) read back as the ids.
#[test]
fn a_sampled_records_ids_decode() {
    assert_eq!(
        splinter_data::decode_ids("AAF/gAGAgAHboAk=").unwrap(),
        vec![0, 1, 127, 128, 16384, 151_643]
    );
    assert!(
        splinter_data::decode_ids("gA==").is_err(),
        "cut inside a varint"
    );
}

/// A call trains its traced ids when the server's `sampled` record (kept as
/// the exact text it sent) names them too: generated ids, and the prompt by
/// length, digest and generation header. A record naming other ids leaves
/// the call untrained.
#[test]
fn a_call_trains_the_ids_its_sampled_record_names() {
    // The sha256 of the prompt ids 1, 2, 7, 8, 9 as little-endian u32.
    const CONTEXT_HASH: &str = "4e4166f2e43c7b261d18887d3b23b1b3d61cfa5c19e39b2f76d37c0d55bf70a6";
    let project = |token_ids: &str| {
        let first = request(&[1, 2], &[7, 8], "4");
        let second = request(&[1, 2, 7, 8, 9], &[5], "3");
        // sven keeps the record beside the step's metrics.
        let mut answered = step(3, "One file: a.txt.", single(&second));
        answered.extra = Some(json!({"sven": {"sampled": format!(
            r#"{{"token_ids":"{token_ids}","prefill_token_ids":"CQ==","context_len":5,"context_hash":"sha256:{CONTEXT_HASH}","finish":"stop"}}"#
        )}}));
        let trajectory = with_span(
            trajectory(vec![
                user(1, "List the files."),
                step(2, "", single(&first)),
                answered,
            ]),
            [3, 4],
        );
        project_calls(&trajectory, &calls(), &sampler())
    };
    let agreeing = project("BQ==");
    assert!(agreeing.exact(), "{:?}", agreeing.proxy);
    assert_eq!(
        agreeing.records[1].sampled.turns[0]
            .as_ref()
            .unwrap()
            .token_ids,
        [5]
    );

    let differing = project("Bg==");
    assert!(!differing.exact());
    assert_eq!(
        differing
            .proxy
            .get(&splinter_data::ProxyReason::SampledDiffers),
        Some(&1)
    );
}
