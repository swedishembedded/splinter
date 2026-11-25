// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training-data pipelines held to a second,
// independent implementation, for its clients. If your team needs expertise
// in validating preference data, you can procure our services by sending an
// email to info@swedishembedded.com.

//! Spec, against an independent reference (plan 6B): for each frozen case,
//! the preference view makes of recorded preferences exactly the records the
//! reference (`validation/oracles/data/recorded_preferences.py`, written from
//! the view's documentation) made, in its order, each the chosen
//! experience's instruction and the two final outputs with their last tool
//! calls, and leaves out exactly the relations it left out, for the reason
//! it gave. No experience carries a verdict, so the view's other sources
//! make nothing; what they leave out is not compared.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use crate::common;

use common::{call, experience, relation, say, task, trajectory, user};
use serde_json::{json, Map, Value};
use splinter_conformance::{compare, fixture, Tolerance};
use splinter_core::annotation::{RelationKind, Strength};
use splinter_core::experience::{Experience, Privileged, PrivilegedKind};
use splinter_data::{Corpus, Preference, RecordBody, View};

/// What the recorded source leaves out; the other sources' reasons are not
/// this fixture's.
const RECORDED: [&str; 5] = [
    "RelatedMissing",
    "Duplicate",
    "DifferentTask",
    "NoFinalOutput",
    "IdenticalOutputs",
];

fn experiences(case: &Value) -> Vec<Experience> {
    case["experiences"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(n, e)| {
            let instruction = if e["task"] == "A" {
                "What is two plus two?"
            } else {
                "Name a prime above ten."
            };
            let critiques = if e["critique"].as_bool().unwrap() {
                vec![Privileged {
                    kind: PrivilegedKind::Critique,
                    content: "The earlier attempt did not show its work.".into(),
                    span: None,
                }]
            } else {
                Vec::new()
            };
            let output = e["output"].as_str();
            let last = if e["calls"].as_array().unwrap().is_empty() {
                say(2, output.unwrap_or(""))
            } else {
                call(2, "c1", "submit", json!({}), "accepted")
            };
            experience(
                &task("arithmetic", instruction, critiques),
                trajectory(vec![user(1, instruction), last]),
                output,
                &format!("2026-09-30T01:00:{n:02}.000Z"),
            )
        })
        .collect()
}

fn observed(case: &Value) -> Value {
    let all = experiences(case);
    let mut corpus = Corpus::new();
    for (n, e) in case["experiences"].as_array().unwrap().iter().enumerate() {
        if !e["in_corpus"].as_bool().unwrap() {
            continue;
        }
        let notes = e["prefers"]
            .as_array()
            .unwrap()
            .iter()
            .map(|j| {
                relation(
                    &all[n],
                    RelationKind::PreferredOver,
                    &all[j.as_u64().unwrap() as usize],
                )
            })
            .collect();
        corpus.insert(all[n].clone(), notes).unwrap();
    }
    let index = |id: &splinter_core::experience::ExperienceId| {
        all.iter().position(|e| e.id().unwrap() == *id).unwrap()
    };
    let projection = Preference::new(Strength::Judged).project(&corpus).unwrap();
    let records: Vec<Value> = projection
        .records
        .iter()
        .map(|r| {
            let RecordBody::Preference {
                prompt,
                chosen,
                rejected,
                ..
            } = &r.body
            else {
                panic!("a preference record, got {:?}", r.body);
            };
            let side = |m: &[splinter_core::chat::WireMessage]| {
                let [reply] = m else {
                    panic!("one reply, got {m:?}")
                };
                let calls: Vec<&str> = reply
                    .tool_calls
                    .iter()
                    .map(|c| c.function.name.as_str())
                    .collect();
                json!({"content": reply.content, "calls": calls})
            };
            json!({
                "chosen": index(&r.metadata.experiences[0]),
                "rejected": index(&r.metadata.experiences[1]),
                "prompt": prompt.iter().rev().find(|m| m.role == "user").unwrap().content,
                "better": side(chosen),
                "worse": side(rejected),
            })
        })
        .collect();
    let excluded: Map<String, Value> = projection
        .excluded
        .iter()
        .map(|(reason, n)| (format!("{reason:?}"), json!(n)))
        .filter(|(reason, _)| RECORDED.contains(&reason.as_str()))
        .collect();
    let mut out = case.clone();
    out["records"] = Value::Array(records);
    out["excluded"] = Value::Object(excluded);
    out
}

#[test]
fn recorded_preferences_are_paired_and_left_out_as_the_reference_does() {
    let fixture = fixture(env!("CARGO_MANIFEST_DIR"), "data/recorded-preferences").unwrap();
    let mut wrong = Vec::new();
    for case in fixture.cases() {
        if let Err(mismatch) = compare(&observed(case), case, Tolerance::new(0.0, 0.0)) {
            wrong.push(format!("{mismatch}\n  case: {case}"));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of {} cases disagree with the reference; the first:\n{}",
        wrong.len(),
        fixture.cases().len(),
        wrong[..wrong.len().min(3)].join("\n")
    );
}
