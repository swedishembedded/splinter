// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training-data pipelines held to a second,
// independent implementation, for its clients. If your team needs expertise
// in validating fine-tuning data, you can procure our services by sending an
// email to info@swedishembedded.com.

//! Spec, against an independent reference (plan 6B): the sft-step and
//! sft-final views write, for each frozen case, exactly the records and
//! counts the reference projector (`validation/oracles/data/sft_projection.py`,
//! a plain Python implementation written from the views' documentation)
//! wrote: every message's role, text and supervision, every tool call's
//! scoped id, tool and arguments, the call each tool reply answers, and every
//! reason something was left out.
#![allow(clippy::unwrap_used)]

use crate::common;

use common::*;
use serde_json::{json, Map, Value};
use splinter_conformance::{compare, fixture, Tolerance};
use splinter_core::annotation::{Label, Outcome, Strength};
use splinter_core::chat::WireMessage;
use splinter_core::experience::{Environment, Task};
use splinter_core::prompt::SYSTEM_PROMPT;
use splinter_data::curation::HistoryCuration;
use splinter_data::{Corpus, Projection, RecordBody, SftFinal, SftStep, View};

/// The calls the reference names by kind: tool, arguments, result.
fn call_of(kind: &str) -> Option<(&'static str, Value, &'static str)> {
    match kind {
        "history" => Some((
            "shell",
            json!({"command": "git log --oneline -3"}),
            "abc fix",
        )),
        "read" => Some((
            "read_file",
            json!({"path": "src/parse.rs"}),
            "fn parse() {}",
        )),
        "read-other" => Some(("read_file", json!({"path": "src/lib.rs"}), "pub mod parse;")),
        "write" => Some((
            "shell",
            json!({"command": "sed -i s/a/b/ src/parse.rs"}),
            "ok",
        )),
        _ => None,
    }
}

fn step_id(i: usize) -> u64 {
    i as u64 + 2
}

/// The reference's text of step `i` of `kind`.
fn text_of(kind: &str, i: usize) -> String {
    match kind {
        "say" => format!("  Reply {}.  ", step_id(i)),
        "feedback" => format!("The checks still fail ({}).", step_id(i)),
        _ if i.is_multiple_of(3) => "Looking.".into(),
        _ => String::new(),
    }
}

fn workspace_task() -> Task {
    Task::new(
        "workspace-change",
        Vec::new(),
        Environment::new(Environment::WORKSPACE, json!({"snapshot": "tree-1"})),
        "Fix the parser.",
        Vec::new(),
    )
    .unwrap()
}

/// The corpus of `case`: one experience, its verdict and labels.
fn corpus(case: &Value) -> Corpus {
    let verdict = case["verdict"].as_str();
    let (exp, labels) = if case["view"] == "sft-step" || case["setting"] == "workspace" {
        let task = workspace_task();
        let mut steps = vec![user(1, &task.instruction)];
        let kinds: Vec<&str> = case["steps"]
            .as_array()
            .map(|s| s.iter().map(|k| k.as_str().unwrap()).collect())
            .unwrap_or_else(|| vec!["say"]);
        for (i, kind) in kinds.iter().enumerate() {
            let id = step_id(i);
            steps.push(match call_of(kind) {
                Some((tool, arguments, result)) => {
                    let mut step = call(id, &format!("c{id}"), tool, arguments, result);
                    step.message = text_of(kind, i).into();
                    step
                }
                None if *kind == "say" => say(id, &text_of(kind, i)),
                None => user(id, &text_of(kind, i)),
            });
        }
        let output = case["output"].as_str().unwrap_or("done");
        let exp = experience(
            &task,
            trajectory(steps),
            Some(output),
            "2026-11-24T01:00:00.000Z",
        );
        (exp, case["labels"].as_array().cloned().unwrap_or_default())
    } else {
        let task = task("closed-qa", "What is two plus two?", vec![]);
        let output = case["output"].as_str().unwrap();
        let exp = experience(
            &task,
            trajectory(vec![user(1, &task.instruction), say(2, output)]),
            Some(output),
            "2026-11-24T01:00:00.000Z",
        );
        (exp, Vec::new())
    };
    let mut notes = Vec::new();
    match verdict {
        Some("pass") => notes.push(verdict_of(&exp, Outcome::Pass)),
        Some("fail") => notes.push(verdict_of(&exp, Outcome::Fail)),
        _ => {}
    }
    for (i, l) in labels.iter().enumerate() {
        match l.as_str() {
            Some("good") => notes.push(label(&exp, step_id(i), Label::Good)),
            Some("bad") => notes.push(label(&exp, step_id(i), Label::Bad)),
            _ => {}
        }
    }
    let mut corpus = Corpus::new();
    corpus.insert(exp, notes).unwrap();
    corpus
}

fn verdict_of(
    exp: &splinter_core::experience::Experience,
    outcome: Outcome,
) -> splinter_core::annotation::Annotation {
    verdict(exp, outcome, Strength::Executable)
}

fn view(case: &Value) -> Box<dyn View> {
    if case["view"] == "sft-final" {
        return Box::new(SftFinal::new(Strength::Executable));
    }
    let options = &case["options"];
    let history = match options["history"].as_str().unwrap() {
        "drop" => HistoryCuration::DropStep,
        "keep" => HistoryCuration::Keep,
        _ => HistoryCuration::Truncate,
    };
    let mut view = SftStep::new(Strength::Executable)
        .packed(options["packed"].as_bool().unwrap())
        .history(history);
    if let Some(only) = options["only"].as_str() {
        view = view.only_actions(vec![only.into()]);
    }
    Box::new(view)
}

/// A message in the reference's terms.
fn message(m: &WireMessage) -> Value {
    let content = if m.role == "system" && m.content == SYSTEM_PROMPT {
        "SYSTEM_PROMPT".to_string()
    } else {
        m.content.clone()
    };
    let mut out = Map::new();
    out.insert("role".into(), json!(m.role));
    out.insert("content".into(), json!(content));
    out.insert("train".into(), json!(m.train));
    if !m.tool_calls.is_empty() {
        let calls: Vec<Value> = m
            .tool_calls
            .iter()
            .map(|c| {
                json!({
                    "id": c.id,
                    "name": c.function.name,
                    "arguments": serde_json::from_str::<Value>(&c.function.arguments).unwrap(),
                })
            })
            .collect();
        out.insert("tool_calls".into(), Value::Array(calls));
    }
    if let Some(id) = &m.tool_call_id {
        out.insert("tool_call_id".into(), json!(id));
    }
    Value::Object(out)
}

fn observed(case: &Value, projection: &Projection) -> Value {
    let records: Vec<Value> = projection
        .records
        .iter()
        .map(|r| match &r.body {
            RecordBody::Chat { messages, .. } => {
                Value::Array(messages.iter().map(message).collect())
            }
            other => panic!("a chat record, got {other:?}"),
        })
        .collect();
    let excluded: Map<String, Value> = projection
        .excluded
        .iter()
        .map(|(reason, n)| (format!("{reason:?}"), json!(n)))
        .collect();
    let mut out = case.clone();
    out["records"] = Value::Array(records);
    out["excluded"] = Value::Object(excluded);
    out
}

#[test]
fn the_views_write_what_the_reference_projector_writes() {
    let fixture = fixture(env!("CARGO_MANIFEST_DIR"), "data/sft-projection").unwrap();
    let mut wrong = Vec::new();
    for case in fixture.cases() {
        let projection = view(case).project(&corpus(case)).unwrap();
        let actual = observed(case, &projection);
        if let Err(mismatch) = compare(&actual, case, Tolerance::new(0.0, 0.0)) {
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
