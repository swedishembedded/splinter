// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training-data pipelines that refuse a
// damaged trajectory or record by name instead of crashing, for its clients.
// If your team needs expertise in property-based testing of data pipelines,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec, by property: what the data crate reads from other programs is
//! answered, a value or an error, however it is damaged, and never panics:
//!
//! * a trajectory (another agent's ATIF export), its steps removed, retyped
//!   or joined by unknown keys, through every view that projects one;
//! * a model request log, its lines truncated or changed;
//! * a training record line, as a dataset file holds it.
//!
//! A projection that is made also keeps the supervision rule: a record
//! supervises only the assistant's turns (the transition view: only the
//! tool's).
#![allow(clippy::unwrap_used)]

use crate::common;

use common::*;
use proptest::prelude::*;
use serde_json::{json, Value};
use splinter_conformance::fuzz::{config, damaged_bytes, damaged_json, damaged_text};
use splinter_core::annotation::{Label, Outcome, Strength};
use splinter_core::clock::FixedClock;
use splinter_core::experience::{Environment, Experience, Provenance, Task};
use splinter_data::load::{ChatRecord, Record};
use splinter_data::request_log::read_request_log;
use splinter_data::{Corpus, PolicyGradient, RecordBody, SftFinal, SftStep, Transition, View};

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

/// A trajectory holding each kind of step: the instruction, a history read,
/// a file read, feedback, a write, the answer.
fn seed() -> Value {
    serde_json::to_value(trajectory(vec![
        user(1, "Fix the parser."),
        call(
            2,
            "c1",
            "shell",
            json!({"command": "git log -3"}),
            "abc fix",
        ),
        call(
            3,
            "c2",
            "read_file",
            json!({"path": "src/parse.rs"}),
            "fn parse() {}",
        ),
        user(4, "The tests still fail."),
        call(
            5,
            "c3",
            "shell",
            json!({"command": "sed -i s/a/b/ src/parse.rs"}),
            "ok",
        ),
        say(6, "The parser is fixed."),
    ]))
    .unwrap()
}

/// Every view that projects an agent's trajectory, as a user would run it.
fn views() -> Vec<Box<dyn View>> {
    vec![
        Box::new(SftStep::new(Strength::Executable)),
        Box::new(
            SftStep::new(Strength::Executable)
                .packed(true)
                .serialise_calls(true),
        ),
        Box::new(SftFinal::new(Strength::Executable)),
        Box::new(PolicyGradient::new(Strength::Executable)),
        Box::new(Transition::new()),
    ]
}

/// Projects `trajectory` (if it is one) through every view, checking each
/// record's mask.
fn project(trajectory: Value) -> Result<(), TestCaseError> {
    let Ok(trajectory) = serde_json::from_value::<atif::Trajectory>(trajectory) else {
        return Ok(());
    };
    let task = workspace_task();
    let Ok(exp) = Experience::new(
        task,
        trajectory,
        Some("The parser is fixed.".into()),
        Provenance::new(
            "scripted/solver-1",
            &FixedClock::new("2026-11-24T01:00:00.000Z"),
        ),
    ) else {
        return Ok(());
    };
    let notes = vec![
        verdict(&exp, Outcome::Pass, Strength::Executable),
        label(&exp, 3, Label::Good),
    ];
    let mut corpus = Corpus::new();
    if corpus.insert(exp, notes).is_err() {
        return Ok(());
    }
    for view in views() {
        let Ok(projection) = view.project(&corpus) else {
            continue;
        };
        // The transition view supervises what the environment answered and
        // never a model's turn; every other view the model's turns only.
        let supervisable = if view.name() == "transition" {
            "tool"
        } else {
            "assistant"
        };
        for record in &projection.records {
            if let RecordBody::Chat { messages, .. } = &record.body {
                prop_assert!(
                    messages.iter().all(|m| !m.train || m.role == supervisable),
                    "{} supervised a turn not the {supervisable}'s",
                    view.name()
                );
            }
        }
    }
    Ok(())
}

const REQUEST_LOG: &str = r#"{"kind":"request","seq":1,"model":"m","request":{"messages":[{"role":"system","content":"Be brief."},{"role":"user","content":"Fix it."}],"tools":[{"type":"function","function":{"name":"shell","parameters":{"type":"object"}}}]}}
{"kind":"response","seq":1,"response":{"text":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"shell","arguments":"{\"command\":\"ls\"}"}}]}}
{"kind":"request","seq":2,"model":"m","request":{"messages":[{"role":"user","content":"Fix it."},{"role":"tool","tool_call_id":"c1","content":"src"}],"tools":[]}}
{"kind":"response","seq":2,"response":{"text":"Done."}}
"#;

const RECORDS: &str = include_str!("../fixtures/training-records/v2/chat_rollout.jsonl");

#[test]
fn every_seed_is_read_as_written() {
    let trajectory: atif::Trajectory = serde_json::from_value(seed()).unwrap();
    assert_eq!(trajectory.steps.len(), 6);
    assert_eq!(read_request_log(REQUEST_LOG.as_bytes()).unwrap().len(), 2);
    for line in RECORDS.lines() {
        ChatRecord::parse(line).unwrap();
    }
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn a_damaged_trajectory_is_answered_by_every_view(trajectory in damaged_json(seed())) {
        project(trajectory)?;
    }

    #[test]
    fn a_trajectory_damaged_as_bytes_is_answered(
        bytes in damaged_bytes(serde_json::to_vec(&seed()).unwrap())
    ) {
        if let Ok(value) = serde_json::from_slice::<Value>(&bytes) {
            project(value)?;
        }
    }

    #[test]
    fn a_damaged_request_log_is_answered(log in damaged_bytes(REQUEST_LOG.as_bytes().to_vec())) {
        let _ = read_request_log(&log);
    }

    #[test]
    fn a_damaged_record_line_is_answered(
        line in prop::sample::select(RECORDS.lines().collect::<Vec<_>>())
            .prop_flat_map(damaged_text)
    ) {
        if let Ok(record) = ChatRecord::parse(&line) {
            prop_assert_eq!(record.line(), line.as_str(), "the line is kept as read");
        }
    }
}
