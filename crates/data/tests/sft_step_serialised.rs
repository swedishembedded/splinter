// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training small models on the
// verified work of larger ones for its clients. If your team needs expertise
// in turning agent trajectories into fine-tuning data then you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec: a trajectory exported as one agent step that holds a whole run's tool
//! calls and observations projects, when asked to serialise the calls, as one
//! turn per call - the assistant's call, then its result - followed by the
//! final reply, every assistant turn supervised; without the option the
//! projection is what it was.

#![allow(clippy::unwrap_used)]

mod common;

use common::*;
use serde_json::json;
use splinter_core::annotation::{Outcome, Strength};
use splinter_core::chat::WireMessage;
use splinter_core::experience::PrivilegedKind;
use splinter_data::{Corpus, Exclusion, RecordBody, SftStep, View};

use atif::{ObservationEntry, StepObservation, StepOrigin, ToolInvocation, TraceStep};

fn one_step_run() -> splinter_core::experience::Experience {
    let task = task(
        "arithmetic",
        "Compute two sums.",
        vec![privileged(PrivilegedKind::Reference, "2 4")],
    );
    let mut run = TraceStep::new(2, StepOrigin::Agent, "The answers are 2 and 4");
    run.tool_calls = Some(vec![
        ToolInvocation::new("c1", "run_code").with_arguments(json!({"code": "1+1"})),
        ToolInvocation::new("c2", "run_code").with_arguments(json!({"code": "2+2"})),
    ]);
    run.observation = Some(StepObservation {
        results: vec![
            ObservationEntry::for_call("c1", "2"),
            ObservationEntry::for_call("c2", "4"),
        ],
    });
    experience(
        &task,
        trajectory(vec![user(1, "Compute two sums."), run]),
        Some("The answers are 2 and 4"),
        "2026-09-30T01:00:00.000Z",
    )
}

fn project(serialise: bool) -> Vec<WireMessage> {
    let experience = one_step_run();
    let mut corpus = Corpus::new();
    let note = verdict(&experience, Outcome::Pass, Strength::Executable);
    corpus.insert(experience, vec![note]).unwrap();
    let projection = SftStep::new(Strength::Executable)
        .packed(true)
        .serialise_calls(serialise)
        .project(&corpus)
        .unwrap();
    match &projection.records[0].body {
        RecordBody::Chat { messages, .. } => messages.clone(),
        other => panic!("a chat record, got {other:?}"),
    }
}

#[test]
fn serialised_calls_become_one_turn_each_with_their_results_then_the_final_reply() {
    let messages = project(true);
    let roles: Vec<_> = messages.iter().map(|m| m.role.as_str()).collect();
    assert_eq!(
        roles[1..],
        [
            "user",
            "assistant",
            "tool",
            "assistant",
            "tool",
            "assistant"
        ]
    );
    let calls = |m: &WireMessage| {
        m.tool_calls
            .iter()
            .map(|c| c.id.clone().unwrap())
            .collect::<Vec<_>>()
    };
    assert_eq!(calls(&messages[2]), ["s2-c1"]);
    assert_eq!(messages[3].tool_call_id.as_deref(), Some("s2-c1"));
    assert_eq!(messages[3].content, "2");
    assert_eq!(calls(&messages[4]), ["s2-c2"]);
    assert_eq!(messages[5].content, "4");
    assert!(messages[6].tool_calls.is_empty());
    assert_eq!(messages[6].content, "The answers are 2 and 4");
    let supervised: Vec<_> = messages.iter().map(|m| m.train).collect();
    assert_eq!(supervised[1..], [false, true, false, true, false, true]);
}

#[test]
fn without_the_option_the_step_is_one_assistant_turn_as_before() {
    let messages = project(false);
    let roles: Vec<_> = messages.iter().map(|m| m.role.as_str()).collect();
    assert_eq!(roles[1..], ["user", "assistant"]);
    assert_eq!(messages[2].tool_calls.len(), 2);
}

#[test]
fn a_run_kept_as_separate_agent_steps_keeps_its_parallel_calls_in_one_turn() {
    let task = task(
        "arithmetic",
        "Compute two sums.",
        vec![privileged(PrivilegedKind::Reference, "2 4")],
    );
    let mut batch = TraceStep::new(2, StepOrigin::Agent, "");
    batch.tool_calls = Some(vec![
        ToolInvocation::new("c1", "run_code").with_arguments(json!({"code": "1+1"})),
        ToolInvocation::new("c2", "run_code").with_arguments(json!({"code": "2+2"})),
    ]);
    batch.observation = Some(StepObservation {
        results: vec![
            ObservationEntry::for_call("c1", "2"),
            ObservationEntry::for_call("c2", "4"),
        ],
    });
    let experience = experience(
        &task,
        trajectory(vec![
            user(1, "Compute two sums."),
            batch,
            say(3, "The answers are 2 and 4"),
        ]),
        Some("The answers are 2 and 4"),
        "2026-09-30T01:00:00.000Z",
    );
    let mut corpus = Corpus::new();
    let note = verdict(&experience, Outcome::Pass, Strength::Executable);
    corpus.insert(experience, vec![note]).unwrap();
    let projection = SftStep::new(Strength::Executable)
        .packed(true)
        .serialise_calls(true)
        .project(&corpus)
        .unwrap();
    let RecordBody::Chat { messages, .. } = &projection.records[0].body else {
        panic!("a chat record")
    };
    assert_eq!(
        messages[2].tool_calls.len(),
        2,
        "the two calls stay one turn"
    );
}

/// A run recorded one agent step per model response numbers each response's
/// calls from zero, so `call_0` repeats across steps. Each result is paired
/// with its call within its step, and the record names every call uniquely
/// (`s<step>-<id>`), which is what brain's record check requires.
#[test]
fn ids_repeated_across_steps_are_scoped_to_their_step() {
    let task = task(
        "arithmetic",
        "Compute two sums.",
        vec![privileged(PrivilegedKind::Reference, "2 4")],
    );
    let step = |id: u64, code: &str, out: &str| {
        let mut s = TraceStep::new(id, StepOrigin::Agent, "");
        s.tool_calls = Some(vec![
            ToolInvocation::new("call_0", "run_code").with_arguments(json!({ "code": code }))
        ]);
        s.observation = Some(StepObservation {
            results: vec![ObservationEntry::for_call("call_0", out)],
        });
        s
    };
    let experience = experience(
        &task,
        trajectory(vec![
            user(1, "Compute two sums."),
            step(2, "1+1", "2"),
            step(3, "2+2", "4"),
            say(4, "The answers are 2 and 4"),
        ]),
        Some("The answers are 2 and 4"),
        "2026-09-30T01:00:00.000Z",
    );
    let mut corpus = Corpus::new();
    let note = verdict(&experience, Outcome::Pass, Strength::Executable);
    corpus.insert(experience, vec![note]).unwrap();
    let projection = SftStep::new(Strength::Executable)
        .packed(true)
        .project(&corpus)
        .unwrap();
    assert!(projection.excluded.is_empty(), "{:?}", projection.excluded);
    let RecordBody::Chat { messages, .. } = &projection.records[0].body else {
        panic!("a chat record")
    };
    let called: Vec<_> = messages
        .iter()
        .flat_map(|m| m.tool_calls.iter().filter_map(|c| c.id.clone()))
        .collect();
    let answered: Vec<_> = messages
        .iter()
        .filter_map(|m| m.tool_call_id.clone())
        .collect();
    assert_eq!(called, ["s2-call_0", "s3-call_0"]);
    assert_eq!(answered, called);
}

/// The corruption this guards against: a step whose calls share an id (as a
/// run exported as one step of a server that numbered every call `call_0`
/// does) pairs every result with every call. The trajectory is refused and
/// counted, not projected.
#[test]
fn a_run_whose_calls_share_an_id_is_refused_and_counted() {
    let task = task(
        "arithmetic",
        "Compute two sums.",
        vec![privileged(PrivilegedKind::Reference, "2 4")],
    );
    let mut run = TraceStep::new(2, StepOrigin::Agent, "The answers are 2 and 4");
    run.tool_calls = Some(vec![
        ToolInvocation::new("call_0", "run_code").with_arguments(json!({"code": "1+1"})),
        ToolInvocation::new("call_0", "run_code").with_arguments(json!({"code": "2+2"})),
    ]);
    run.observation = Some(StepObservation {
        results: vec![
            ObservationEntry::for_call("call_0", "2"),
            ObservationEntry::for_call("call_0", "4"),
        ],
    });
    let experience = experience(
        &task,
        trajectory(vec![user(1, "Compute two sums."), run]),
        Some("The answers are 2 and 4"),
        "2026-09-30T01:00:00.000Z",
    );
    let mut corpus = Corpus::new();
    let note = verdict(&experience, Outcome::Pass, Strength::Executable);
    corpus.insert(experience, vec![note]).unwrap();
    for serialise in [true, false] {
        let projection = SftStep::new(Strength::Executable)
            .packed(true)
            .serialise_calls(serialise)
            .project(&corpus)
            .unwrap();
        assert!(projection.records.is_empty(), "serialise={serialise}");
        assert_eq!(projection.count(Exclusion::DuplicateCallId), 1);
    }
}
