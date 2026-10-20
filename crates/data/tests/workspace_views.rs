// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training-data projection for agents that
// work in a source tree, for its clients. If your team needs expertise in
// turning verified agent trajectories into fine-tuning data, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: an agent that worked in a workspace is trained on each action it
//! took, as it will be served.
//!
//! * The feedback the visible checks sent back is context in every record
//!   after it; elsewhere a second user turn is still refused.
//! * A record carries the tool schemas the agent was offered, in the form
//!   brain's chat template renders, and opens with the system prompt the
//!   harness pinned when the trajectory records one.
//! * Every assistant turn is trimmed.
//! * What cannot be trained on is left out and counted by reason: reasoning
//!   markup in text, a compacted context, tool arguments that were not JSON,
//!   and a record over the training block.
//! * A workspace episode is not a closed-book final answer.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use atif::{function_tool, ContextManagement, MessageBody, StepOrigin, TraceStep};
use common::*;
use serde_json::{json, Value};
use splinter_core::annotation::{Outcome, Strength};
use splinter_core::chat::WireMessage;
use splinter_core::experience::{Environment, Experience, Task};
use splinter_core::prompt::SYSTEM_PROMPT;
use splinter_data::{Corpus, Exclusion, Projection, Record, RecordBody, SftFinal, SftStep, View};

const INSTRUCTION: &str = "Make the failing test pass.";

fn workspace_task() -> Task {
    Task::new(
        "workspace-change",
        Vec::new(),
        Environment::new(Environment::WORKSPACE, json!({"snapshot": "tree-1"})),
        INSTRUCTION,
        Vec::new(),
    )
    .unwrap()
}

fn offered() -> Vec<Value> {
    vec![
        function_tool("shell", "Run a command.", &json!({"type": "object"})),
        function_tool("edit", "Edit a file.", &json!({"type": "object"})),
    ]
}

fn with_tools(mut t: atif::Trajectory, tools: Vec<Value>) -> atif::Trajectory {
    t.agent.tool_definitions = Some(tools);
    t
}

/// An agent step whose text is `text` and which calls `name` once.
fn narrated(step: u64, text: &str, id: &str, name: &str) -> TraceStep {
    let mut s = call(step, id, name, json!({"cmd": "ls"}), "a.py");
    s.message = MessageBody::Text(text.into());
    s
}

/// A workspace run with two rounds of feedback from the visible checks.
fn two_feedback_rounds() -> Vec<TraceStep> {
    vec![
        user(1, &format!("{INSTRUCTION}\n{SECRET}")),
        call(2, "c1", "shell", json!({"cmd": "ls"}), "a.py"),
        call(3, "c2", "edit", json!({"path": "a.py", "edit": 1}), "ok"),
        user(4, "Check `unit` failed:\nassert 1 == 2"),
        call(5, "c3", "edit", json!({"path": "a.py", "edit": 2}), "ok"),
        user(6, "Check `unit` failed:\nassert 2 == 3"),
        call(7, "c4", "edit", json!({"path": "a.py", "edit": 3}), "ok"),
        say(8, "Done."),
    ]
}

fn passed(task: &Task, steps: Vec<TraceStep>, tools: Vec<Value>, at: &str) -> Experience {
    experience(
        task,
        with_tools(trajectory(steps), tools),
        Some("PATCH"),
        at,
    )
}

fn corpus_of(experiences: Vec<Experience>) -> Corpus {
    let mut corpus = Corpus::new();
    for exp in experiences {
        let notes = vec![verdict(&exp, Outcome::Pass, Strength::Executable)];
        corpus.insert(exp, notes).unwrap();
    }
    corpus
}

fn step_records(experiences: Vec<Experience>) -> Projection {
    SftStep::new(Strength::Executable)
        .project(&corpus_of(experiences))
        .unwrap()
}

fn parts(record: &Record) -> (&[WireMessage], &[Value]) {
    match &record.body {
        RecordBody::Chat { messages, tools } => (messages, tools),
        other => panic!("a chat record, got {other:?}"),
    }
}

fn trained(messages: &[WireMessage]) -> Vec<usize> {
    messages
        .iter()
        .enumerate()
        .filter(|(_, m)| m.train)
        .map(|(i, _)| i)
        .collect()
}

#[test]
fn a_workspace_run_yields_one_record_per_action_with_the_feedback_as_context() {
    let projection = step_records(vec![passed(
        &workspace_task(),
        two_feedback_rounds(),
        offered(),
        "2026-10-13T01:00:00.000Z",
    )]);

    assert_eq!(
        projection.records.len(),
        5,
        "four tool calls and the answer"
    );
    for record in &projection.records {
        let (messages, tools) = parts(record);
        assert_eq!(tools, offered(), "the tools the agent was offered");
        assert_eq!(trained(messages), [messages.len() - 1], "only the action");
        assert_eq!(messages.last().unwrap().role, "assistant");
        assert!(
            messages.iter().all(|m| !m.content.contains(SECRET)),
            "the solver's own prompt is not the student's"
        );
    }
    let (last, _) = parts(&projection.records[4]);
    assert_eq!(last.last().unwrap().content, "Done.");
    let feedback: Vec<&str> = last
        .iter()
        .filter(|m| m.role == "user")
        .map(|m| m.content.as_str())
        .collect();
    assert_eq!(
        feedback,
        [
            INSTRUCTION,
            "Check `unit` failed:\nassert 1 == 2",
            "Check `unit` failed:\nassert 2 == 3"
        ],
        "the checks' output is in the context, in order"
    );
    let (before_feedback, _) = parts(&projection.records[1]);
    assert_eq!(
        before_feedback.iter().filter(|m| m.role == "user").count(),
        1,
        "an action is shown only what the agent had seen when it took it"
    );
}

#[test]
fn outside_a_workspace_a_second_user_turn_is_still_refused() {
    let closed = task("arithmetic", INSTRUCTION, Vec::new());
    let projection = step_records(vec![passed(
        &closed,
        two_feedback_rounds(),
        Vec::new(),
        "2026-10-13T01:00:00.000Z",
    )]);
    assert!(projection.records.is_empty());
    assert_eq!(projection.count(Exclusion::Unrepresentable), 1);
}

#[test]
fn a_reply_that_feedback_answered_is_context_and_not_an_action() {
    let steps = vec![
        user(1, INSTRUCTION),
        call(2, "c1", "edit", json!({"path": "a.py", "edit": 1}), "ok"),
        say(3, "I think that is done."),
        user(4, "Check `unit` failed."),
        call(5, "c2", "edit", json!({"path": "a.py", "edit": 2}), "ok"),
        say(6, "Fixed."),
    ];
    let projection = step_records(vec![passed(
        &workspace_task(),
        steps,
        offered(),
        "2026-10-13T01:00:00.000Z",
    )]);
    assert_eq!(projection.records.len(), 3);
    let (final_record, _) = parts(&projection.records[2]);
    let premature = final_record
        .iter()
        .find(|m| m.content == "I think that is done.")
        .unwrap();
    assert!(!premature.train, "shown, never supervised");
}

#[test]
fn every_assistant_turn_is_trimmed() {
    let steps = vec![
        user(1, INSTRUCTION),
        narrated(2, "\n\nLook around.\n", "c1", "shell"),
        narrated(3, "  Edit it.  ", "c2", "edit"),
        say(4, "\nDone.\n\n"),
    ];
    let projection = step_records(vec![passed(
        &workspace_task(),
        steps,
        offered(),
        "2026-10-13T01:00:00.000Z",
    )]);
    assert_eq!(projection.records.len(), 3);
    for record in &projection.records {
        for message in parts(record).0.iter().filter(|m| m.role == "assistant") {
            assert_eq!(message.content, message.content.trim());
        }
    }
    let (last, _) = parts(&projection.records[2]);
    let said: Vec<&str> = last
        .iter()
        .filter(|m| m.role == "assistant")
        .map(|m| m.content.as_str())
        .collect();
    assert_eq!(said, ["Look around.", "Edit it.", "Done."]);
}

#[test]
fn tool_schemas_reach_the_record_in_the_form_brain_renders() {
    let bare = json!({"name": "grep", "description": "Search.", "parameters": {"type": "object"}});
    let steps = vec![user(1, INSTRUCTION), say(2, "Done.")];
    let at = "2026-10-13T01:00:00.000Z";
    let projection = step_records(vec![passed(
        &workspace_task(),
        steps.clone(),
        vec![offered().remove(0), bare.clone()],
        at,
    )]);
    let (_, tools) = parts(&projection.records[0]);
    assert_eq!(tools[0], offered()[0], "an OpenAI entry is kept as it is");
    assert_eq!(
        tools[1],
        json!({"type": "function", "function": bare}),
        "a bare function schema is wrapped"
    );

    let unusable = step_records(vec![passed(
        &workspace_task(),
        steps,
        vec![json!("shell")],
        at,
    )]);
    assert!(unusable.records.is_empty());
    assert_eq!(unusable.count(Exclusion::Unrepresentable), 1);
}

#[test]
fn what_cannot_be_trained_on_is_left_out_and_counted_by_reason() {
    let task = task("arithmetic", INSTRUCTION, Vec::new());
    let at = |n: u32| format!("2026-10-13T01:00:0{n}.000Z");

    let thinking = vec![
        user(1, INSTRUCTION),
        say(2, "<think>\nhmm\n</think>\nDone."),
    ];

    let mut marker = TraceStep::new(2, StepOrigin::System, "context_compaction: 900 -> 100");
    ContextManagement::new("compaction", "replace").insert_into_extra(&mut marker.extra);
    let compacted = vec![user(1, INSTRUCTION), marker, say(3, "Done.")];

    let mut garbled = call(2, "c1", "shell", json!({}), "ok");
    garbled.tool_calls.as_mut().unwrap()[0].extra =
        Some(json!({"raw_arguments": "{\"cmd\": ", "unparsed_arguments": true}));
    let unparsed = vec![user(1, INSTRUCTION), garbled, say(3, "Done.")];

    let clean = vec![user(1, INSTRUCTION), say(2, "Done.")];

    let projection = step_records(vec![
        passed(&task, thinking, Vec::new(), &at(1)),
        passed(&task, compacted, Vec::new(), &at(2)),
        passed(&task, unparsed, offered(), &at(3)),
        passed(&task, clean, Vec::new(), &at(4)),
    ]);

    assert_eq!(projection.records.len(), 1);
    assert_eq!(projection.count(Exclusion::ReasoningInText), 1);
    assert_eq!(projection.count(Exclusion::Compacted), 1);
    assert_eq!(projection.count(Exclusion::UnparsedArguments), 1);
    assert_eq!(projection.count(Exclusion::Unrepresentable), 0);
}

#[test]
fn a_record_over_the_training_block_is_left_out_and_counted() {
    let projection = step_records(vec![passed(
        &workspace_task(),
        two_feedback_rounds(),
        offered(),
        "2026-10-13T01:00:00.000Z",
    )]);
    let size = |record: &Record| -> usize {
        let (messages, tools) = parts(record);
        messages.iter().map(|m| m.content.len()).sum::<usize>() + tools.len() * 100
    };
    let sizes: Vec<usize> = projection.records.iter().map(size).collect();
    assert!(
        sizes.windows(2).all(|w| w[0] < w[1]),
        "each action sees more"
    );
    let bound = sizes[2];

    let bounded = projection.clone().within_block(bound, &size);
    assert_eq!(bounded.records.len(), 3);
    assert_eq!(bounded.count(Exclusion::OverBlock), 2);
    let unbounded = projection.within_block(usize::MAX, &size);
    assert_eq!(unbounded.records.len(), 5);
    assert_eq!(unbounded.count(Exclusion::OverBlock), 0);
}

#[test]
fn the_record_opens_with_the_prompt_the_harness_pinned() {
    let pinned = "You are a coding agent. Work in the tree you are given.";
    let at = "2026-10-13T01:00:00.000Z";
    let mut with_prompt = two_feedback_rounds();
    with_prompt.insert(0, TraceStep::new(0, StepOrigin::System, pinned));

    let pinned_run = step_records(vec![passed(&workspace_task(), with_prompt, offered(), at)]);
    let unpinned_run = step_records(vec![passed(
        &workspace_task(),
        two_feedback_rounds(),
        offered(),
        at,
    )]);

    for record in &pinned_run.records {
        let (messages, _) = parts(record);
        assert_eq!(
            (messages[0].role.as_str(), messages[0].content.as_str()),
            ("system", pinned)
        );
        assert!(!messages[0].train);
        assert_eq!(messages.iter().filter(|m| m.role == "system").count(), 1);
    }
    for record in &unpinned_run.records {
        assert_eq!(parts(record).0[0].content, SYSTEM_PROMPT);
    }
    assert_eq!(pinned_run.records.len(), unpinned_run.records.len());
}

#[test]
fn a_workspace_episode_is_not_a_closed_book_final_answer() {
    let at = |n: u32| format!("2026-10-13T01:00:0{n}.000Z");
    let closed = task("arithmetic", INSTRUCTION, Vec::new());
    let projection = SftFinal::new(Strength::Executable)
        .project(&corpus_of(vec![
            passed(&workspace_task(), two_feedback_rounds(), offered(), &at(1)),
            answered(&closed, "4", &at(2)),
        ]))
        .unwrap();
    assert_eq!(projection.records.len(), 1);
    assert_eq!(projection.count(Exclusion::WorkspaceEpisode), 1);
}
