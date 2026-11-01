// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training data for multi-role agent systems,
// where each model role learns from exactly the requests it was sent, for its
// clients. If your team needs expertise in fine-tuning the roles of an agent
// harness, including those that read images, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Spec: a harness conversation that recorded the requests its model was sent
//! is trained on them. Each action's record is its request as sent - the
//! pictures the model was shown in the turns that carried them - with the
//! model's answer as the only supervised turn; for a conversation of text it
//! is the record its trajectory gives. A conversation whose model was shown
//! images is built only when the view is told the base reads them, and an
//! image never lands in a supervised turn.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use atif::{StepOrigin, TraceStep};
use common::*;
use serde_json::{json, Value};
use splinter_core::annotation::{Outcome, Strength};
use splinter_core::chat::WireMessage;
use splinter_core::experience::{Environment, Task};
use splinter_core::harness::{ModelCall, ModelResponse, ModelToolCall, RecordedCalls};
use splinter_data::{Corpus, Exclusion, Projection, Record, RecordBody, SftStep, View};

const SYSTEM: &str = "You are a careful engineer.";
const PNG: &str = "data:image/png;base64,iVBORw0KGgo=";
const PNG_TOO: &str = "data:image/png;base64,iVBORw0KGgoAAAA=";

fn bench_task() -> Task {
    Task::new(
        "bench/cad",
        Vec::new(),
        Environment::new(Environment::BENCH, json!({"snapshot": "s1"})),
        "Phase: design",
        Vec::new(),
    )
    .unwrap()
}

fn shell() -> Value {
    json!({"name": "shell", "description": "Run a command.", "parameters": {"type": "object"}})
}

fn text(role: &str, content: &str) -> Value {
    json!({"role": role, "content": content})
}

/// A sven history of the model's turn: its text with reasoning, one message
/// per call, the call's result.
fn turn_and_result(id: &str, result: Value) -> Vec<Value> {
    vec![
        json!({"role": "assistant", "content": "\n\n\n", "reasoning": " look "}),
        json!({"role": "assistant", "content": {"tool_call_id": id, "function": {
            "name": "shell", "arguments": "{\"command\":\"ls\"}"}}}),
        json!({"role": "tool", "content": {"tool_call_id": id, "content": result}}),
    ]
}

fn call_response() -> ModelResponse {
    ModelResponse {
        text: "\n\n\n".into(),
        reasoning: Some("look".into()),
        tool_calls: vec![ModelToolCall {
            id: "c0".into(),
            name: "shell".into(),
            // A response carries the arguments as the model spaced them.
            arguments: "{\"command\": \"ls\"}".into(),
        }],
    }
}

fn final_response() -> ModelResponse {
    ModelResponse {
        text: "\n\nDone.".into(),
        reasoning: None,
        tool_calls: Vec::new(),
    }
}

/// The two requests of a conversation: the opening (system, `user`) and, after
/// one tool call whose result was `result`, the request that is answered with
/// the final text.
fn two_calls(user: Value, result: Value) -> Vec<ModelCall> {
    let opening = vec![
        text("system", SYSTEM),
        json!({"role": "user", "content": user}),
    ];
    let mut later = opening.clone();
    later.extend(turn_and_result("c0", result));
    vec![
        ModelCall {
            seq: 1,
            messages: opening,
            tools: vec![shell()],
            response: call_response(),
        },
        ModelCall {
            seq: 2,
            messages: later,
            tools: vec![shell()],
            response: final_response(),
        },
    ]
}

/// What the harness recorded of the conversation.
fn harness(role: &str, shown_images: bool) -> Value {
    json!({"conversation": role, "role": role, "role_source": "export",
           "shown_images": shown_images})
}

/// A conversation whose trajectory is `steps` (a system step, the user step
/// when `instruction` is given, a step that calls `shell`, a final step),
/// with its recorded `calls` if it has any.
fn conversation(
    role: &str,
    instruction: Option<&str>,
    calls: Option<&[ModelCall]>,
    shown_images: bool,
) -> Corpus {
    let mut call_step = call(3, "c0", "shell", json!({"command": "ls"}), "part.py");
    call_step.reasoning_content = Some("look".into());
    let mut steps = vec![TraceStep::new(1, StepOrigin::System, SYSTEM)];
    if let Some(instruction) = instruction {
        steps.push(user(2, instruction));
    }
    steps.push(call_step);
    steps.push(say(4, "\n\nDone."));
    let mut trajectory = trajectory(steps);
    trajectory.agent.tool_definitions =
        Some(vec![json!({"type": "function", "function": shell()})]);
    let mut extra = json!({ "splinter_harness": harness(role, shown_images) });
    if let Some(calls) = calls {
        extra["splinter_harness_calls"] = RecordedCalls::record("abc", calls).to_value();
    }
    trajectory.extra = Some(extra);
    let experience = experience(
        &bench_task(),
        trajectory,
        Some("Done."),
        "2026-10-30T01:00:00.000Z",
    );
    let notes = vec![verdict(&experience, Outcome::Pass, Strength::Executable)];
    let mut corpus = Corpus::new();
    corpus.insert(experience, notes).unwrap();
    corpus
}

fn view() -> SftStep {
    SftStep::new(Strength::Executable)
}

fn chat(record: &Record) -> (&[WireMessage], &[Value]) {
    match &record.body {
        RecordBody::Chat {
            messages, tools, ..
        } => (messages, tools),
        other => panic!("a chat record, got {other:?}"),
    }
}

fn project(view: &SftStep, corpus: &Corpus) -> Projection {
    view.project(corpus).unwrap()
}

#[test]
fn a_conversation_of_text_gives_the_records_its_trajectory_gives() {
    let calls = two_calls(json!("Phase: design"), json!("part.py"));
    let from_trajectory = project(
        &view(),
        &conversation("designer", Some("Phase: design"), None, false),
    );
    let from_requests = project(
        &view(),
        &conversation("designer", Some("Phase: design"), Some(&calls), false),
    );
    assert_eq!(from_trajectory.records.len(), 2);
    let bodies = |p: &Projection| p.records.iter().map(|r| r.body.clone()).collect::<Vec<_>>();
    assert_eq!(bodies(&from_requests), bodies(&from_trajectory));
    // The call is paired as the trajectory pairs it, and what the model
    // emitted is one assistant message with its reasoning.
    let (messages, tools) = chat(&from_requests.records[1]);
    assert_eq!(tools.len(), 1);
    assert_eq!(messages[2].reasoning_content.as_deref(), Some("look"));
    assert_eq!(messages[2].tool_calls[0].id.as_deref(), Some("s3-c0"));
    assert_eq!(messages[3].tool_call_id.as_deref(), Some("s3-c0"));
}

#[test]
fn the_record_holds_what_the_model_was_asked_and_offered_not_what_the_trial_recorded() {
    // The trial's task describes the conversation, its agent record lists
    // every tool of the harness, and the request holds the turn the model
    // was given and the tools it was offered.
    let calls = two_calls(json!("Phase: design"), json!("part.py"));
    let mut corpus = Corpus::new();
    let source = conversation("designer", Some("Phase: design"), Some(&calls), false);
    for entry in source.entries() {
        let mut experience = entry.experience.clone();
        experience.instruction = "Conversation design of benchmark task org/cad.".into();
        experience.trajectory.agent.tool_definitions = Some(vec![
            json!({"type": "function", "function": shell()}),
            json!({"type": "function", "function": {"name": "harness_vars_set"}}),
        ]);
        let notes = vec![verdict(&experience, Outcome::Pass, Strength::Executable)];
        corpus.insert(experience, notes).unwrap();
    }
    let projection = project(&view(), &corpus);
    assert_eq!(projection.records.len(), 2);
    let (messages, tools) = chat(&projection.records[0]);
    assert_eq!(messages[1].content, "Phase: design");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0]["function"]["name"], "shell");
}

#[test]
fn a_conversation_shown_images_is_left_out_unless_the_base_reads_them() {
    let calls = two_calls(
        json!([{"type": "text", "text": "Phase: survey. "},
               {"type": "text", "text": "Attached image: a.png"},
               {"type": "image", "image_url": PNG}]),
        json!("part.py"),
    );
    let corpus = conversation("visual", None, Some(&calls), true);

    let text_only = project(&view(), &corpus);
    assert!(text_only.records.is_empty());
    assert_eq!(text_only.count(Exclusion::ShownImages), 1);

    let with_images = project(&view().images(true), &corpus);
    assert_eq!(with_images.records.len(), 2);
    let (messages, _) = chat(&with_images.records[0]);
    // The turn the trajectory lacks is the request's, picture in place.
    assert_eq!(messages[1].role, "user");
    assert_eq!(messages[1].content, "Phase: survey. Attached image: a.png");
    assert_eq!(messages[1].images.len(), 1);
    assert_eq!(messages[1].images[0].url, PNG);
    assert_eq!(messages[1].images[0].at, messages[1].content.len());
    // Only the answer is supervised, and no supervised turn holds an image.
    let supervised: Vec<_> = messages.iter().filter(|m| m.train).collect();
    assert_eq!(supervised.len(), 1);
    assert!(supervised[0].images.is_empty());
    // The record is written as brain reads it.
    let written = serde_json::to_value(&messages[1]).unwrap();
    assert_eq!(written["content"][1]["image_url"]["url"], PNG);
}

#[test]
fn a_picture_a_tool_returned_is_in_the_tool_turn_of_the_later_records() {
    let calls = two_calls(
        json!("Phase: review"),
        json!([{"type": "text", "text": "Attached image: b.png"},
               {"type": "image", "image_url": PNG_TOO}]),
    );
    let corpus = conversation("reviewer", Some("Phase: review"), Some(&calls), true);
    let projection = project(&view().images(true), &corpus);
    assert_eq!(projection.records.len(), 2);
    let (first, _) = chat(&projection.records[0]);
    assert!(first.iter().all(|m| m.images.is_empty()));
    let (second, _) = chat(&projection.records[1]);
    let tool = second.iter().find(|m| m.role == "tool").unwrap();
    assert_eq!(tool.content, "Attached image: b.png");
    assert_eq!(tool.images[0].url, PNG_TOO);
}

#[test]
fn images_wanted_of_a_conversation_that_recorded_no_requests_is_counted() {
    let corpus = conversation("visual", Some("Phase: survey"), None, true);
    let projection = project(&view().images(true), &corpus);
    assert!(projection.records.is_empty());
    assert_eq!(projection.count(Exclusion::RequestsNotRecorded), 1);
}

#[test]
fn a_record_may_hold_at_most_the_images_brain_reads() {
    let many: Vec<Value> = (0..9)
        .map(|_| json!({"type": "image", "image_url": PNG}))
        .collect();
    let calls = two_calls(Value::Array(many), json!("part.py"));
    let corpus = conversation("visual", None, Some(&calls), true);
    let projection = project(&view().images(true), &corpus);
    assert!(projection.records.is_empty());
    assert_eq!(projection.count(Exclusion::TooManyImages), 2);
}

#[test]
fn a_conversation_whose_context_was_rewritten_between_requests_is_counted_compacted() {
    let mut calls = two_calls(json!("Phase: design"), json!("part.py"));
    calls[1].messages[1] = json!({"role": "user", "content": "A summary of what went before"});
    let corpus = conversation("designer", Some("Phase: design"), Some(&calls), false);
    let projection = project(&view(), &corpus);
    assert!(projection.records.is_empty());
    assert_eq!(projection.count(Exclusion::Compacted), 1);
}

#[test]
fn packed_the_conversation_is_one_record_with_every_answer_supervised() {
    let calls = two_calls(
        json!([{"type": "text", "text": "Phase: survey"}, {"type": "image", "image_url": PNG}]),
        json!("part.py"),
    );
    let corpus = conversation("visual", None, Some(&calls), true);
    let projection = project(&view().images(true).packed(true), &corpus);
    assert_eq!(projection.records.len(), 1);
    let (messages, _) = chat(&projection.records[0]);
    assert_eq!(messages.iter().filter(|m| m.train).count(), 2);
    assert_eq!(messages.iter().map(|m| m.images.len()).sum::<usize>(), 1);
}

#[test]
fn an_image_costs_the_rows_its_picture_takes_and_no_text() {
    let calls = two_calls(
        json!([{"type": "text", "text": "Phase"}, {"type": "image", "image_url": PNG}]),
        json!("part.py"),
    );
    let corpus = conversation("visual", None, Some(&calls), true);
    let projection = project(&view().images(true), &corpus);
    let (text, rows) = projection.records[0].body.text_and_image_rows();
    assert!(!text.contains("base64"), "{text}");
    // The stand-in PNG has no readable header: the most one image can take.
    assert_eq!(rows, splinter_data::images::MAX_ROWS_PER_IMAGE);
}

#[test]
fn a_png_costs_one_row_per_32_by_32_pixels_of_its_header_size() {
    // 640 x 480 pixels: 20 x 15 blocks.
    let url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAoAAAAHgCAIAAAAAAAAAAAAAAAAA";
    assert_eq!(splinter_data::images::image_rows(url), 300);
}
