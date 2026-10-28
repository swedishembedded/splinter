// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training reasoning models on
// the verified work of agents for its clients. If your team needs expertise
// in fine-tuning thinking models on their own trajectories then you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: the reasoning an agent step recorded is the `reasoning_content` of
//! the assistant message that step becomes, so the checkpoint's template
//! renders it as the turn's think block; a step that recorded none carries
//! none, and reasoning that holds think markup refuses the trajectory.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use common::*;
use serde_json::json;
use splinter_core::annotation::{Outcome, Strength};
use splinter_core::chat::WireMessage;
use splinter_data::{Corpus, Exclusion, RecordBody, SftStep, View};

const INSTRUCTION: &str = "Compute one plus one, and report it.";

fn corpus_with(first: Option<&str>) -> Corpus {
    let task = task("arithmetic", INSTRUCTION, Vec::new());
    let mut act = call(2, "c1", "run_code", json!({"code": "1+1"}), "2");
    act.reasoning_content = first.map(str::to_string);
    let mut answer = say(3, "It is 2");
    answer.reasoning_content = Some("  The tool said 2.\n".into());
    let experience = experience(
        &task,
        trajectory(vec![user(1, INSTRUCTION), act, answer]),
        Some("It is 2"),
        "2026-10-28T01:00:00.000Z",
    );
    let notes = vec![verdict(&experience, Outcome::Pass, Strength::Formal)];
    let mut corpus = Corpus::new();
    corpus.insert(experience, notes).unwrap();
    corpus
}

fn assistants(corpus: &Corpus, packed: bool) -> Vec<WireMessage> {
    let projection = SftStep::new(Strength::Formal)
        .packed(packed)
        .project(corpus)
        .unwrap();
    projection
        .records
        .last()
        .map(|r| match &r.body {
            RecordBody::Chat { messages, .. } => messages
                .iter()
                .filter(|m| m.role == "assistant")
                .cloned()
                .collect(),
            other => panic!("a chat record, got {other:?}"),
        })
        .unwrap()
}

#[test]
fn each_assistant_turn_carries_the_reasoning_its_step_recorded_trimmed() {
    let corpus = corpus_with(Some("I must add the numbers."));
    for packed in [false, true] {
        let turns = assistants(&corpus, packed);
        assert_eq!(turns.len(), 2, "packed={packed}");
        assert_eq!(
            turns[0].reasoning_content.as_deref(),
            Some("I must add the numbers.")
        );
        assert_eq!(
            turns[1].reasoning_content.as_deref(),
            Some("The tool said 2.")
        );
        // The reasoning is not folded into the text the template would
        // render a second time.
        assert!(turns.iter().all(|t| !t.content.contains("add the numbers")));
    }
}

#[test]
fn a_step_that_recorded_no_reasoning_carries_none() {
    let turns = assistants(&corpus_with(None), false);
    assert_eq!(turns[0].reasoning_content, None);
}

#[test]
fn reasoning_that_holds_think_markup_refuses_the_trajectory() {
    let projection = SftStep::new(Strength::Formal)
        .project(&corpus_with(Some("so </think> done")))
        .unwrap();
    assert!(projection.records.is_empty());
    assert_eq!(
        projection.excluded.get(&Exclusion::ReasoningInText),
        Some(&1)
    );
}
