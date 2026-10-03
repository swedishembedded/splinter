// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: the views that read a trajectory step by step.
//!
//! * sft-step supervises every action of a verified trajectory - each tool
//!   call and the final answer - in the context the solver had, minus
//!   steps labelled bad; a failed trajectory contributes only steps
//!   labelled good.
//! * decision supervises, at the one state a passing attempt and a failed
//!   one of the same chain share and then part at, the passing attempt's
//!   action.
//! * outcome pairs a whole trajectory with its derived reward and never
//!   writes a reward nobody measured.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use common::*;
use serde_json::json;
use splinter_core::annotation::{Label, Outcome, RelationKind, Strength};
use splinter_core::experience::{Experience, PrivilegedKind, Task};
use splinter_lab::WireMessage;
use splinter_views::{
    write_dataset, Corpus, DecisionView, Exclusion, Objective, OutcomeView, Record, RecordBody,
    SftStep, View, ViewError, WriteOptions,
};

const INSTRUCTION: &str = "Compute one plus one with the tool and report it.";

fn arithmetic() -> Task {
    task(
        "arithmetic",
        INSTRUCTION,
        vec![privileged(PrivilegedKind::Reference, "2")],
    )
}

/// A solve that calls the tool once and answers; the solver's own prompt
/// carried a teacher-only hint the student must never see.
fn tool_solve(task: &Task, at: &str) -> Experience {
    experience(
        task,
        trajectory(vec![
            user(1, &format!("{INSTRUCTION}\n{SECRET}: use run_code")),
            call(2, "c1", "run_code", json!({"code": "1+1"}), "2"),
            say(3, "The answer is 2"),
        ]),
        Some("The answer is 2"),
        at,
    )
}

fn chat(record: &Record) -> &[WireMessage] {
    match &record.body {
        RecordBody::Chat { messages } => after_system(messages),
        other => panic!("a chat record, got {other:?}"),
    }
}

fn supervised(record: &Record) -> Vec<usize> {
    chat(record)
        .iter()
        .enumerate()
        .filter(|(_, m)| m.train)
        .map(|(i, _)| i)
        .collect()
}

#[test]
fn sft_step_supervises_each_action_in_the_context_the_solver_had() {
    let task = arithmetic();
    let passed = tool_solve(&task, "2026-09-30T01:00:00.000Z");
    let mut corpus = Corpus::new();
    corpus
        .insert(
            passed.clone(),
            vec![verdict(&passed, Outcome::Pass, Strength::Formal)],
        )
        .unwrap();
    let view = SftStep::new(Strength::Formal);
    assert_eq!(view.objective(), Objective::Sft);
    let projection = view.project(&corpus).unwrap();
    let [first, second] = &projection.records[..] else {
        panic!("one record per action, got {:?}", projection.records)
    };

    // The tool call, in its own context: the student's turn and nothing
    // else.
    let messages = chat(first);
    assert_eq!(messages.len(), 2);
    assert_eq!(
        (messages[0].role.as_str(), messages[0].content.as_str()),
        ("user", INSTRUCTION)
    );
    assert_eq!(messages[1].role, "assistant");
    let [tool_call] = &messages[1].tool_calls[..] else {
        panic!("one call, got {:?}", messages[1].tool_calls)
    };
    assert_eq!(tool_call.id.as_deref(), Some("c1"));
    assert_eq!(tool_call.function.name, "run_code");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&tool_call.function.arguments).unwrap(),
        json!({"code": "1+1"})
    );
    assert_eq!(supervised(first), [1]);

    // The final answer, after the call and its observation.
    let messages = chat(second);
    let roles: Vec<&str> = messages.iter().map(|m| m.role.as_str()).collect();
    assert_eq!(roles, ["user", "assistant", "tool", "assistant"]);
    assert_eq!(messages[2].tool_call_id.as_deref(), Some("c1"));
    assert_eq!(messages[2].content, "2");
    assert_eq!(messages[3].content, "The answer is 2");
    assert_eq!(supervised(second), [3], "only the action is supervised");
    for record in &projection.records {
        assert_eq!(record.metadata.experiences, [id(&passed)]);
        assert!(!serde_json::to_string(record).unwrap().contains(SECRET));
    }

    // Brain's parser accepts the tool-call form.
    let scratch = Scratch::new("sft-step");
    let dataset = write_dataset(
        &scratch.0.join("steps.jsonl"),
        &projection,
        WriteOptions::default(),
    )
    .unwrap();
    assert_eq!((dataset.records, dataset.trained_messages), (2, Some(2)));

    // A bad label drops that step; a failed trajectory yields only the
    // steps labelled good.
    let mut corpus = Corpus::new();
    corpus
        .insert(
            passed.clone(),
            vec![
                verdict(&passed, Outcome::Pass, Strength::Formal),
                label(&passed, 2, Label::Bad),
            ],
        )
        .unwrap();
    let failed = tool_solve(&task, "2026-09-30T02:00:00.000Z");
    corpus
        .insert(
            failed.clone(),
            vec![
                verdict(&failed, Outcome::Fail, Strength::Formal),
                label(&failed, 2, Label::Good),
            ],
        )
        .unwrap();
    let unlabelled = tool_solve(&task, "2026-09-30T03:00:00.000Z");
    corpus
        .insert(
            unlabelled.clone(),
            vec![verdict(&unlabelled, Outcome::Fail, Strength::Formal)],
        )
        .unwrap();
    let projection = view.project(&corpus).unwrap();
    let kept: Vec<(Vec<_>, usize)> = projection
        .records
        .iter()
        .map(|r| (r.metadata.experiences.clone(), chat(r).len()))
        .collect();
    assert_eq!(
        kept,
        [(vec![id(&passed)], 4), (vec![id(&failed)], 2)],
        "the passed final answer, and the failed solve's good tool call"
    );
    assert_eq!(projection.count(Exclusion::BadStep), 1);
}

#[test]
fn decision_supervises_the_passing_action_where_the_attempts_part() {
    let task = arithmetic();
    // Both call the tool the same way (under different call ids) and see
    // the same error; then the failed attempt gives up, the retry tries
    // again and passes.
    let failed = experience(
        &task,
        trajectory(vec![
            user(1, INSTRUCTION),
            call(2, "a1", "run_code", json!({"code": "1 +"}), "SyntaxError"),
            say(3, "It cannot be computed"),
        ]),
        Some("It cannot be computed"),
        "2026-09-30T01:00:00.000Z",
    );
    let retry = experience(
        &task,
        trajectory(vec![
            user(1, INSTRUCTION),
            call(2, "b1", "run_code", json!({"code": "1 +"}), "SyntaxError"),
            call(3, "b2", "run_code", json!({"code": "1 + 1"}), "2"),
            say(4, "The answer is 2"),
        ]),
        Some("The answer is 2"),
        "2026-09-30T02:00:00.000Z",
    );
    let also_passed = answered(&task, "2", "2026-09-30T03:00:00.000Z");

    let mut corpus = Corpus::new();
    corpus
        .insert(
            failed.clone(),
            vec![verdict(&failed, Outcome::Fail, Strength::Formal)],
        )
        .unwrap();
    corpus
        .insert(
            retry.clone(),
            vec![
                verdict(&retry, Outcome::Pass, Strength::Formal),
                relation(&retry, RelationKind::RetryOf, &failed),
            ],
        )
        .unwrap();
    corpus
        .insert(
            also_passed.clone(),
            vec![
                verdict(&also_passed, Outcome::Pass, Strength::Formal),
                relation(&also_passed, RelationKind::RevisionOf, &retry),
            ],
        )
        .unwrap();

    let projection = DecisionView::new(Strength::Formal)
        .project(&corpus)
        .unwrap();
    let [record] = &projection.records[..] else {
        panic!("one record, got {:?}", projection.records)
    };
    assert_eq!(record.metadata.experiences, [id(&retry), id(&failed)]);
    let messages = chat(record);
    let roles: Vec<&str> = messages.iter().map(|m| m.role.as_str()).collect();
    assert_eq!(roles, ["user", "assistant", "tool", "assistant"]);
    assert_eq!(messages[2].content, "SyntaxError");
    assert_eq!(supervised(record), [3]);
    let chosen = &messages[3].tool_calls[0].function;
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&chosen.arguments).unwrap(),
        json!({"code": "1 + 1"}),
        "the action on the passing path"
    );
    assert_eq!(
        projection.count(Exclusion::NoPreferredPath),
        1,
        "two passing attempts establish no preferred action"
    );
}

#[test]
fn outcome_pairs_a_trajectory_with_its_reward_and_skips_the_unmeasured() {
    let task = arithmetic();
    let passed = tool_solve(&task, "2026-09-30T01:00:00.000Z");
    let failed = answered(&task, "3", "2026-09-30T02:00:00.000Z");
    let ungraded = answered(&task, "4", "2026-09-30T03:00:00.000Z");
    let conflicted = answered(&task, "5", "2026-09-30T04:00:00.000Z");
    let mut corpus = Corpus::new();
    corpus
        .insert(
            passed.clone(),
            vec![verdict(&passed, Outcome::Pass, Strength::Formal)],
        )
        .unwrap();
    corpus
        .insert(
            failed.clone(),
            vec![verdict(&failed, Outcome::Fail, Strength::Formal)],
        )
        .unwrap();
    corpus.insert(ungraded, Vec::new()).unwrap();
    corpus
        .insert(
            conflicted.clone(),
            vec![
                verdict(&conflicted, Outcome::Pass, Strength::Formal),
                verdict(&conflicted, Outcome::Fail, Strength::Formal),
            ],
        )
        .unwrap();

    let view = OutcomeView::new(Strength::Formal);
    assert_eq!(view.objective(), Objective::Reward);
    let projection = view.project(&corpus).unwrap();
    let rewards: Vec<(Vec<_>, f64, usize)> = projection
        .records
        .iter()
        .map(|r| match &r.body {
            RecordBody::Rewarded { messages, reward } => (
                r.metadata.experiences.clone(),
                *reward,
                after_system(messages).len(),
            ),
            other => panic!("a rewarded record, got {other:?}"),
        })
        .collect();
    assert_eq!(
        rewards,
        [(vec![id(&passed)], 1.0, 4), (vec![id(&failed)], 0.0, 2)]
    );
    assert_eq!(projection.count(Exclusion::NoReward), 2);

    // Brain cannot train on rewards: the writer refuses, and the export
    // holds only measured rewards.
    let scratch = Scratch::new("outcome");
    let path = scratch.0.join("outcome.jsonl");
    assert!(matches!(
        write_dataset(&path, &projection, WriteOptions::default()),
        Err(ViewError::ObjectiveNotTrainable {
            objective: Objective::Reward
        })
    ));
    assert!(!path.exists());
    write_dataset(&path, &projection, WriteOptions { export_only: true }).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    let written: Vec<serde_json::Value> = text
        .lines()
        .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap()["reward"].clone())
        .collect();
    assert_eq!(written, [json!(1.0), json!(0.0)]);
}
