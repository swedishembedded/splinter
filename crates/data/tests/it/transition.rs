// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for learning from an agent's own
// failed attempts without a teacher for its clients. If your team needs
// expertise in turning environment feedback into training data then you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: the transition view supervises what the environment answered, never
//! what a model wrote. Each attempt, passed or failed, is one record: the
//! conversation through its last observation, each observation supervised in
//! the context through the action it answers, every model turn context.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use crate::common;

use common::*;
use serde_json::json;
use splinter_core::annotation::{Outcome, Strength};
use splinter_core::chat::{LossWeight, WireMessage};
use splinter_core::experience::{Experience, Task};
use splinter_data::{Corpus, Exclusion, Projection, RecordBody, Transition, View};

const INSTRUCTION: &str = "Compute one plus one and then two plus two, and report both.";

fn arithmetic() -> Task {
    task("arithmetic", INSTRUCTION, Vec::new())
}

/// Two tool calls answered by `first` and `second`, then a final answer.
fn attempt(task: &Task, first: &str, second: &str, at: &str) -> Experience {
    experience(
        task,
        trajectory(vec![
            user(1, INSTRUCTION),
            call(2, "c1", "run_code", json!({"code": "1+1"}), first),
            call(3, "c2", "run_code", json!({"code": "2+2"}), second),
            say(4, "The answers are 2 and 4"),
        ]),
        Some("The answers are 2 and 4"),
        at,
    )
}

fn corpus(attempts: &[(Experience, Outcome)]) -> Corpus {
    let mut corpus = Corpus::new();
    for (experience, outcome) in attempts {
        corpus
            .insert(
                experience.clone(),
                vec![verdict(experience, *outcome, Strength::Executable)],
            )
            .unwrap();
    }
    corpus
}

fn records(projection: &Projection) -> Vec<&[WireMessage]> {
    projection
        .records
        .iter()
        .map(|r| match &r.body {
            RecordBody::Chat { messages, .. } => &messages[..],
            other => panic!("a chat record, got {other:?}"),
        })
        .collect()
}

fn supervised(record: &[WireMessage]) -> Vec<(&str, &str)> {
    record
        .iter()
        .filter(|m| m.train)
        .map(|m| (m.role.as_str(), m.content.as_str()))
        .collect()
}

#[test]
fn a_failed_attempt_supervises_each_observation_in_one_record_and_no_model_turn() {
    let task = arithmetic();
    let failed = attempt(&task, "2", "5", "2026-11-21T01:00:00.000Z");
    let projection = Transition::new()
        .project(&corpus(&[(failed, Outcome::Fail)]))
        .unwrap();
    let records = records(&projection);
    let [record] = &records[..] else {
        panic!("one record per attempt, got {}", records.len())
    };
    assert_eq!(supervised(record), [("tool", "2"), ("tool", "5")]);
    // The record ends at its last observation; the final answer is not in it.
    assert_eq!(record.last().unwrap().role, "tool");
    // Each observation follows the call it answers, and the calls are context.
    let first_tool = record.iter().position(|m| m.role == "tool").unwrap();
    assert_eq!(record[first_tool - 1].role, "assistant");
    assert!(!record[first_tool - 1].train);
    assert!(record.iter().all(|m| m.loss_weight.is_none()));
}

#[test]
fn the_loss_weight_is_carried_on_every_supervised_observation() {
    let task = arithmetic();
    let passed = attempt(&task, "2", "4", "2026-11-21T01:00:00.000Z");
    let weight = LossWeight::new(0.3).unwrap();
    let projection = Transition::new()
        .loss_weight(Some(weight))
        .project(&corpus(&[(passed, Outcome::Pass)]))
        .unwrap();
    let [record] = &records(&projection)[..] else {
        panic!("one record")
    };
    for message in record.iter() {
        assert_eq!(message.loss_weight, message.train.then_some(weight));
    }
}

#[test]
fn a_transition_seen_too_often_is_context_not_a_target() {
    // Three attempts make the same first call with the same answer: with a
    // cap of two, the third's first observation is context only.
    let task = arithmetic();
    let attempts: Vec<(Experience, Outcome)> = ["01", "02", "03"]
        .iter()
        .enumerate()
        .map(|(n, minute)| {
            let second = ["4", "5", "6"][n];
            (
                attempt(
                    &task,
                    "2",
                    second,
                    &format!("2026-11-21T01:{minute}:00.000Z"),
                ),
                Outcome::Fail,
            )
        })
        .collect();
    let projection = Transition::new()
        .per_transition_cap(2)
        .project(&corpus(&attempts))
        .unwrap();
    let targets: Vec<Vec<(&str, &str)>> =
        records(&projection).iter().map(|r| supervised(r)).collect();
    assert_eq!(targets.iter().flatten().filter(|t| t.1 == "2").count(), 2);
    assert_eq!(targets.iter().flatten().count(), 5);
    assert_eq!(
        projection.excluded.get(&Exclusion::RepeatedTransition),
        Some(&1)
    );
}

#[test]
fn an_attempt_made_open_book_is_refused() {
    let task = arithmetic();
    let mut helped = attempt(&task, "2", "4", "2026-11-21T01:00:00.000Z");
    helped.provenance.teacher = true;
    let projection = Transition::new()
        .project(&corpus(&[(helped, Outcome::Pass)]))
        .unwrap();
    assert!(projection.records.is_empty());
    assert_eq!(projection.excluded.get(&Exclusion::OpenBook), Some(&1));
}

#[test]
fn shuffled_observations_keep_the_records_and_the_targets_but_not_their_pairing() {
    let task = arithmetic();
    let attempts: Vec<(Experience, Outcome)> = (0..6)
        .map(|n| {
            (
                attempt(
                    &task,
                    &format!("first {n}"),
                    &format!("second {n}"),
                    &format!("2026-11-21T01:0{n}:00.000Z"),
                ),
                Outcome::Fail,
            )
        })
        .collect();
    let corpus = corpus(&attempts);
    let plain = Transition::new().project(&corpus).unwrap();
    let shuffled = Transition::new()
        .shuffle_observations(Some(7))
        .project(&corpus)
        .unwrap();
    let (plain, shuffled) = (records(&plain), records(&shuffled));
    assert_eq!(plain.len(), shuffled.len());
    let mut before: Vec<&str> = plain
        .iter()
        .flat_map(|r| supervised(r))
        .map(|t| t.1)
        .collect();
    let mut after: Vec<&str> = shuffled
        .iter()
        .flat_map(|r| supervised(r))
        .map(|t| t.1)
        .collect();
    let moved = before.iter().zip(&after).filter(|(a, b)| a != b).count();
    assert_eq!(moved, before.len(), "every observation moves");
    before.sort_unstable();
    after.sort_unstable();
    assert_eq!(before, after, "the same targets, redistributed");
    // Everything but the supervised observations is unchanged.
    for (p, s) in plain.iter().zip(&shuffled) {
        let context = |r: &[WireMessage]| -> Vec<WireMessage> {
            r.iter().filter(|m| !m.train).cloned().collect()
        };
        assert_eq!(context(p), context(s));
    }
}
