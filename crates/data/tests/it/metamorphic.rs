// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training-data pipelines whose datasets do
// not depend on what should not matter, for its clients. If your team needs
// expertise in validating fine-tuning data, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Spec, by metamorphic relation (plan 6F): what must not change a dataset
//! does not change it, and what must change it does.
//!
//! * The order experiences were added in changes no view's records or
//!   counts.
//! * An attempt nobody graded adds no record to any view and changes none
//!   of the others': it is counted as undecided, never as a failure.
//! * Two policies' attempts at one task are never one group: changing the
//!   model one attempt's verdict names splits the group it was in, so its reward is never
//!   compared with another policy's.
//! * A dataset written and read back is the projection's records, message
//!   for message, each supervised exactly as projected.
#![allow(clippy::unwrap_used)]

use crate::common;

use common::*;
use serde_json::{json, Value};
use splinter_core::annotation::{Annotation, AnnotationBody, Outcome, Strength};
use splinter_core::clock::FixedClock;
use splinter_core::experience::{Experience, Provenance, Task};
use splinter_data::{
    write_dataset, Corpus, Exclusion, PolicyGradient, Preference, Projection, RecordBody, SftFinal,
    SftStep, Unchecked, View,
};

/// One attempt: its task, its solver, whether it passed (`None`: ungraded),
/// when.
type Attempt<'a> = (&'a Task, &'a str, Option<bool>, &'a str);

fn attempt(
    task: &Task,
    solver: &str,
    passed: Option<bool>,
    at: &str,
) -> (Experience, Vec<Annotation>) {
    let answer = format!("answer {at}");
    let exp = Experience::new(
        task.clone(),
        trajectory(vec![user(1, &task.instruction), say(2, &answer)]),
        Some(answer),
        Provenance::new(solver, &FixedClock::new(at)),
    )
    .unwrap();
    // The verdict names the trial, the task version and the model that ran
    // it, as the bench ingestion records them: a policy is the model named.
    let notes = passed
        .map(|p| {
            let mut graded = verdict(
                &exp,
                if p { Outcome::Pass } else { Outcome::Fail },
                Strength::Executable,
            );
            if let AnnotationBody::Verdict { evidence, .. } = &mut graded.body {
                *evidence = json!({
                    "trial": at,
                    "task": {"id": task.instruction, "version": "1"},
                    "model": solver,
                    "metrics": {},
                });
            }
            vec![graded]
        })
        .unwrap_or_default();
    (exp, notes)
}

fn corpus(attempts: &[Attempt<'_>]) -> Corpus {
    let mut corpus = Corpus::new();
    for &(task, solver, passed, at) in attempts {
        let (exp, notes) = attempt(task, solver, passed, at);
        corpus.insert(exp, notes).unwrap();
    }
    corpus
}

fn views() -> Vec<Box<dyn View>> {
    vec![
        Box::new(SftFinal::new(Strength::Executable)),
        Box::new(SftStep::new(Strength::Executable)),
        Box::new(SftStep::new(Strength::Executable).packed(true)),
        Box::new(Preference::new(Strength::Executable)),
        Box::new(PolicyGradient::new(Strength::Executable)),
    ]
}

/// A projection's records (in an order of their own) and counts.
fn content(projection: &Projection) -> (Vec<String>, String) {
    let mut records: Vec<String> = projection
        .records
        .iter()
        .map(|r| format!("{r:?}"))
        .collect();
    records.sort();
    (records, format!("{:?}", projection.excluded))
}

fn tasks() -> [Task; 3] {
    [
        task("closed-qa", "What is two plus two?", vec![]),
        task("closed-qa", "Name a prime above ten.", vec![]),
        task("closed-qa", "Spell 'cat' backwards.", vec![]),
    ]
}

fn attempts(t: &[Task; 3]) -> Vec<Attempt<'_>> {
    vec![
        (&t[0], "solver-1", Some(true), "2026-11-24T01:00:00.000Z"),
        (&t[0], "solver-1", Some(false), "2026-11-24T01:00:01.000Z"),
        (&t[0], "solver-1", Some(true), "2026-11-24T01:00:02.000Z"),
        (&t[1], "solver-1", Some(false), "2026-11-24T01:00:03.000Z"),
        (&t[1], "solver-1", Some(true), "2026-11-24T01:00:04.000Z"),
        (&t[2], "solver-1", Some(false), "2026-11-24T01:00:05.000Z"),
    ]
}

#[test]
fn the_order_experiences_were_added_in_changes_no_view() {
    let t = tasks();
    let forward = attempts(&t);
    let mut backward = forward.clone();
    backward.reverse();
    let mut interleaved = forward.clone();
    interleaved.rotate_left(2);
    let (a, b, c) = (corpus(&forward), corpus(&backward), corpus(&interleaved));
    for view in views() {
        let want = content(&view.project(&a).unwrap());
        assert!(
            !want.0.is_empty(),
            "{} makes records of the corpus",
            view.name()
        );
        for other in [&b, &c] {
            assert_eq!(
                content(&view.project(other).unwrap()),
                want,
                "{}",
                view.name()
            );
        }
    }
}

#[test]
fn an_ungraded_attempt_adds_nothing_and_changes_nothing() {
    let t = tasks();
    let graded = attempts(&t);
    let mut with_ungraded = graded.clone();
    with_ungraded.push((&t[0], "solver-1", None, "2026-11-24T01:00:09.000Z"));
    with_ungraded.push((&t[2], "solver-1", None, "2026-11-24T01:00:10.000Z"));
    for view in views() {
        let before = view.project(&corpus(&graded)).unwrap();
        let after = view.project(&corpus(&with_ungraded)).unwrap();
        assert_eq!(
            content(&after).0,
            content(&before).0,
            "{}: records",
            view.name()
        );
        assert_eq!(
            after.count(Exclusion::Failed),
            before.count(Exclusion::Failed),
            "{}: an ungraded attempt counted as failed",
            view.name()
        );
        let added: usize =
            after.excluded.values().sum::<usize>() - before.excluded.values().sum::<usize>();
        assert_eq!(
            added,
            2,
            "{}: each ungraded attempt is counted once",
            view.name()
        );
    }
}

#[test]
fn two_policies_attempts_are_never_one_group() {
    let t = tasks();
    let one_policy = [
        (&t[0], "solver-1", Some(true), "2026-11-24T01:00:00.000Z"),
        (&t[0], "solver-1", Some(false), "2026-11-24T01:00:01.000Z"),
    ];
    let mut two_policies = one_policy;
    two_policies[1].1 = "solver-2";
    let view = PolicyGradient::new(Strength::Executable);
    let together = view.project(&corpus(&one_policy)).unwrap();
    assert_eq!(together.groups.len(), 1);
    assert_eq!(together.groups[0].trials.len(), 2);
    assert!(together.groups[0].signal);
    let apart = view.project(&corpus(&two_policies)).unwrap();
    assert_eq!(apart.groups.len(), 2, "{:?}", apart.groups);
    assert!(apart
        .groups
        .iter()
        .all(|g| g.trials.len() == 1 && !g.signal));
    assert!(
        apart.records.is_empty(),
        "no reward was compared across policies"
    );
}

#[test]
fn a_dataset_read_back_is_the_projection_message_for_message() {
    let t = tasks();
    let corpus = corpus(&attempts(&t));
    for view in [
        SftStep::new(Strength::Executable),
        SftStep::new(Strength::Executable).packed(true),
    ] {
        let projection = view.project(&corpus).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("train.jsonl");
        let dataset = write_dataset(&path, &projection, &Unchecked).unwrap();
        assert_eq!(dataset.records, projection.records.len());
        let lines: Vec<Value> = std::fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(lines.len(), projection.records.len());
        for (line, record) in lines.iter().zip(&projection.records) {
            let RecordBody::Chat { messages, .. } = &record.body else {
                panic!("a chat record");
            };
            let written = line["messages"].as_array().unwrap();
            assert_eq!(written.len(), messages.len());
            for (w, m) in written.iter().zip(messages) {
                assert_eq!(w["role"], json!(m.role));
                assert_eq!(w["content"], json!(m.content));
                assert_eq!(w["train"].as_bool().unwrap_or(false), m.train, "{w}");
            }
        }
    }
}
