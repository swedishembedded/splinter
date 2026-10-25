// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training data in which a teacher's
// corrections are supervised and the student's own prefix is only context,
// for its clients. If your team needs expertise in on-policy agent data then
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: a correction is one trajectory in the one trajectory format. The
//! student's recorded steps lead it and are context in every record; the
//! teacher's continuation is the only supervised material, and a
//! continuation nobody verified yields nothing.

#![allow(clippy::unwrap_used)]

mod common;

use common::*;
use serde_json::json;
use splinter_core::annotation::{Outcome, Strength};
use splinter_core::chat::WireMessage;
use splinter_core::experience::{Environment, Experience};
use splinter_data::{Corpus, RecordBody, SftStep, View};

/// The task the correction continues.
fn workspace_task() -> splinter_core::experience::Task {
    splinter_core::experience::Task::new(
        "workspace-change",
        vec![],
        Environment::new(Environment::WORKSPACE, json!({ "path": "repo" })),
        "Make the failing test pass.",
        vec![],
    )
    .unwrap()
}

/// A student attempt that failed, continued by a teacher: the student's
/// steps are marked as copied context, exactly as the correction stage
/// writes them.
fn corrected(task: &splinter_core::experience::Task) -> Experience {
    let mut student = call(
        2,
        "s1",
        "write_file",
        json!({"path": "src/lib.rs", "text": "wrong"}),
        "L1:red",
    );
    // The student's step is the recorded prefix: never supervised.
    student.is_copied_context = Some(true);
    experience(
        task,
        trajectory(vec![
            user(1, &task.instruction),
            student,
            call(
                3,
                "t1",
                "write_file",
                json!({"path": "src/lib.rs", "text": "right"}),
                "wrote 5 bytes",
            ),
            say(4, "The test now passes: the field is set."),
        ]),
        Some("diff --git a/src/lib.rs b/src/lib.rs"),
        "2026-10-25T01:00:00.000Z",
    )
}

fn corpus_of(experience: &Experience, notes: Vec<splinter_core::annotation::Annotation>) -> Corpus {
    let mut corpus = Corpus::new();
    corpus.insert(experience.clone(), notes).unwrap();
    corpus
}

/// Every message of `record` that is supervised, in order.
fn supervised(record: &RecordBody) -> Vec<WireMessage> {
    match record {
        RecordBody::Chat { messages, .. } => messages.iter().filter(|m| m.train).cloned().collect(),
        other => panic!("a chat record, got {other:?}"),
    }
}

#[test]
fn the_teacher_actions_are_supervised_and_the_students_prefix_is_context() {
    let task = workspace_task();
    let correction = corrected(&task);
    let corpus = corpus_of(
        &correction,
        vec![verdict(&correction, Outcome::Pass, Strength::Executable)],
    );

    let projection = SftStep::new(Strength::Executable).project(&corpus).unwrap();
    assert_eq!(
        projection.records.len(),
        2,
        "the teacher's call and its answer, not the student's step"
    );
    for record in &projection.records {
        let supervised = supervised(&record.body);
        assert_eq!(
            supervised.len(),
            1,
            "one action per record is supervised: {supervised:?}"
        );
        assert_ne!(
            supervised[0].content, "wrong",
            "a student's recorded action is never trained on"
        );
    }
    // The student's step is still there, as context: the teacher answered
    // the state the student actually reached.
    let context: Vec<String> = match &projection.records[0].body {
        RecordBody::Chat { messages, .. } => messages
            .iter()
            .filter(|m| !m.train)
            .map(|m| m.content.clone())
            .collect(),
        _ => unreachable!(),
    };
    assert!(
        context.iter().any(|text| text == "L1:red"),
        "the student's recorded observation is the context the teacher continued: {context:?}"
    );
}

#[test]
fn a_correction_nobody_verified_yields_no_record() {
    let task = workspace_task();
    let correction = corrected(&task);
    let corpus = corpus_of(
        &correction,
        vec![verdict(&correction, Outcome::Fail, Strength::Executable)],
    );
    let projection = SftStep::new(Strength::Executable).project(&corpus).unwrap();
    assert!(
        projection.records.is_empty(),
        "an unverified continuation is discarded, not softened: {:#?}",
        projection.records
    );
}

#[test]
fn the_prefix_rides_along_in_the_packed_record_without_being_learned() {
    let task = workspace_task();
    let correction = corrected(&task);
    let corpus = corpus_of(
        &correction,
        vec![verdict(&correction, Outcome::Pass, Strength::Executable)],
    );
    let projection = SftStep::new(Strength::Executable)
        .packed(true)
        .project(&corpus)
        .unwrap();
    assert_eq!(projection.records.len(), 1);
    let supervised = supervised(&projection.records[0].body);
    assert_eq!(
        supervised.len(),
        2,
        "the teacher's two actions: {supervised:?}"
    );
    assert!(
        supervised.iter().all(|m| m.content != "wrong"),
        "the student's action stays context"
    );
}
