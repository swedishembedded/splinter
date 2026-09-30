// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: the SFT-final view turns a passed experience into one chat record
//! the student can learn from - the instruction as the user turn, the final
//! output as the only supervised turn - and nothing the teacher alone saw
//! ever reaches the record or the dataset written from it.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::path::PathBuf;

use serde_json::json;
use splinter_store::annotation::{Annotation, AnnotationBody, Outcome, Producer, Strength};
use splinter_store::clock::FixedClock;
use splinter_store::experience::{
    Digest, Environment, Experience, Privileged, PrivilegedKind, Provenance, Span, Task,
};
use splinter_views::{write_dataset, Objective, SftFinal, View, ViewError};
use sven_sdk::atif::{AgentProfile, Trajectory};

const INSTRUCTION: &str = "Restore the passage: brown quick the fox";
const ANSWER: &str = "the quick brown fox";
const HINT: &str = "TEACHER-ONLY: the second word is quick";
const PASSAGE: &str = "TEACHER-ONLY: the quick brown fox jumps over the lazy dog";

fn experience() -> Experience {
    let span = Span::new(Digest::of(PASSAGE.as_bytes()), 14, 33).unwrap();
    let privileged = vec![
        Privileged {
            kind: PrivilegedKind::Hint,
            content: HINT.into(),
            span: None,
        },
        Privileged {
            kind: PrivilegedKind::Passage,
            content: PASSAGE.into(),
            span: Some(span.clone()),
        },
    ];
    let task = Task::new(
        "denoise",
        vec![span],
        Environment {
            kind: "closed-book".into(),
            spec: json!({}),
            snapshot: None,
        },
        INSTRUCTION,
        privileged,
    )
    .unwrap();
    let profile = AgentProfile {
        name: "t".into(),
        version: "1".into(),
        model_name: None,
        tool_definitions: None,
        extra: None,
    };
    Experience::new(
        task,
        Trajectory::new("ATIF-v1.7", profile),
        Some(ANSWER.into()),
        Provenance::new("scripted", &FixedClock::new("2026-09-30T00:00:00.000Z")),
    )
    .unwrap()
}

fn verdict(exp: &Experience, outcome: Outcome, strength: Strength) -> Annotation {
    Annotation {
        experience: exp.id().unwrap(),
        producer: Producer {
            name: "grader".into(),
            version: "1".into(),
        },
        body: AnnotationBody::Verdict {
            outcome,
            strength,
            evidence: json!({}),
        },
    }
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("splinter-views-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

#[test]
fn a_passed_experience_becomes_one_record_supervising_only_the_answer() {
    let exp = experience();
    let view = SftFinal::new(Strength::Formal);
    assert_eq!(view.objective(), Objective::Sft);
    let records = view
        .project(&exp, &[verdict(&exp, Outcome::Pass, Strength::Formal)])
        .unwrap();
    let [record] = &records[..] else {
        panic!("one record, got {records:?}")
    };
    let turns: Vec<(&str, &str, bool)> = record
        .messages
        .iter()
        .map(|m| (m.role.as_str(), m.content.as_str(), m.train))
        .collect();
    assert_eq!(
        turns,
        [("user", INSTRUCTION, false), ("assistant", ANSWER, true)]
    );
    assert_eq!(record.metadata.experience, exp.id().unwrap());

    let text = serde_json::to_string(record).unwrap();
    for secret in [HINT, PASSAGE, "TEACHER-ONLY"] {
        assert!(!text.contains(secret), "privileged content leaked: {text}");
    }
}

#[test]
fn only_a_pass_decided_at_the_minimum_strength_or_above_is_projected() {
    let exp = experience();
    let view = SftFinal::new(Strength::Formal);
    let project = |notes: &[Annotation]| view.project(&exp, notes).unwrap().len();
    assert_eq!(project(&[]), 0, "no verdict");
    assert_eq!(
        project(&[verdict(&exp, Outcome::Pass, Strength::Judged)]),
        0,
        "too weak"
    );
    assert_eq!(
        project(&[verdict(&exp, Outcome::Fail, Strength::Executable)]),
        0
    );
    assert_eq!(
        project(&[
            verdict(&exp, Outcome::Pass, Strength::Formal),
            verdict(&exp, Outcome::Fail, Strength::Executable),
        ]),
        0,
        "a stronger fail overrules the pass"
    );
    assert_eq!(
        project(&[verdict(&exp, Outcome::Pass, Strength::Executable)]),
        1
    );

    let mut other = experience();
    other.final_output = Some("something else".into());
    assert!(matches!(
        view.project(&exp, &[verdict(&other, Outcome::Pass, Strength::Formal)]),
        Err(ViewError::ForeignAnnotation { .. })
    ));
}

#[test]
fn the_dataset_is_generic_messages_v2_and_names_its_digest() {
    let exp = experience();
    let records = SftFinal::new(Strength::Formal)
        .project(&exp, &[verdict(&exp, Outcome::Pass, Strength::Formal)])
        .unwrap();
    let dir = scratch("dataset");
    let dataset = write_dataset(&dir.join("sft.jsonl"), &records).unwrap();
    let bytes = std::fs::read(&dataset.path).unwrap();
    assert_eq!(dataset.digest, Digest::of(&bytes));
    assert_eq!((dataset.records, dataset.trained_messages), (1, 1));

    let text = String::from_utf8(bytes).unwrap();
    let line: serde_json::Value = serde_json::from_str(text.trim_end()).unwrap();
    assert_eq!(
        line["messages"],
        json!([
            {"role": "user", "content": INSTRUCTION, "train": false},
            {"role": "assistant", "content": ANSWER, "train": true},
        ])
    );
    assert!(!text.contains("TEACHER-ONLY"));

    assert!(matches!(
        write_dataset(&dir.join("empty.jsonl"), &[]),
        Err(ViewError::Empty)
    ));
    assert!(!dir.join("empty.jsonl").exists());
    std::fs::remove_dir_all(&dir).unwrap();
}
