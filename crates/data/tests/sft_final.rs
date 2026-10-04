// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: the SFT-final view turns a passed experience into one chat record
//! the student can learn from - the system prompt every solve runs under,
//! the instruction as the user turn, the final output as the only
//! supervised turn - and nothing the teacher alone saw ever reaches the
//! record or the dataset written from it.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::path::PathBuf;

use atif::{AgentProfile, StepOrigin, TraceStep, Trajectory};
use serde_json::json;
use splinter_core::annotation::{Annotation, AnnotationBody, Outcome, Producer, Strength};
use splinter_core::clock::FixedClock;
use splinter_core::experience::{
    Digest, Environment, Experience, Privileged, PrivilegedKind, Provenance, Span, Task,
};
use splinter_core::prompt::SYSTEM_PROMPT;
use splinter_data::{
    write_dataset, Corpus, Format, Objective, RecordBody, SftFinal, Unchecked, View, ViewError,
};

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

/// The records `view` projects from `exp` graded by `notes`.
fn project(view: &SftFinal, exp: &Experience, notes: &[Annotation]) -> Vec<splinter_data::Record> {
    let mut corpus = Corpus::new();
    corpus.insert(exp.clone(), notes.to_vec()).unwrap();
    view.project(&corpus).unwrap().records
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("splinter-data-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

#[test]
fn a_passed_experience_becomes_one_record_supervising_only_the_answer() {
    let exp = experience();
    let view = SftFinal::new(Strength::Formal);
    assert_eq!(view.objective(), Objective::Sft);
    let records = project(
        &view,
        &exp,
        &[verdict(&exp, Outcome::Pass, Strength::Formal)],
    );
    let [record] = &records[..] else {
        panic!("one record, got {records:?}")
    };
    let RecordBody::Chat { messages } = &record.body else {
        panic!("a chat record, got {record:?}")
    };
    let turns: Vec<(&str, &str, bool)> = messages
        .iter()
        .map(|m| (m.role.as_str(), m.content.as_str(), m.train))
        .collect();
    assert_eq!(
        turns,
        [
            ("system", SYSTEM_PROMPT, false),
            ("user", INSTRUCTION, false),
            ("assistant", ANSWER, true)
        ]
    );
    assert_eq!(record.metadata.experiences, [exp.id().unwrap()]);

    let text = serde_json::to_string(record).unwrap();
    for secret in [HINT, PASSAGE, "TEACHER-ONLY"] {
        assert!(!text.contains(secret), "privileged content leaked: {text}");
    }
}

#[test]
fn only_a_pass_decided_at_the_minimum_strength_or_above_is_projected() {
    let exp = experience();
    let view = SftFinal::new(Strength::Formal);
    let count = |notes: &[Annotation]| project(&view, &exp, notes).len();
    assert_eq!(count(&[]), 0, "no verdict");
    assert_eq!(
        count(&[verdict(&exp, Outcome::Pass, Strength::Judged)]),
        0,
        "too weak"
    );
    assert_eq!(
        count(&[verdict(&exp, Outcome::Fail, Strength::Executable)]),
        0
    );
    assert_eq!(
        count(&[
            verdict(&exp, Outcome::Pass, Strength::Formal),
            verdict(&exp, Outcome::Fail, Strength::Executable),
        ]),
        0,
        "a stronger fail overrules the pass"
    );
    assert_eq!(
        count(&[verdict(&exp, Outcome::Pass, Strength::Executable)]),
        1
    );

    let mut other = experience();
    other.final_output = Some("something else".into());
    assert!(matches!(
        Corpus::new().insert(
            exp.clone(),
            vec![verdict(&other, Outcome::Pass, Strength::Formal)]
        ),
        Err(ViewError::ForeignAnnotation { .. })
    ));
}

#[test]
fn the_dataset_is_generic_messages_v2_and_names_its_digest() {
    let exp = experience();
    let mut corpus = Corpus::new();
    corpus
        .insert(
            exp.clone(),
            vec![verdict(&exp, Outcome::Pass, Strength::Formal)],
        )
        .unwrap();
    let projection = SftFinal::new(Strength::Formal).project(&corpus).unwrap();
    let dir = scratch("dataset");
    let dataset = write_dataset(&dir.join("sft.jsonl"), &projection, &Unchecked).unwrap();
    let bytes = std::fs::read(&dataset.path).unwrap();
    assert_eq!(dataset.digest, Digest::of(&bytes));
    assert_eq!(dataset.format, Format::GenericMessagesV2);
    assert_eq!((dataset.records, dataset.trained_messages), (1, Some(1)));

    let text = String::from_utf8(bytes).unwrap();
    let line: serde_json::Value = serde_json::from_str(text.trim_end()).unwrap();
    assert_eq!(
        line["messages"],
        json!([
            {"role": "system", "content": SYSTEM_PROMPT, "train": false},
            {"role": "user", "content": INSTRUCTION, "train": false},
            {"role": "assistant", "content": ANSWER, "train": true},
        ])
    );
    assert!(!text.contains("TEACHER-ONLY"));

    let mut empty = projection.clone();
    empty.records.clear();
    assert!(matches!(
        write_dataset(&dir.join("empty.jsonl"), &empty, &Unchecked),
        Err(ViewError::Empty)
    ));
    assert!(!dir.join("empty.jsonl").exists());
    std::fs::remove_dir_all(&dir).unwrap();
}

/// A dialogue of three exchanges: the first user step is what the teacher
/// was prompted with, the later ones are the other speaker's own words.
fn dialogue() -> Experience {
    let task = Task::new(
        "converse",
        Vec::new(),
        Environment {
            kind: "closed-book".into(),
            spec: json!({}),
            snapshot: None,
        },
        INSTRUCTION,
        Vec::new(),
    )
    .unwrap();
    let profile = AgentProfile {
        name: "t".into(),
        version: "1".into(),
        model_name: None,
        tool_definitions: None,
        extra: None,
    };
    let mut trajectory = Trajectory::new("ATIF-v1.7", profile);
    let step = |id: u64, origin: StepOrigin, text: &str| TraceStep::new(id, origin, text);
    trajectory.steps = vec![
        step(1, StepOrigin::User, &format!("{PASSAGE}\n\n{INSTRUCTION}")),
        step(2, StepOrigin::Agent, "First reply."),
        step(3, StepOrigin::User, "And then?"),
        step(4, StepOrigin::Agent, "Second reply."),
        step(5, StepOrigin::User, "Why so?"),
        step(6, StepOrigin::Agent, "Third reply."),
    ];
    Experience::new(
        task,
        trajectory,
        Some("First reply.\n\nSecond reply.\n\nThird reply.".into()),
        Provenance::new("scripted", &FixedClock::new("2026-09-30T00:00:00.000Z")),
    )
    .unwrap()
}

#[test]
fn a_dialogue_is_one_record_supervising_every_reply_and_none_of_the_teachers_context() {
    let exp = dialogue();
    let records = project(
        &SftFinal::new(Strength::Judged),
        &exp,
        &[verdict(&exp, Outcome::Pass, Strength::Judged)],
    );
    let [record] = &records[..] else {
        panic!("one record, got {records:?}")
    };
    let RecordBody::Chat { messages } = &record.body else {
        panic!("a chat record, got {record:?}")
    };
    let turns: Vec<(&str, &str, bool)> = messages
        .iter()
        .map(|m| (m.role.as_str(), m.content.as_str(), m.train))
        .collect();
    assert_eq!(
        turns,
        [
            ("system", SYSTEM_PROMPT, false),
            ("user", INSTRUCTION, false),
            ("assistant", "First reply.", true),
            ("user", "And then?", false),
            ("assistant", "Second reply.", true),
            ("user", "Why so?", false),
            ("assistant", "Third reply.", true),
        ]
    );
    let text = serde_json::to_string(record).unwrap();
    assert!(!text.contains("TEACHER-ONLY"), "{text}");
}

#[test]
fn a_projection_opens_its_records_with_a_persons_prompt_and_the_manifest_says_so() {
    let exp = experience();
    let mut corpus = Corpus::new();
    corpus
        .insert(
            exp.clone(),
            vec![verdict(&exp, Outcome::Pass, Strength::Formal)],
        )
        .unwrap();
    let projection = SftFinal::new(Strength::Formal).project(&corpus).unwrap();
    let dir = scratch("persona");
    let manifest_of = |path: &std::path::Path| -> serde_json::Value {
        let manifest = splinter_data::manifest_path(path);
        serde_json::from_str(&std::fs::read_to_string(manifest).unwrap()).unwrap()
    };

    // The default prompt is the default: the manifest names none.
    let plain = write_dataset(&dir.join("plain.jsonl"), &projection, &Unchecked).unwrap();
    assert!(manifest_of(&plain.path).get("system_prompt").is_none());

    // A person's prompt replaces the first turn of every record, and the
    // manifest records it, so what the dataset trains is what it names.
    const PERSONA: &str = "You are a surveyor. Answer as one would.";
    let persona = projection.clone().with_system_prompt(PERSONA);
    let written = write_dataset(&dir.join("persona.jsonl"), &persona, &Unchecked).unwrap();
    let text = std::fs::read_to_string(&written.path).unwrap();
    let line: serde_json::Value = serde_json::from_str(text.trim_end()).unwrap();
    assert_eq!(
        line["messages"][0],
        json!({"role": "system", "content": PERSONA, "train": false})
    );
    assert_eq!(line["messages"][1]["content"], INSTRUCTION);
    assert_eq!(manifest_of(&written.path)["system_prompt"], PERSONA);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_prompt_is_inserted_where_a_conversation_has_no_system_turn() {
    use splinter_core::chat::WireMessage;
    let turn = |role: &str, content: &str, train: bool| WireMessage {
        role: role.into(),
        content: content.into(),
        tool_calls: Vec::new(),
        tool_call_id: None,
        train,
    };
    let exp = experience();
    let mut corpus = Corpus::new();
    corpus
        .insert(
            exp.clone(),
            vec![verdict(&exp, Outcome::Pass, Strength::Formal)],
        )
        .unwrap();
    let mut projection = SftFinal::new(Strength::Formal).project(&corpus).unwrap();
    if let RecordBody::Chat { messages } = &mut projection.records[0].body {
        messages.remove(0);
        assert_eq!(messages[0].role, "user");
    }
    let projection = projection.with_system_prompt("You are a surveyor.");
    let RecordBody::Chat { messages } = &projection.records[0].body else {
        panic!("a chat record");
    };
    let roles: Vec<&str> = messages.iter().map(|m| m.role.as_str()).collect();
    assert_eq!(roles, ["system", "user", "assistant"]);
    assert_eq!(messages[0], turn("system", "You are a surveyor.", false));
}
