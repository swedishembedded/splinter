// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements content-addressed experience stores that
// every training set is projected from, for its clients. If your team needs
// expertise in training-data provenance or durable learning state, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: the identity of what Splinter records is a function of its content
//! alone, so it survives every refactor of the code that builds it.
//!
//! Every release, dataset and lineage edge names experiences, tasks and
//! sources by these addresses. A change to the canonical form, to the
//! digest, or to how a field serializes silently orphans everything already
//! stored, so the exact bytes and addresses of one literal value of each
//! kind are pinned here, and the store must hand back what it was given
//! under the same address.

#![allow(clippy::unwrap_used)]

use std::fs;
use std::path::PathBuf;

use serde_json::json;
use splinter_record::annotation::{Annotation, AnnotationBody, Outcome, Producer, Strength};
use splinter_record::clock::FixedClock;
use splinter_record::experience::{
    Digest, Environment, Experience, Privileged, PrivilegedKind, Provenance, Span, Task,
};
use splinter_record::experiences::ExperienceStore;
use splinter_record::source::{CapturedSource, Origin, PartContent};
use splinter_record::workspace::Workspace;
use splinter_record::StateRoot;
use sven_sdk::atif::{AgentProfile, Trajectory};

const STAMP: &str = "2026-09-30T12:00:00.000Z";
const TEXT: &str = "The quick brown fox jumps over the lazy dog.";

fn clock() -> FixedClock {
    FixedClock::new(STAMP)
}

fn captured() -> CapturedSource {
    CapturedSource::new(
        Origin::Document {
            path: "/corpus/fox.md".into(),
        },
        vec![PartContent {
            name: "fox.md".into(),
            media_type: "text/markdown".into(),
            bytes: TEXT.as_bytes().to_vec(),
        }],
        &clock(),
    )
    .unwrap()
}

fn task() -> Task {
    let span = Span::new(Digest::of(TEXT.as_bytes()), 4, 19).unwrap();
    Task::new(
        "denoise",
        vec![span.clone()],
        Environment {
            kind: "closed-book".into(),
            spec: json!({"seed": 7, "ops": ["drop", "swap"]}),
            snapshot: None,
        },
        "Restore: quick fox brown",
        vec![Privileged {
            kind: PrivilegedKind::Reference,
            content: "quick brown fox".into(),
            span: Some(span),
        }],
    )
    .unwrap()
}

fn experience() -> Experience {
    let trajectory = Trajectory::new(
        "ATIF-v1.7",
        AgentProfile {
            name: "test".into(),
            version: "1".into(),
            model_name: Some("scripted".into()),
            tool_definitions: None,
            extra: None,
        },
    );
    Experience::new(
        task(),
        trajectory,
        Some("quick brown fox".into()),
        Provenance::new("scripted/solver-1", &clock()),
    )
    .unwrap()
}

fn annotation(experience: &Experience) -> Annotation {
    Annotation {
        experience: experience.id().unwrap(),
        producer: Producer {
            name: "test-grader".into(),
            version: "1".into(),
        },
        body: AnnotationBody::Verdict {
            outcome: Outcome::Pass,
            strength: Strength::Formal,
            evidence: json!({"matched": true}),
        },
    }
}

#[test]
fn a_source_has_the_address_its_origin_and_parts_give_it() {
    let source = captured().into_source();
    assert_eq!(
        serde_json::to_string(&source).unwrap(),
        GOLDEN_SOURCE_JSON,
        "the source's serialized form changed"
    );
    assert_eq!(
        source.id.as_str(),
        GOLDEN_SOURCE_ID,
        "the source's address changed"
    );
}

#[test]
fn a_task_has_the_address_its_content_gives_it() {
    let task = task();
    assert_eq!(
        serde_json::to_string(&task).unwrap(),
        GOLDEN_TASK_JSON,
        "the task's serialized form changed"
    );
    assert_eq!(
        task.task.id.as_str(),
        GOLDEN_TASK_ID,
        "the task's address changed"
    );
}

#[test]
fn an_experience_has_the_address_its_canonical_form_gives_it() {
    let experience = experience();
    assert_eq!(
        String::from_utf8(experience.canonical().unwrap()).unwrap(),
        GOLDEN_EXPERIENCE_CANONICAL,
        "the experience's canonical form changed"
    );
    assert_eq!(
        experience.id().unwrap().as_str(),
        GOLDEN_EXPERIENCE_ID,
        "the experience's address changed"
    );
}

#[test]
fn an_annotation_serializes_to_the_form_stored_evidence_is_read_from() {
    assert_eq!(
        serde_json::to_string(&annotation(&experience())).unwrap(),
        GOLDEN_ANNOTATION_JSON,
        "the annotation's serialized form changed"
    );
}

#[test]
fn the_store_returns_what_it_was_given_under_the_address_it_computed() {
    let root = std::env::temp_dir().join(format!("splinter-identity-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    let store = ExperienceStore::new(&Workspace::at(&StateRoot::new(PathBuf::from(&root))));

    let experience = experience();
    let id = store.put(&experience).unwrap();
    assert_eq!(id.as_str(), GOLDEN_EXPERIENCE_ID);
    assert_eq!(store.get(&id).unwrap(), experience);
    assert_eq!(store.get(&id).unwrap().id().unwrap(), id);

    let annotation = annotation(&experience);
    store.annotate(&annotation).unwrap();
    assert_eq!(
        store.annotations(&id).unwrap().annotations,
        vec![annotation]
    );

    let _ = fs::remove_dir_all(&root);
}

const GOLDEN_SOURCE_JSON: &str = r#"{"id":"blake3:f1cbb30671b34a73ee134343c2c634118470cc12e4519aa976de3fb31a9448f7","origin":{"kind":"document","path":"/corpus/fox.md"},"captured_at":"2026-09-30T12:00:00.000Z","parts":[{"name":"fox.md","media_type":"text/markdown","content":"blake3:4c9bd68d7f0baa2e167cef98295eb1ec99a3ec8f0656b33dbae943b387f31d5d","bytes":44}]}"#;
const GOLDEN_SOURCE_ID: &str =
    r#"blake3:f1cbb30671b34a73ee134343c2c634118470cc12e4519aa976de3fb31a9448f7"#;
const GOLDEN_TASK_JSON: &str = r#"{"task":{"id":"blake3:0aac5162f331491cdf33a25d378866990c213bce9540d01f674cad4c69426d44","kind":"denoise"},"evidence":[{"source":"blake3:4c9bd68d7f0baa2e167cef98295eb1ec99a3ec8f0656b33dbae943b387f31d5d","start":4,"end":19}],"environment":{"kind":"closed-book","spec":{"ops":["drop","swap"],"seed":7},"snapshot":null},"instruction":"Restore: quick fox brown","privileged":[{"kind":"reference","content":"quick brown fox","span":{"source":"blake3:4c9bd68d7f0baa2e167cef98295eb1ec99a3ec8f0656b33dbae943b387f31d5d","start":4,"end":19}}]}"#;
const GOLDEN_TASK_ID: &str =
    r#"blake3:0aac5162f331491cdf33a25d378866990c213bce9540d01f674cad4c69426d44"#;
const GOLDEN_EXPERIENCE_CANONICAL: &str = r#"{"environment":{"kind":"closed-book","snapshot":null,"spec":{"ops":["drop","swap"],"seed":7}},"evidence":[{"end":19,"source":"blake3:4c9bd68d7f0baa2e167cef98295eb1ec99a3ec8f0656b33dbae943b387f31d5d","start":4}],"final_output":"quick brown fox","instruction":"Restore: quick fox brown","privileged":[{"content":"quick brown fox","kind":"reference","span":{"end":19,"source":"blake3:4c9bd68d7f0baa2e167cef98295eb1ec99a3ec8f0656b33dbae943b387f31d5d","start":4}}],"provenance":{"created_at":"2026-09-30T12:00:00.000Z","generator":null,"policy":null,"prompt_digests":[],"solver":"scripted/solver-1"},"task":{"id":"blake3:0aac5162f331491cdf33a25d378866990c213bce9540d01f674cad4c69426d44","kind":"denoise"},"trajectory":{"agent":{"model_name":"scripted","name":"test","version":"1"},"schema_version":"ATIF-v1.7","steps":[]}}"#;
const GOLDEN_EXPERIENCE_ID: &str =
    r#"blake3:2ec390ecd67675a6fa552d2d84f6cf399f92fa7fd1abf1da632d903f8bc99a43"#;
const GOLDEN_ANNOTATION_JSON: &str = r#"{"experience":"blake3:2ec390ecd67675a6fa552d2d84f6cf399f92fa7fd1abf1da632d903f8bc99a43","producer":{"name":"test-grader","version":"1"},"body":{"type":"verdict","outcome":"pass","strength":"formal","evidence":{"matched":true}}}"#;
