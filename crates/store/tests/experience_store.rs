// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements content-addressed experience stores that
// every training set is projected from, for its clients. If your team needs
// expertise in training-data provenance or durable learning state, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: an experience is immutable and content-addressed, an annotation is
//! append-only and never rewrites it, and reward is derived from the
//! annotations rather than stored.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};

use serde_json::json;
use splinter_store::annotation::{
    reward, Annotation, AnnotationBody, Outcome, Producer, RelationKind, Strength,
};
use splinter_store::clock::FixedClock;
use splinter_store::experience::{
    Digest, Environment, Experience, Privileged, PrivilegedKind, Provenance, Span, Task,
};
use splinter_store::experiences::{ExperienceSet, ExperienceStore, StoreError};
use splinter_store::StateRoot;
use sven_sdk::atif::{AgentProfile, Trajectory};

/// A fresh state root per test, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        static N: AtomicU32 = AtomicU32::new(0);
        let path = std::env::temp_dir().join(format!(
            "splinter-exp-{name}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&path);
        Self(path)
    }
    fn store(&self) -> ExperienceStore {
        ExperienceStore::open(&StateRoot::new(&self.0))
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

const SOURCE: &str = "The quick brown fox jumps over the lazy dog.";

fn source_digest() -> Digest {
    Digest::of(SOURCE.as_bytes())
}

fn task(spec: serde_json::Value) -> Task {
    let span = Span::new(source_digest(), 4, 19).unwrap();
    Task::new(
        "denoise",
        vec![span.clone()],
        Environment {
            kind: "closed-book".into(),
            spec,
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

fn experience(output: &str) -> Experience {
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
        task(json!({"seed": 7, "ops": ["drop", "swap"]})),
        trajectory,
        Some(output.into()),
        Provenance::new(
            "scripted/solver-1",
            &FixedClock::new("2026-09-30T12:00:00.000Z"),
        ),
    )
    .unwrap()
}

fn verdict(
    id: &splinter_store::experience::ExperienceId,
    outcome: Outcome,
    strength: Strength,
) -> Annotation {
    Annotation {
        experience: id.clone(),
        producer: Producer {
            name: "test-grader".into(),
            version: "1".into(),
        },
        body: AnnotationBody::Verdict {
            outcome,
            strength,
            evidence: json!({}),
        },
    }
}

/// The id is the digest of a canonical form: independent of the order keys
/// were inserted in, and different for different content.
#[test]
fn the_id_is_a_canonical_content_address() {
    let a = experience("quick brown fox");
    let mut b = a.clone();
    b.environment.spec = json!({"ops": ["drop", "swap"], "seed": 7});
    b.task = task(json!({"ops": ["drop", "swap"], "seed": 7})).task;
    assert_eq!(a.id().unwrap(), b.id().unwrap(), "key order is not content");
    let id = a.id().unwrap();
    assert!(
        id.as_str().starts_with("sha256:") && id.as_str().len() == 71,
        "{id}"
    );
    assert_ne!(id, experience("quick fox").id().unwrap());
}

/// An environment's snapshot is the digest of everything that determines
/// its behaviour: any change to its spec changes it, and a record whose
/// snapshot does not match its spec is refused.
#[test]
fn an_environment_snapshot_addresses_its_kind_and_spec() {
    let closed = Environment::closed_book();
    assert_eq!(closed.kind, Environment::CLOSED_BOOK);
    assert_eq!(
        closed.snapshot,
        Some(Environment::snapshot_of("closed-book", &json!({})))
    );
    let limits = |cpu: u64| Environment::new("runtime:python3", json!({"limits": {"cpu": cpu}}));
    assert_eq!(limits(5).snapshot, limits(5).snapshot);
    assert_ne!(limits(5).snapshot, limits(6).snapshot);
    assert_ne!(limits(5).snapshot, closed.snapshot);

    let mut forged = limits(5);
    forged.spec = json!({"limits": {"cpu": 600}});
    let refused = Task::new(
        "denoise",
        vec![],
        forged,
        "Restore: quick fox brown",
        vec![],
    );
    assert!(
        matches!(
            refused,
            Err(splinter_store::experience::ExperienceError::EnvironmentSnapshot { .. })
        ),
        "{refused:?}"
    );
}

/// Put is write-once: the same content twice is one object and one id, and
/// get returns exactly what was put.
#[test]
fn a_critique_added_for_a_retry_leaves_the_task_the_same_task() {
    let original = task(json!({}));
    let critique = Privileged {
        kind: PrivilegedKind::Critique,
        content: "the words are out of order".into(),
        span: None,
    };
    let retried = original.with_privileged(critique).unwrap();
    assert_ne!(
        retried.task.id, original.task.id,
        "a new task, addressed anew"
    );
    assert_eq!(retried.instruction, original.instruction);
    assert!(retried.same_apart_from_critiques(&original));
    assert!(original.same_apart_from_critiques(&retried));

    let hinted = original
        .with_privileged(Privileged {
            kind: PrivilegedKind::Hint,
            content: "start with quick".into(),
            span: None,
        })
        .unwrap();
    assert!(
        !hinted.same_apart_from_critiques(&original),
        "any other teacher-only item makes it another task"
    );
}

#[test]
fn put_is_write_once_and_get_round_trips() {
    let scratch = Scratch::new("put");
    let store = scratch.store();
    let exp = experience("quick brown fox");
    let first = store.put(&exp).unwrap();
    let object = scratch
        .0
        .join("experiences/objects")
        .join(format!("{}.json", first.hex()));
    let written = fs::read(&object).unwrap();
    let second = store.put(&exp).unwrap();
    assert_eq!(first, second);
    assert_eq!(fs::read(&object).unwrap(), written, "never rewritten");
    assert_eq!(fs::read_dir(object.parent().unwrap()).unwrap().count(), 1);
    assert_eq!(store.get(&first).unwrap(), exp);
}

/// A stored object whose bytes no longer hash to its address is an error on
/// read, and a later put of the true content does not paper over it.
#[test]
fn corruption_is_an_error_not_silently_accepted() {
    let scratch = Scratch::new("corrupt");
    let store = scratch.store();
    let id = store.put(&experience("quick brown fox")).unwrap();
    let object = scratch
        .0
        .join("experiences/objects")
        .join(format!("{}.json", id.hex()));
    let text = fs::read_to_string(&object)
        .unwrap()
        .replace("quick brown fox", "quick brown cat");
    fs::write(&object, text).unwrap();
    assert!(matches!(store.get(&id), Err(StoreError::Corrupt { .. })));
    assert!(matches!(
        store.put(&experience("quick brown fox")),
        Err(StoreError::Corrupt { .. })
    ));
}

/// Input is validated at the boundary: a span that ends before it starts
/// and a malformed digest are refused.
#[test]
fn malformed_input_is_refused() {
    assert!(Span::new(source_digest(), 9, 3).is_err());
    assert!(Digest::parse("sha256:xyz").is_err());
    assert!(Digest::parse("md5:d41d8cd98f00b204e9800998ecf8427e").is_err());
    assert!(serde_json::from_value::<Digest>(json!("sha256:ABC")).is_err());

    let scratch = Scratch::new("invalid");
    let mut exp = experience("x");
    exp.evidence[0].start = 30;
    assert!(matches!(
        scratch.store().put(&exp),
        Err(StoreError::Invalid(_))
    ));
    let mut exp = experience("x");
    exp.instruction = "a different instruction".into();
    assert!(
        matches!(scratch.store().put(&exp), Err(StoreError::Invalid(_))),
        "the task reference must address the task it names"
    );
}

/// Annotations append in order, survive a fresh store handle, and are
/// refused for an experience the store does not hold.
#[test]
fn annotations_append_in_order_and_are_refused_for_unknown_experiences() {
    let scratch = Scratch::new("annotate");
    let store = scratch.store();
    let id = store.put(&experience("quick brown fox")).unwrap();
    let other = store.put(&experience("quick fox")).unwrap();

    let unknown = experience("never stored").id().unwrap();
    assert!(matches!(
        store.annotate(&verdict(&unknown, Outcome::Pass, Strength::Formal)),
        Err(StoreError::UnknownExperience(_))
    ));

    let notes = [
        verdict(&id, Outcome::Fail, Strength::Judged),
        Annotation {
            experience: id.clone(),
            producer: Producer {
                name: "labeler".into(),
                version: "2".into(),
            },
            body: AnnotationBody::Relation {
                kind: RelationKind::RetryOf,
                other: other.clone(),
            },
        },
        verdict(&id, Outcome::Pass, Strength::Formal),
    ];
    for note in &notes {
        store.annotate(note).unwrap();
    }
    let log = scratch.store().annotations(&id).unwrap();
    assert_eq!(log.annotations, notes);
    assert_eq!(log.unreadable, 0);
    assert!(scratch
        .store()
        .annotations(&other)
        .unwrap()
        .annotations
        .is_empty());
    assert_eq!(
        store.get(&id).unwrap(),
        experience("quick brown fox"),
        "never rewritten"
    );
}

/// A crash mid-append leaves a torn final line: it is skipped and counted,
/// never a panic, and a later append is not glued onto it.
#[test]
fn a_torn_final_line_is_counted_not_fatal() {
    let scratch = Scratch::new("torn");
    let store = scratch.store();
    let id = store.put(&experience("quick brown fox")).unwrap();
    store
        .annotate(&verdict(&id, Outcome::Pass, Strength::Formal))
        .unwrap();
    let log_path = scratch
        .0
        .join("experiences/annotations")
        .join(format!("{}.jsonl", id.hex()));
    let mut text = fs::read_to_string(&log_path).unwrap();
    text.push_str("{\"experience\":\"sha256:");
    fs::write(&log_path, text).unwrap();

    let log = store.annotations(&id).unwrap();
    assert_eq!(log.annotations.len(), 1);
    assert_eq!(log.unreadable, 1);

    store
        .annotate(&verdict(&id, Outcome::Fail, Strength::Executable))
        .unwrap();
    let log = store.annotations(&id).unwrap();
    assert_eq!(
        log.annotations.len(),
        2,
        "the next append stands on its own line"
    );
    assert_eq!(log.unreadable, 1);
}

/// Reward is derived: the strongest non-abstaining verdicts decide, a
/// conflict at that strength decides nothing, and no verdict is no reward.
#[test]
fn reward_is_derived_from_the_strongest_verdicts() {
    let id = experience("x").id().unwrap();
    let v = |o, s| verdict(&id, o, s);
    assert_eq!(reward(&[]), None);
    assert_eq!(reward(&[v(Outcome::Abstain, Strength::Executable)]), None);
    assert_eq!(reward(&[v(Outcome::Pass, Strength::Judged)]), Some(1.0));
    assert_eq!(
        reward(&[
            v(Outcome::Pass, Strength::Judged),
            v(Outcome::Fail, Strength::Formal),
            v(Outcome::Abstain, Strength::Executable),
        ]),
        Some(0.0),
        "a formal fail outranks a judged pass; an abstention decides nothing"
    );
    assert_eq!(
        reward(&[
            v(Outcome::Pass, Strength::Consistency),
            v(Outcome::Fail, Strength::Consistency)
        ]),
        None,
        "a conflict at the deciding strength is not a reward"
    );
    assert_eq!(
        reward(&[
            v(Outcome::Fail, Strength::Formal),
            v(Outcome::Pass, Strength::Executable)
        ]),
        Some(1.0)
    );
}

/// A set is a named, content-addressed list of ids; it names only
/// experiences the store holds.
#[test]
fn experience_sets_are_content_addressed() {
    let scratch = Scratch::new("sets");
    let store = scratch.store();
    let a = store.put(&experience("a")).unwrap();
    let b = store.put(&experience("b")).unwrap();
    let set = ExperienceSet {
        name: "denoise-pass".into(),
        members: vec![a.clone(), b.clone()],
    };
    let id = store.put_set(&set).unwrap();
    assert_eq!(store.put_set(&set).unwrap(), id);
    assert_eq!(store.get_set(&id).unwrap(), set);
    let reordered = ExperienceSet {
        name: "denoise-pass".into(),
        members: vec![b, a.clone()],
    };
    assert_ne!(
        store.put_set(&reordered).unwrap(),
        id,
        "a set is an ordered list"
    );

    let unknown = experience("never stored").id().unwrap();
    let dangling = ExperienceSet {
        name: "dangling".into(),
        members: vec![a, unknown],
    };
    assert!(matches!(
        store.put_set(&dangling),
        Err(StoreError::UnknownExperience(_))
    ));
}
