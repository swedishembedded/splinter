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
use splinter_expdb::model::{Entity, RecordKind};
use splinter_expdb::query::Query;
use splinter_expdb::{Config, Database, Session, WriterIdentity};
use splinter_record::annotation::{
    reward, Annotation, AnnotationBody, Outcome, Producer, RelationKind, Strength,
};
use splinter_record::clock::FixedClock;
use splinter_record::experience::{
    Digest, Environment, Experience, Privileged, PrivilegedKind, Provenance, Span, Task,
};
use splinter_record::experiences::{ExperienceSet, ExperienceStore, StoreError};
use splinter_record::workspace::Workspace;
use splinter_record::StateRoot;
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
    /// A store over a newly opened workspace: a separate writer on the same
    /// database, as another process would be.
    fn store(&self) -> ExperienceStore {
        ExperienceStore::new(&self.workspace())
    }

    fn workspace(&self) -> Workspace {
        Workspace::at(&StateRoot::new(&self.0))
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
    id: &splinter_record::experience::ExperienceId,
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
        id.as_str().starts_with("blake3:") && id.as_str().len() == 71,
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
            Err(splinter_record::experience::ExperienceError::EnvironmentSnapshot { .. })
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
    assert_eq!(store.put(&exp).unwrap(), first);
    assert_eq!(
        scratch.store().put(&exp).unwrap(),
        first,
        "another writer storing the same content agrees on the id"
    );
    let db = Database::open(StateRoot::new(&scratch.0).expdb(), Config::default()).unwrap();
    let snapshot = db.snapshot().unwrap();
    let stored = snapshot
        .query(&Query::all().kind(RecordKind::Entity))
        .unwrap()
        .records
        .iter()
        .filter(|r| matches!(&r.body, splinter_expdb::model::Body::Entity(e) if e.class == "experience"))
        .count();
    assert_eq!(stored, 1, "one record however often it is put");
    assert_eq!(
        snapshot
            .query(&Query::all().kind(RecordKind::Attempt))
            .unwrap()
            .records
            .len(),
        1,
        "and one attempt"
    );
    assert_eq!(store.get(&first).unwrap(), exp);
}

/// A stored object that no longer matches its address is an error on read,
/// and a later put of the true content does not paper over it.
#[test]
fn an_object_that_no_longer_matches_its_address_is_an_error_not_silently_accepted() {
    let scratch = Scratch::new("corrupt");
    let id = experience("quick brown fox").id().unwrap();
    // Something else stored under the true content's address.
    let db = Database::open(StateRoot::new(&scratch.0).expdb(), Config::default()).unwrap();
    let mut session = Session::open(&db, &WriterIdentity::new("test", "forge", "node", 0)).unwrap();
    let forged = serde_json::to_value(experience("quick brown cat")).unwrap();
    session
        .put_entity(&Entity::keyed(
            "experience",
            id.0.content_id().unwrap(),
            forged,
        ))
        .unwrap();
    session.flush().unwrap();

    let store = scratch.store();
    assert!(matches!(store.get(&id), Err(StoreError::Altered { .. })));
    assert!(matches!(
        store.put(&experience("quick brown fox")),
        Err(StoreError::Altered { .. })
    ));
}

/// Damage to the files under the database is an error, never a quietly wrong
/// read.
#[test]
fn damaged_storage_is_an_error_not_a_wrong_answer() {
    let scratch = Scratch::new("damaged");
    let id = scratch.store().put(&experience("quick brown fox")).unwrap();
    let segments: Vec<_> = walk(&scratch.0.join("expdb/segments"));
    assert!(!segments.is_empty());
    for path in segments {
        let mut bytes = fs::read(&path).unwrap();
        let middle = bytes.len() / 2;
        bytes[middle] ^= 0xff;
        fs::write(&path, bytes).unwrap();
    }
    let outcome = ExperienceStore::new(&Workspace::at(&StateRoot::new(&scratch.0))).get(&id);
    assert!(outcome.is_err());
}

fn walk(dir: &std::path::Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    for entry in fs::read_dir(dir).into_iter().flatten().flatten() {
        let path = entry.path();
        if path.is_dir() {
            found.extend(walk(&path));
        } else {
            found.push(path);
        }
    }
    found
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

/// Writes made in a batch are committed together: another writer sees none
/// of them until the batch ends, and all of them after.
#[test]
fn a_batch_commits_its_writes_together() {
    let scratch = Scratch::new("batch");
    let workspace = scratch.workspace();
    let store = ExperienceStore::new(&workspace);
    let reader = scratch.store();

    let batch = workspace.batch();
    let first = store.put(&experience("quick brown fox")).unwrap();
    let second = store.put(&experience("quick fox")).unwrap();
    store
        .annotate(&verdict(&first, Outcome::Pass, Strength::Formal))
        .unwrap();
    assert_eq!(
        store.annotations(&first).unwrap().annotations.len(),
        1,
        "the writer reads its own"
    );
    assert!(
        reader.list().unwrap().is_empty(),
        "no one else sees a thing"
    );
    batch.commit().unwrap();

    let reader_workspace = scratch.workspace();
    reader_workspace.refresh().unwrap();
    let after = ExperienceStore::new(&reader_workspace);
    let mut both = vec![first.clone(), second];
    both.sort();
    assert_eq!(after.list().unwrap(), both);
    assert_eq!(after.annotations(&first).unwrap().annotations.len(), 1);
}

/// A batch does not hold its writes back for ever: it commits them in groups
/// as it goes, so a crash costs the last group and not the whole batch.
#[test]
fn a_long_batch_commits_in_groups_as_it_goes() {
    let scratch = Scratch::new("groups");
    let workspace = scratch.workspace();
    let store = ExperienceStore::new(&workspace);
    let batch = workspace.batch();
    for n in 0..300 {
        store.put(&experience(&format!("answer {n}"))).unwrap();
    }
    let seen = scratch.store().list().unwrap().len();
    assert!(
        (1..300).contains(&seen),
        "some of it is committed and the rest is held: {seen}"
    );
    batch.commit().unwrap();
    assert_eq!(scratch.store().list().unwrap().len(), 300);
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

/// A very large experience does not make every small neighbour in its block
/// expensive to read: it is kept as a blob and reads back whole.
#[test]
fn a_very_large_experience_is_spilled_to_a_blob_and_reads_back_whole() {
    let scratch = Scratch::new("spill");
    let store = scratch.store();
    let big = experience(&"x".repeat(700_000));
    let id = store.put(&big).unwrap();
    assert_eq!(store.get(&id).unwrap(), big);
    assert_eq!(
        scratch.store().get(&id).unwrap(),
        big,
        "through a fresh handle too"
    );

    let db = Database::open(StateRoot::new(&scratch.0).expdb(), Config::default()).unwrap();
    let entity = db
        .snapshot()
        .unwrap()
        .entity_body(&id.0.content_id().unwrap())
        .unwrap()
        .unwrap();
    assert!(
        entity.value.get("$spilled").is_some(),
        "the record holds a pointer, not the text"
    );
    assert_eq!(entity.blobs.len(), 1);
}

/// Re-running a stage over the same experience says the same thing again; it
/// is recorded once.
#[test]
fn saying_the_same_thing_twice_records_it_once() {
    let scratch = Scratch::new("repeat");
    let store = scratch.store();
    let id = store.put(&experience("quick brown fox")).unwrap();
    let note = verdict(&id, Outcome::Pass, Strength::Formal);
    for _ in 0..3 {
        store.annotate(&note).unwrap();
    }
    assert_eq!(
        store.annotations(&id).unwrap().annotations,
        vec![note.clone()]
    );
    // A different verdict from the same grader is a new statement.
    let again = verdict(&id, Outcome::Fail, Strength::Formal);
    store.annotate(&again).unwrap();
    assert_eq!(store.annotations(&id).unwrap().annotations.len(), 2);
}
