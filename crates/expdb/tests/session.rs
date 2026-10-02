// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: a session is one process's handle for writing and reading back what
//! it wrote. It sees its own writes at once, sees other writers' only when
//! it refreshes, and answers a lookup after thousands of writes without
//! rebuilding anything from the whole database.
#![allow(clippy::unwrap_used)]

mod common;

use common::Scratch;
use splinter_expdb::analyze::EvalFilter;
use splinter_expdb::model::{Entity, Evaluation, EvaluatorRef, Rel, Target};
use splinter_expdb::{Database, Session, WriterIdentity};

fn identity(rank: u32) -> WriterIdentity {
    WriterIdentity::new("exp", "session", "node", rank)
}

fn open(db: &Database, rank: u32) -> Session {
    Session::open(db, &identity(rank)).unwrap()
}

fn note(target: Target, who: &str, score: f64) -> Evaluation {
    Evaluation::new(target, EvaluatorRef::new(who, "1"), "c", score, 1.0)
}

#[test]
fn a_session_reads_what_it_wrote_before_anything_is_flushed() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut session = open(&db, 0);
    let thing = Entity::new("source", serde_json::json!({"n": 1}));
    let id = session.put_entity(&thing).unwrap();

    assert_eq!(session.entity(&id).unwrap(), Some(thing.clone()));
    assert_eq!(session.entities("source").unwrap().len(), 1);
    assert_eq!(session.put_entity(&thing).unwrap(), id);
    assert_eq!(session.entities("source").unwrap().len(), 1, "stored once");
    assert!(
        db.snapshot().unwrap().entity(&id).unwrap().is_none(),
        "others see nothing until it is flushed"
    );

    session.flush().unwrap();
    assert!(db.snapshot().unwrap().entity(&id).unwrap().is_some());
    assert_eq!(
        session.entity(&id).unwrap(),
        Some(thing),
        "and still sees it"
    );
}

#[test]
fn another_writers_entities_appear_on_refresh_and_not_before() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut mine = open(&db, 0);
    let mut theirs = open(&db, 1);
    let id = theirs
        .put_entity(&Entity::new("task", serde_json::json!("theirs")))
        .unwrap();
    theirs.flush().unwrap();

    assert_eq!(
        mine.entity(&id).unwrap(),
        None,
        "a session reads a point in time"
    );
    mine.refresh().unwrap();
    assert!(mine.entity(&id).unwrap().is_some());
    assert_eq!(
        mine.put_entity(&Entity::new("task", serde_json::json!("theirs")))
            .unwrap(),
        id,
        "the same content is the same entity whoever wrote it"
    );
}

#[test]
fn evaluations_of_a_target_include_the_unflushed_and_honour_withdrawals() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut session = open(&db, 0);
    let id = session
        .put_entity(&Entity::new("experience", serde_json::json!(1)))
        .unwrap();
    let target = Target::Entity(id);
    session.evaluate(note(target, "judge", 1.0)).unwrap();
    session.flush().unwrap();
    session.evaluate(note(target, "tests", 0.0)).unwrap();

    let seen = session
        .evaluations(&EvalFilter::new().target(target))
        .unwrap();
    assert_eq!(seen.len(), 2, "one flushed and one not");
    assert_eq!(seen[0].evaluation.evaluator.name, "judge", "in write order");
    assert!(session
        .evaluations(&EvalFilter::new().target(Target::Entity(splinter_expdb::ContentId::of(b"x"))))
        .unwrap()
        .is_empty());

    let mut other = open(&db, 1);
    other
        .retract(EvaluatorRef::new("judge", "1"), "wrong")
        .unwrap();
    other.flush().unwrap();
    session.refresh().unwrap();
    let seen = session
        .evaluations(&EvalFilter::new().target(target))
        .unwrap();
    assert_eq!(seen.len(), 1, "the withdrawn evaluator no longer counts");
    assert_eq!(seen[0].evaluation.evaluator.name, "tests");
}

#[test]
fn links_between_entities_are_recorded_by_content_id() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut session = open(&db, 0);
    let a = session
        .put_entity(&Entity::new("experience", serde_json::json!("a")))
        .unwrap();
    let b = session
        .put_entity(&Entity::new("experience", serde_json::json!("b")))
        .unwrap();
    session.link(&b, Rel::RetryOf, &a).unwrap();
    session.flush().unwrap();

    let snapshot = db.snapshot().unwrap();
    let (ra, rb) = (
        snapshot.entity(&a).unwrap().unwrap(),
        snapshot.entity(&b).unwrap().unwrap(),
    );
    let index = snapshot.index().unwrap();
    assert_eq!(index.edges_from(rb, Some(Rel::RetryOf)), vec![ra]);
    assert!(session
        .link(&a, Rel::RetryOf, &splinter_expdb::ContentId::of(b"nobody"))
        .is_err());
}

#[test]
fn bytes_are_readable_by_the_session_that_wrote_them_and_by_others_once_flushed() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut session = open(&db, 0);
    let blob = session.put_blob(b"raw part").unwrap();
    assert_eq!(session.read_blob(&blob.id).unwrap(), b"raw part");
    session.flush().unwrap();
    let mut other = open(&db, 1);
    assert_eq!(other.read_blob(&blob.id).unwrap(), b"raw part");
    assert!(other
        .read_blob(&splinter_expdb::ContentId::of(b"absent"))
        .is_err());
}

#[test]
fn thousands_of_interleaved_writes_and_lookups_stay_correct_across_refreshes() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut session = open(&db, 0).refreshing_every(64);
    let mut ids = Vec::new();
    for n in 0..1_500u64 {
        let id = session
            .put_entity(&Entity::new("experience", serde_json::json!({ "n": n })))
            .unwrap();
        session
            .evaluate(note(Target::Entity(id), "judge", 1.0))
            .unwrap();
        if n % 100 == 0 {
            session.flush().unwrap();
        }
        let back = session
            .evaluations(&EvalFilter::new().target(Target::Entity(id)))
            .unwrap();
        assert_eq!(back.len(), 1, "write {n}");
        ids.push(id);
    }
    assert_eq!(session.entities("experience").unwrap().len(), 1_500);
    session.flush().unwrap();
    for id in ids.iter().step_by(97) {
        assert!(session.entity(id).unwrap().is_some());
    }
    let fresh = db.snapshot().unwrap();
    assert_eq!(fresh.entities("experience").unwrap().len(), 1_500);
}

#[test]
fn a_session_records_an_attempt_and_the_evaluations_and_links_about_it() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut session = open(&db, 0);
    let (first, second) = session
        .with_collector(|c| {
            let (_, a) = common::attempt(c, 1, "p", 2, splinter_expdb::model::Outcome::Fail);
            let (_, b) = common::attempt(c, 1, "p", 2, splinter_expdb::model::Outcome::Pass);
            Ok((a, b))
        })
        .unwrap();
    session
        .evaluate(note(Target::Record(second), "tests", 1.0))
        .unwrap();
    session.link_records(second, Rel::RetryOf, first).unwrap();
    assert_eq!(
        session
            .evaluations(&EvalFilter::new().target(Target::Record(second)))
            .unwrap()
            .len(),
        1
    );
    session.flush().unwrap();

    let snapshot = db.snapshot().unwrap();
    let index = snapshot.index().unwrap();
    assert_eq!(index.edges_from(second, Some(Rel::RetryOf)), vec![first]);
    assert_eq!(snapshot.families().unwrap()[0].attempts.len(), 2);
}

#[test]
fn a_write_that_fails_halfway_leaves_nothing_behind() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut session = open(&db, 0);
    let kept = session
        .put_entity(&Entity::new("task", serde_json::json!("kept")))
        .unwrap();

    let mark = session.begin();
    let first = session
        .put_entity(&Entity::new("task", serde_json::json!("first")))
        .unwrap();
    session
        .evaluate(note(Target::Entity(first), "judge", 1.0))
        .unwrap();
    session
        .retract(EvaluatorRef::new("judge", "1"), "mistake")
        .unwrap();
    session.rollback(mark);

    assert_eq!(
        session.entity(&first).unwrap(),
        None,
        "the session no longer reads it"
    );
    assert_eq!(session.entities("task").unwrap().len(), 1);
    assert!(session
        .evaluations(&EvalFilter::new().target(Target::Entity(first)))
        .unwrap()
        .is_empty());
    session.flush().unwrap();
    let snapshot = db.snapshot().unwrap();
    assert!(
        snapshot.entity(&first).unwrap().is_none(),
        "and it was never committed"
    );
    assert!(
        snapshot.entity(&kept).unwrap().is_some(),
        "what came before is untouched"
    );
    assert_eq!(snapshot.evaluations(&EvalFilter::new()).unwrap().len(), 0);
}
