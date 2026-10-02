// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: an application can store its own typed, content-addressed objects
//! next to the experience graph and find them again by content id or class,
//! across writers and after a flush.
#![allow(clippy::unwrap_used)]

mod common;

use common::Scratch;
use splinter_expdb::blob::BlobStore;
use splinter_expdb::model::{Body, Entity, RecordKind};
use splinter_expdb::{ContentId, WriterIdentity};

fn identity(rank: u32) -> WriterIdentity {
    WriterIdentity::new("exp1", "job-1", "node-a", rank)
}

#[test]
fn an_entity_is_found_by_its_content_id_after_a_flush() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut collector = db.collector(&identity(0)).unwrap();
    let source = Entity::new(
        "source",
        serde_json::json!({"path": "notes/a.md", "parts": 2}),
    );

    let id = collector.put_entity(&source).unwrap();
    assert_eq!(
        id,
        source.id().unwrap(),
        "the id is a function of the content"
    );
    assert!(
        db.snapshot().unwrap().entity(&id).unwrap().is_none(),
        "nothing is visible before the flush"
    );
    collector.flush().unwrap();

    let snapshot = db.snapshot().unwrap();
    let record = snapshot.entity(&id).unwrap().expect("found by content id");
    match snapshot.get(record).unwrap().unwrap().body {
        Body::Entity(found) => assert_eq!(found, source),
        other => panic!("not an entity: {other:?}"),
    }
    assert_eq!(snapshot.entity_body(&id).unwrap(), Some(source));
    assert_eq!(snapshot.entity(&ContentId::of(b"absent")).unwrap(), None);
}

#[test]
fn the_same_content_from_two_writers_is_one_logical_entity() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let thing = Entity::new("task", serde_json::json!({"instruction": "list the files"}));
    let mut ids = Vec::new();
    for rank in 0..2 {
        let mut collector = db.collector(&identity(rank)).unwrap();
        ids.push(collector.put_entity(&thing).unwrap());
        assert_eq!(
            collector.put_entity(&thing).unwrap(),
            ids[0],
            "putting twice through one collector stores one record"
        );
        collector.flush().unwrap();
    }
    assert_eq!(ids[0], ids[1]);

    let snapshot = db.snapshot().unwrap();
    let found = snapshot.entities("task").unwrap();
    assert_eq!(found.len(), 1, "listing by class names it once");
    assert_eq!(found[0].id, ids[0]);
    assert_eq!(found[0].entity, thing);
}

#[test]
fn classes_keep_entities_apart_and_the_class_is_part_of_the_identity() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut collector = db.collector(&identity(0)).unwrap();
    let value = serde_json::json!({"name": "same"});
    let a = collector
        .put_entity(&Entity::new("task_set", value.clone()))
        .unwrap();
    let b = collector
        .put_entity(&Entity::new("experience_set", value))
        .unwrap();
    collector.flush().unwrap();
    assert_ne!(a, b);

    let snapshot = db.snapshot().unwrap();
    assert_eq!(snapshot.entities("task_set").unwrap().len(), 1);
    assert_eq!(snapshot.entities("experience_set").unwrap().len(), 1);
    assert!(snapshot.entities("release").unwrap().is_empty());
    let by_kind = snapshot
        .query(&splinter_expdb::query::Query::all().kind(RecordKind::Entity))
        .unwrap();
    assert_eq!(by_kind.records.len(), 2);
}

#[test]
fn entities_listed_by_class_come_back_in_write_order() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut collector = db.collector(&identity(0)).unwrap();
    let ids: Vec<ContentId> = (0..5)
        .map(|n| {
            collector
                .put_entity(&Entity::new("run_event", serde_json::json!({"n": n})))
                .unwrap()
        })
        .collect();
    collector.flush().unwrap();
    let listed: Vec<ContentId> = db
        .snapshot()
        .unwrap()
        .entities("run_event")
        .unwrap()
        .iter()
        .map(|e| e.id)
        .collect();
    assert_eq!(listed, ids);
}

#[test]
fn an_entity_names_the_blobs_it_carries() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut collector = db.collector(&identity(0)).unwrap();
    let blob = collector.content(&common::noise(100_000, 3)).unwrap();
    let splinter_expdb::model::Content::Blob { blob } = blob else {
        panic!("a large payload is stored as a blob");
    };
    let part = Entity::new("source", serde_json::json!({"name": "big"})).with_blobs(vec![blob]);
    let id = collector.put_entity(&part).unwrap();
    collector.flush().unwrap();

    let snapshot = db.snapshot().unwrap();
    let found = snapshot.entity_body(&id).unwrap().unwrap();
    assert_eq!(found.blobs, vec![blob]);
    let packs: Vec<ContentId> = snapshot.blob_packs().iter().map(|p| p.id).collect();
    let store = BlobStore::with_packs(&db, &packs).unwrap();
    assert_eq!(
        store.get(&found.blobs[0]).unwrap(),
        common::noise(100_000, 3)
    );
}
