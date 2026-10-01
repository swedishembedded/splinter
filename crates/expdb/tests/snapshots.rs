// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: a file is part of the database only when a manifest names it;
//! manifests merge without conflicts; a snapshot never changes under its
//! reader; and garbage collection removes only what nothing can reach.
#![allow(clippy::unwrap_used)]

mod common;

use std::time::Duration;

use common::{mixed, Scratch};
use splinter_expdb::format::{seal_segment, Segment};
use splinter_expdb::manifest::{Manifest, ObjectRef};
use splinter_expdb::{Config, Database};

/// Seals a segment of `count` records by writer `writer` and returns its ref.
fn segment(db: &Database, writer: u64, count: u64) -> ObjectRef {
    let (records, edges) = mixed(writer, count);
    let id = seal_segment(db.backend(), &records, &edges, db.config()).unwrap();
    Segment::open(db.backend_arc(), id)
        .unwrap()
        .object_ref()
        .unwrap()
}

#[test]
fn a_sealed_segment_is_invisible_until_a_manifest_names_it() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let orphan = segment(&db, 1, 8);
    assert!(db.snapshot().unwrap().records().unwrap().is_empty());

    db.publish("job-1", vec![orphan], vec![]).unwrap();
    assert_eq!(db.snapshot().unwrap().records().unwrap().len(), 8);
}

#[test]
fn a_snapshot_does_not_change_when_more_is_published() {
    let scratch = Scratch::new();
    let db = scratch.open();
    db.publish("job-1", vec![segment(&db, 1, 4)], vec![])
        .unwrap();
    let before = db.snapshot().unwrap();

    db.publish("job-1", vec![segment(&db, 2, 6)], vec![])
        .unwrap();

    assert_eq!(before.records().unwrap().len(), 4);
    assert_eq!(db.snapshot().unwrap().records().unwrap().len(), 10);
    assert_ne!(before.id(), db.snapshot().unwrap().id());
}

#[test]
fn a_snapshot_can_be_reopened_by_its_id_and_reads_the_same() {
    let scratch = Scratch::new();
    let db = scratch.open();
    db.publish("job-1", vec![segment(&db, 1, 4)], vec![])
        .unwrap();
    let snapshot = db.snapshot().unwrap();
    snapshot.pin("training-run-1").unwrap();
    db.publish("job-1", vec![segment(&db, 2, 3)], vec![])
        .unwrap();

    let again = db.snapshot_at(snapshot.id()).unwrap();
    assert_eq!(again.records().unwrap(), snapshot.records().unwrap());
}

#[test]
fn concurrent_jobs_publish_without_coordination_and_a_snapshot_sees_both() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let (a, b) = (segment(&db, 1, 5), segment(&db, 2, 7));
    // Two independent handles, as two nodes would hold.
    let other = scratch.open();
    db.publish("job-a", vec![a], vec![]).unwrap();
    other.publish("job-b", vec![b], vec![]).unwrap();

    assert_eq!(db.snapshot().unwrap().records().unwrap().len(), 12);
    assert_eq!(other.snapshot().unwrap().id(), db.snapshot().unwrap().id());
}

#[test]
fn merging_the_same_heads_always_gives_the_same_manifest() {
    let a = Manifest::merge(vec![
        splinter_expdb::ContentId::of(b"a"),
        splinter_expdb::ContentId::of(b"b"),
    ]);
    let b = Manifest::merge(vec![
        splinter_expdb::ContentId::of(b"b"),
        splinter_expdb::ContentId::of(b"a"),
        splinter_expdb::ContentId::of(b"a"),
    ]);
    assert_eq!(a.id().unwrap(), b.id().unwrap());
}

#[test]
fn an_object_removed_on_one_branch_stays_removed_when_another_branch_still_names_it() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let old = segment(&db, 1, 6);
    db.publish("seed", vec![old.clone()], vec![]).unwrap();
    // Branch A compacts old away; branch B, unaware, publishes something else.
    let replacement = segment(&db, 1, 6);
    assert_eq!(replacement, old, "same records seal to the same segment");
    let merged = segment(&db, 3, 2);
    db.publish("compactor", vec![merged.clone()], vec![old.clone()])
        .unwrap();
    db.publish("bystander", vec![segment(&db, 4, 2)], vec![])
        .unwrap();

    let snapshot = db.snapshot().unwrap();
    assert!(!snapshot.segments().contains(&old));
    assert!(snapshot.segments().contains(&merged));
}

#[test]
fn a_reader_never_sees_the_same_record_twice_even_if_two_segments_hold_it() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let (records, edges) = mixed(1, 6);
    let small = seal_segment(db.backend(), &records[..4], &edges[..3], db.config()).unwrap();
    let all = seal_segment(db.backend(), &records, &edges, db.config()).unwrap();
    for (job, id) in [("a", small), ("b", all)] {
        let object = Segment::open(db.backend_arc(), id)
            .unwrap()
            .object_ref()
            .unwrap();
        db.publish(job, vec![object], vec![]).unwrap();
    }
    assert_eq!(db.snapshot().unwrap().records().unwrap().len(), 6);
}

#[test]
fn a_checkpoint_resolves_to_the_same_files_as_walking_the_whole_history() {
    let scratch = Scratch::new();
    let db = scratch.open();
    for writer in 1..=5 {
        db.publish("job", vec![segment(&db, writer, 3)], vec![])
            .unwrap();
    }
    let walked = db.snapshot().unwrap().segments().to_vec();
    db.checkpoint().unwrap();
    assert_eq!(db.snapshot().unwrap().segments(), walked);
}

#[test]
fn pins_are_listed_and_released() {
    let scratch = Scratch::new();
    let db = scratch.open();
    db.publish("job", vec![segment(&db, 1, 3)], vec![]).unwrap();
    let snapshot = db.snapshot().unwrap();
    snapshot.pin("run-7").unwrap();
    assert_eq!(db.pins().unwrap(), [("run-7".to_owned(), snapshot.id())]);
    db.unpin("run-7").unwrap();
    assert!(db.pins().unwrap().is_empty());
}

#[test]
fn collection_removes_unreachable_files_but_keeps_what_a_pin_needs() {
    let scratch = Scratch::new();
    let db = scratch.open_with(Config {
        orphan_grace: Duration::ZERO,
        ..Config::default()
    });
    let old = segment(&db, 1, 6);
    db.publish("job", vec![old.clone()], vec![]).unwrap();
    let pinned = db.snapshot().unwrap();
    pinned.pin("training-run").unwrap();
    // Compact `old` away and leave a stray file nothing names.
    let merged = segment(&db, 2, 6);
    db.publish("job", vec![merged.clone()], vec![old.clone()])
        .unwrap();
    let stray = segment(&db, 9, 2);

    let report = db.gc().unwrap();
    assert_eq!(report.removed, 1, "only the stray file goes");
    let reread = db.snapshot_at(pinned.id()).unwrap();
    assert_eq!(reread.records().unwrap().len(), 6);

    db.unpin("training-run").unwrap();
    let report = db.gc().unwrap();
    assert_eq!(
        report.removed, 1,
        "the compacted-away segment goes once nothing pins it"
    );
    let _ = (stray, merged);
}

#[test]
fn a_file_younger_than_the_grace_period_is_never_collected() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let in_flight = segment(&db, 1, 3);
    assert_eq!(db.gc().unwrap().removed, 0);
    db.publish("job", vec![in_flight], vec![]).unwrap();
    assert_eq!(db.snapshot().unwrap().records().unwrap().len(), 3);
}
