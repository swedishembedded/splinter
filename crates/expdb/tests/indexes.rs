// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: indexes are immutable projections of segments. Persisting one saves
//! readers from scanning, a missing one is rebuilt from the segments, runs
//! merge, and an index never serves a record whose segment left the snapshot.
#![allow(clippy::unwrap_used)]

mod common;

use common::{attempt, Scratch};
use splinter_expdb::model::Outcome;
use splinter_expdb::WriterIdentity;

fn fill(db: &splinter_expdb::Database, rank: u32, task: u64) {
    let mut c = db
        .collector(&WriterIdentity::new("exp", "job", "node", rank))
        .unwrap();
    attempt(&mut c, task, "p", 3, Outcome::Pass);
    c.flush().unwrap();
}

#[test]
fn without_a_persisted_index_a_reader_scans_the_segments() {
    let scratch = Scratch::new();
    let db = scratch.open();
    fill(&db, 0, 1);
    let snapshot = db.snapshot().unwrap();
    snapshot.index().unwrap();
    assert!(snapshot.stats().blocks_read > 0);
}

#[test]
fn a_persisted_index_is_read_without_touching_a_single_record_block() {
    let scratch = Scratch::new();
    let db = scratch.open();
    fill(&db, 0, 1);
    fill(&db, 1, 2);
    assert!(db.build_indexes().unwrap().is_some());

    let snapshot = db.snapshot().unwrap();
    let index = snapshot.index().unwrap();
    assert_eq!(snapshot.stats().blocks_read, 0);
    assert_eq!(index.len(), snapshot.records().unwrap().len());
}

#[test]
fn an_index_covering_some_segments_is_completed_by_scanning_the_rest() {
    let scratch = Scratch::new();
    let db = scratch.open();
    fill(&db, 0, 1);
    db.build_indexes().unwrap();
    fill(&db, 1, 2);

    let snapshot = db.snapshot().unwrap();
    let index = snapshot.index().unwrap();
    assert_eq!(index.len(), snapshot.records().unwrap().len());
    let scanned = snapshot.stats().blocks_read;
    assert!(scanned > 0, "the new segment had to be scanned");
    // Building again indexes only what was missing, and then nothing is.
    assert!(db.build_indexes().unwrap().is_some());
    assert!(db.build_indexes().unwrap().is_none());
}

#[test]
fn index_runs_merge_into_one_that_answers_the_same() {
    let scratch = Scratch::new();
    let db = scratch.open();
    for rank in 0..4 {
        fill(&db, rank, u64::from(rank));
        db.build_indexes().unwrap();
    }
    let before = db.snapshot().unwrap();
    assert_eq!(before.index_runs().len(), 4);
    let reference = before.index().unwrap().len();

    assert!(db.compact_indexes().unwrap().is_some());
    let after = db.snapshot().unwrap();
    assert_eq!(after.index_runs().len(), 1);
    assert_eq!(after.index().unwrap().len(), reference);
    assert_eq!(after.stats().blocks_read, 0);
}

#[test]
fn an_index_never_serves_a_record_whose_segment_left_the_snapshot() {
    let scratch = Scratch::new();
    let db = scratch.open();
    fill(&db, 0, 1);
    fill(&db, 1, 2);
    db.build_indexes().unwrap();
    let snapshot = db.snapshot().unwrap();
    let doomed = snapshot.segments().into_iter().next().unwrap();
    let gone: Vec<_> = snapshot
        .open_segment(&doomed)
        .unwrap()
        .records()
        .unwrap()
        .iter()
        .map(|r| r.id)
        .collect();

    db.publish("compactor", vec![], vec![doomed]).unwrap();
    let index = db.snapshot().unwrap().index().unwrap();
    for id in gone {
        assert!(
            !index.contains(id),
            "{id} was served from a removed segment"
        );
    }
}
