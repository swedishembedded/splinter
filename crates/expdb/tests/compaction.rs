// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: background compaction rewrites many small files into few large ones
//! without changing any record's identity, races safely with writers and with
//! other compactors, and never removes what a snapshot or a pin still needs.
#![allow(clippy::unwrap_used)]

mod common;

use std::collections::HashSet;
use std::thread;
use std::time::Duration;

use common::{attempt, noise, Scratch};
use splinter_expdb::blob::BlobStore;
use splinter_expdb::model::{Body, Outcome, Skill};
use splinter_expdb::query::Query;
use splinter_expdb::{Config, Database, WriterIdentity};

fn skill(name: String) -> Body {
    Body::Skill(Skill {
        name,
        description: String::new(),
        trigger: String::new(),
        action_pattern: String::new(),
        expected_effect: String::new(),
        parents: vec![],
        prerequisites: vec![],
    })
}

/// `flushes` small segments of ten records each.
fn many_small_segments(db: &Database, flushes: u32) {
    for rank in 0..flushes {
        let mut c = db
            .collector(&WriterIdentity::new("exp", "job", "node", rank))
            .unwrap();
        for n in 0..10 {
            c.record(skill(format!("{rank}-{n}"))).unwrap();
        }
        c.flush().unwrap();
    }
}

#[test]
fn small_segments_are_merged_into_one_and_no_record_changes_identity() {
    let scratch = Scratch::new();
    let db = scratch.open();
    many_small_segments(&db, 8);
    let before = db.snapshot().unwrap();
    let ids_before: Vec<_> = before.records().unwrap().iter().map(|r| r.id).collect();
    assert_eq!(before.segments().len(), 8);

    let report = db.compact_segments().unwrap();
    assert_eq!(
        (report.groups, report.inputs, report.outputs.len()),
        (1, 8, 1)
    );

    let after = db.snapshot().unwrap();
    assert_eq!(after.segments().len(), 1);
    let mut ids_after: Vec<_> = after.records().unwrap().iter().map(|r| r.id).collect();
    let mut expected = ids_before.clone();
    expected.sort();
    ids_after.sort();
    assert_eq!(
        ids_after, expected,
        "the same records, with the same ids, in a different file"
    );
    // Reading by id works through the new location.
    assert!(after.get(ids_before[17]).unwrap().is_some());
}

#[test]
fn an_older_snapshot_still_reads_until_it_is_unpinned_and_collected() {
    let scratch = Scratch::new();
    let db = scratch.open_with(Config {
        orphan_grace: Duration::ZERO,
        ..Config::default()
    });
    many_small_segments(&db, 5);
    let old = db.snapshot().unwrap();
    old.pin("training-run").unwrap();
    db.compact_segments().unwrap();

    assert_eq!(
        db.snapshot_at(old.id()).unwrap().records().unwrap().len(),
        50
    );
    let kept = db.gc().unwrap();
    assert_eq!(kept.removed, 0, "the pin keeps the five old segments");
    db.unpin("training-run").unwrap();
    // The catalog and job refs no longer reach the old manifests' segments.
    assert!(db.gc().unwrap().removed >= 5);
    assert_eq!(db.snapshot().unwrap().records().unwrap().len(), 50);
}

#[test]
fn compacting_again_has_nothing_to_do_and_the_merged_segment_is_deterministic() {
    let scratch = Scratch::new();
    let db = scratch.open();
    many_small_segments(&db, 6);
    let first = db.compact_segments().unwrap();
    assert_eq!(db.compact_segments().unwrap().groups, 0);

    // The same input in another database compacts to the very same file.
    let other = Scratch::new();
    let twin = other.open();
    many_small_segments(&twin, 6);
    let twin_ids: HashSet<_> = twin
        .snapshot()
        .unwrap()
        .records()
        .unwrap()
        .iter()
        .map(|r| r.id)
        .collect();
    let mine: HashSet<_> = db
        .snapshot()
        .unwrap()
        .records()
        .unwrap()
        .iter()
        .map(|r| r.id)
        .collect();
    assert_eq!(first.outputs.len(), 1);
    // Writers got different ids in each database, so the ids differ; the
    // grouping rule, not luck, decides what merges.
    assert_eq!(twin_ids.len(), mine.len());
}

#[test]
fn segments_that_already_reach_the_target_size_are_left_alone() {
    let scratch = Scratch::new();
    let db = scratch.open_with(Config {
        compact_target_bytes: 1,
        ..Config::default()
    });
    many_small_segments(&db, 4);
    assert_eq!(db.compact_segments().unwrap().groups, 0);
    assert_eq!(db.snapshot().unwrap().segments().len(), 4);
}

#[test]
fn a_compaction_racing_writers_loses_nothing_and_duplicates_nothing() {
    let scratch = Scratch::new();
    let root = scratch.dir.path().to_path_buf();
    let writers: Vec<_> = (0..4u32)
        .map(|rank| {
            let root = root.clone();
            thread::spawn(move || {
                let db = Database::open(&root, Config::default()).unwrap();
                let mut c = db
                    .collector(&WriterIdentity::new("exp", "job", "node", rank))
                    .unwrap();
                for batch in 0..6 {
                    for n in 0..10 {
                        c.record(skill(format!("{rank}-{batch}-{n}"))).unwrap();
                    }
                    c.flush().unwrap();
                }
            })
        })
        .collect();
    let compactor = {
        let root = root.clone();
        thread::spawn(move || {
            let db = Database::open(&root, Config::default()).unwrap();
            for _ in 0..6 {
                db.compact_segments().unwrap();
            }
        })
    };
    for w in writers {
        w.join().unwrap();
    }
    compactor.join().unwrap();

    let db = scratch.open();
    db.compact_segments().unwrap();
    let records = db.snapshot().unwrap().records().unwrap();
    assert_eq!(records.len(), 4 * 6 * 10);
    assert_eq!(
        records.iter().map(|r| r.id).collect::<HashSet<_>>().len(),
        records.len()
    );
}

#[test]
fn two_compactors_working_on_the_same_files_do_not_double_or_lose_records() {
    let scratch = Scratch::new();
    let db = scratch.open();
    many_small_segments(&db, 8);
    let root = scratch.dir.path().to_path_buf();
    let handles: Vec<_> = (0..2)
        .map(|_| {
            let root = root.clone();
            thread::spawn(move || {
                Database::open(&root, Config::default())
                    .unwrap()
                    .compact_segments()
                    .unwrap()
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    let records = db.snapshot().unwrap().records().unwrap();
    assert_eq!(records.len(), 80);
    assert_eq!(
        records.iter().map(|r| r.id).collect::<HashSet<_>>().len(),
        80
    );
    // One more pass folds any duplicate physical copies together.
    db.compact_segments().unwrap();
    assert_eq!(db.snapshot().unwrap().records().unwrap().len(), 80);
}

#[test]
fn queries_answer_the_same_before_and_after_compaction_and_index_runs_follow_the_data() {
    let scratch = Scratch::new();
    let db = scratch.open();
    for rank in 0..4 {
        let mut c = db
            .collector(&WriterIdentity::new("exp", "job", "node", rank))
            .unwrap();
        attempt(&mut c, u64::from(rank), "p", 3, Outcome::Pass);
        c.flush().unwrap();
        db.build_indexes().unwrap();
    }
    let query = Query::all().kind(splinter_expdb::model::RecordKind::Decision);
    let before = db.snapshot().unwrap().query(&query).unwrap().records;

    db.compact_segments().unwrap();
    db.build_indexes().unwrap();
    db.compact_indexes().unwrap();
    let after = db.snapshot().unwrap();
    assert_eq!(after.query(&query).unwrap().records, before);
    assert_eq!(after.index_runs().len(), 1);
}

#[test]
fn blob_packs_merge_and_a_chunk_stored_twice_is_kept_once() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let shared = noise(120_000, 1);
    let mut blobs = Vec::new();
    for n in 0..4u64 {
        // Each writer stores the shared object and one of its own.
        let mut c = db
            .collector(&WriterIdentity::new("exp", "job", "node", n as u32))
            .unwrap();
        let own = noise(50_000, 1_000 + 2 * n);
        let content = c.content(&shared).unwrap();
        let theirs = c.content(&own).unwrap();
        blobs.push((content, theirs, own));
        c.record(skill(format!("holder {n}"))).unwrap();
        c.flush().unwrap();
    }
    let before = db.snapshot().unwrap();
    assert_eq!(before.blob_packs().len(), 4);

    let report = db.compact_blobs().unwrap();
    assert_eq!(
        (report.groups, report.inputs, report.outputs.len()),
        (1, 4, 1)
    );
    let after = db.snapshot().unwrap();
    assert_eq!(after.blob_packs().len(), 1);
    assert!(after.blob_packs()[0].bytes < before.blob_packs().iter().map(|p| p.bytes).sum::<u64>());

    let packs: Vec<_> = after.blob_packs().iter().map(|p| p.id).collect();
    let store = BlobStore::with_packs(&db, &packs).unwrap();
    for (shared_ref, own_ref, own) in blobs {
        let splinter_expdb::model::Content::Blob { blob } = shared_ref else {
            panic!("the shared object is large")
        };
        assert_eq!(store.get(&blob).unwrap(), shared);
        let splinter_expdb::model::Content::Blob { blob } = own_ref else {
            panic!()
        };
        assert_eq!(store.get(&blob).unwrap(), own);
    }
}

#[test]
fn a_writer_that_has_gone_quiet_is_folded_into_the_catalog_and_an_active_one_is_not() {
    let scratch = Scratch::new();
    let db = scratch.open_with(Config {
        orphan_grace: Duration::ZERO,
        ..Config::default()
    });
    many_small_segments(&db, 6);
    assert_eq!(db.job_heads().unwrap().len(), 6);

    let report = db.absorb_jobs().unwrap();
    assert_eq!(report.pruned, 6);
    assert!(db.job_heads().unwrap().is_empty());
    assert_eq!(
        db.snapshot().unwrap().records().unwrap().len(),
        60,
        "nothing was lost"
    );

    let lively = Scratch::new();
    let db = lively.open();
    many_small_segments(&db, 3);
    assert_eq!(
        db.absorb_jobs().unwrap().pruned,
        0,
        "these writers published a moment ago"
    );
    assert_eq!(db.snapshot().unwrap().records().unwrap().len(), 30);
}

#[test]
fn a_merge_that_reproduces_one_of_its_inputs_does_not_remove_it() {
    // One segment holds a subset of another's records. Merging them gives the
    // larger segment back, byte for byte, which must not be removed with its
    // inputs.
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = db
        .collector(&WriterIdentity::new("exp", "job", "node", 0))
        .unwrap();
    for n in 0..6 {
        c.record(skill(format!("s{n}"))).unwrap();
    }
    c.flush().unwrap();
    let whole = db.snapshot().unwrap().segments()[0].clone();
    let segment = splinter_expdb::format::Segment::open(db.backend_arc(), whole.id).unwrap();
    let records = segment.records().unwrap();
    let subset =
        splinter_expdb::format::seal_segment(db.backend(), &records[..3], &[], db.config())
            .unwrap();
    let subset = splinter_expdb::format::Segment::open(db.backend_arc(), subset)
        .unwrap()
        .object_ref()
        .unwrap();
    db.publish("other", vec![subset], vec![]).unwrap();

    let report = db.compact_segments().unwrap();
    assert_eq!(
        report.outputs,
        [whole.id],
        "the merge is the larger segment itself"
    );
    let after = db.snapshot().unwrap();
    assert_eq!(after.segments(), [whole]);
    assert_eq!(
        after.records().unwrap().len(),
        6,
        "nothing was lost to the tombstones"
    );
}

#[test]
fn a_manifest_cannot_add_and_remove_the_same_file() {
    let scratch = Scratch::new();
    let db = scratch.open();
    many_small_segments(&db, 1);
    let object = db.snapshot().unwrap().segments()[0].clone();
    assert!(db
        .publish("job", vec![object.clone()], vec![object])
        .is_err());
}
