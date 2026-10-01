// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: many writers on one root, each with its own handle and no shared
//! memory, never lose or duplicate a record, and a reader only ever sees
//! whole published batches.
#![allow(clippy::unwrap_used)]

mod common;

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

use common::Scratch;
use splinter_expdb::ingest::{Aggregator, Destination};
use splinter_expdb::model::{Body, Skill};
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

const WRITERS: u32 = 8;
const PER_WRITER: u64 = 200;

#[test]
fn many_independent_writers_lose_and_duplicate_nothing() {
    let scratch = Scratch::new();
    let root = scratch.dir.path().to_path_buf();
    let handles: Vec<_> = (0..WRITERS)
        .map(|rank| {
            let root = root.clone();
            thread::spawn(move || {
                let db = Database::open(
                    &root,
                    Config {
                        max_buffered_records: 64,
                        ..Config::default()
                    },
                )
                .unwrap();
                let mut c = db
                    .collector(&WriterIdentity::new("exp", "job", "node", rank))
                    .unwrap();
                for n in 0..PER_WRITER {
                    c.record(skill(format!("{rank}-{n}"))).unwrap();
                }
                c.flush().unwrap();
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap();
    }

    let records = scratch.open().snapshot().unwrap().records().unwrap();
    assert_eq!(records.len() as u64, u64::from(WRITERS) * PER_WRITER);
    let ids: HashSet<_> = records.iter().map(|r| r.id).collect();
    assert_eq!(ids.len(), records.len());
}

#[test]
fn a_reader_during_ingest_sees_only_whole_batches() {
    const BATCH: u64 = 25;
    const BATCHES: u64 = 8;
    let scratch = Scratch::new();
    let root = scratch.dir.path().to_path_buf();
    let done = Arc::new(AtomicBool::new(false));

    let writers: Vec<_> = (0..4u32)
        .map(|rank| {
            let root = root.clone();
            thread::spawn(move || {
                let db = Database::open(&root, Config::default()).unwrap();
                let mut c = db
                    .collector(&WriterIdentity::new("exp", "job", "node", rank))
                    .unwrap();
                for batch in 0..BATCHES {
                    for n in 0..BATCH {
                        c.record(skill(format!("{rank}-{batch}-{n}"))).unwrap();
                    }
                    c.flush().unwrap();
                }
            })
        })
        .collect();

    let reader = {
        let (root, done) = (root.clone(), Arc::clone(&done));
        thread::spawn(move || {
            let db = Database::open(&root, Config::default()).unwrap();
            let mut observed = 0;
            while !done.load(Ordering::Acquire) {
                let count = db.snapshot().unwrap().records().unwrap().len() as u64;
                assert_eq!(count % BATCH, 0, "a partly published batch was visible");
                observed += 1;
            }
            observed
        })
    };
    for writer in writers {
        writer.join().unwrap();
    }
    done.store(true, Ordering::Release);
    reader.join().unwrap();
    assert_eq!(
        scratch.open().snapshot().unwrap().records().unwrap().len() as u64,
        4 * BATCH * BATCHES
    );
}

#[test]
fn spooled_microsegments_are_invisible_until_an_aggregator_publishes_them() {
    let scratch = Scratch::new();
    let db = scratch.open();
    for rank in 0..3 {
        let mut c = db
            .collector_to(
                &WriterIdentity::new("exp", "job", "node", rank),
                Destination::Spool,
            )
            .unwrap();
        for n in 0..10 {
            c.record(skill(format!("{rank}-{n}"))).unwrap();
        }
        c.flush().unwrap();
    }
    assert!(db.snapshot().unwrap().records().unwrap().is_empty());

    let report = Aggregator::new(&db).drain("node-a").unwrap();
    assert_eq!(report.microsegments, 3);
    assert_eq!(report.records, 30);
    assert_eq!(db.snapshot().unwrap().records().unwrap().len(), 30);
}

#[test]
fn draining_the_spool_twice_publishes_nothing_new() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = db
        .collector_to(
            &WriterIdentity::new("exp", "job", "node", 0),
            Destination::Spool,
        )
        .unwrap();
    for n in 0..5 {
        c.record(skill(format!("{n}"))).unwrap();
    }
    c.flush().unwrap();
    let aggregator = Aggregator::new(&db);
    aggregator.drain("node-a").unwrap();
    let second = aggregator.drain("node-a").unwrap();
    assert_eq!(second.microsegments, 0);
    assert_eq!(db.snapshot().unwrap().records().unwrap().len(), 5);
}
