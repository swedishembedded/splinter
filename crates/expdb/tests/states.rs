// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: a world's history is shared pages, so a branch costs only what it
//! adds, and a state says how far it can be trusted to replay.
#![allow(clippy::unwrap_used)]

mod common;

use std::collections::BTreeMap;

use common::{coding_task, Scratch};
use splinter_expdb::blob::BlobStore;
use splinter_expdb::model::{Body, ContextLog};
use splinter_expdb::model::{Outcome, PolicyRef, ReproLevel, State};
use splinter_expdb::{ContentId, WriterIdentity};

fn event(n: usize) -> Vec<u8> {
    format!("event {n}: the tool printed a line of output").into_bytes()
}

#[test]
fn the_context_at_any_step_is_rebuilt_from_pages() {
    let scratch = Scratch::new();
    let mut store = BlobStore::open(&scratch.open()).unwrap();
    let mut log = ContextLog::new(8);
    for n in 0..50 {
        log.append(&mut store, &event(n)).unwrap();
    }
    let head = log.commit(&mut store).unwrap();

    for upto in [0, 1, 7, 8, 9, 31, 50] {
        let events = ContextLog::read(&store, head, upto).unwrap();
        assert_eq!(
            events,
            (0..upto).map(event).collect::<Vec<_>>(),
            "first {upto} events"
        );
    }
    assert!(ContextLog::read(&store, head, 51).is_err());
}

#[test]
fn a_branch_of_a_context_stores_only_what_it_adds() {
    let scratch = Scratch::new();
    let mut store = BlobStore::open(&scratch.open()).unwrap();
    let mut log = ContextLog::new(8);
    for n in 0..200 {
        log.append(&mut store, &event(n)).unwrap();
    }
    let trunk = log.commit(&mut store).unwrap();
    let written = store.stats().chunks_written;

    // Ten branches from the same point, each three events long.
    let mut heads = Vec::new();
    for branch in 0..10 {
        let mut fork = ContextLog::load(&store, trunk).unwrap();
        for n in 0..3 {
            fork.append(&mut store, &event(1000 + branch * 10 + n))
                .unwrap();
        }
        heads.push(fork.commit(&mut store).unwrap());
    }
    // Each branch: its tail page and its head. The 25 full pages are shared.
    let added = store.stats().chunks_written - written;
    assert!(added <= 10 * 3, "ten branches stored {added} new objects");
    for (branch, head) in heads.iter().enumerate() {
        let events = ContextLog::read(&store, *head, 203).unwrap();
        assert_eq!(events[..200], (0..200).map(event).collect::<Vec<_>>()[..]);
        assert_eq!(events[200], event(1000 + branch * 10));
    }
}

#[test]
fn a_state_records_how_reproducible_it_is_and_what_is_not() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = db
        .collector(&WriterIdentity::new("exp", "job", "node", 0))
        .unwrap();
    let (definition, instance, initial) = coding_task(1);
    let mut live = State::new(
        BTreeMap::from([("browser".to_owned(), ContentId::of(b"dom"))]),
        ReproLevel::Live,
    );
    live.non_reproducible = vec!["the live web page".into()];

    let mut run = c
        .start_attempt(
            &definition,
            &instance,
            &initial,
            &PolicyRef::new("p", "1"),
            None,
        )
        .unwrap();
    let d = run
        .decision()
        .commit(splinter_expdb::model::Action::new(
            "browse",
            serde_json::Value::Null,
        ))
        .unwrap();
    run.transition(&d, &live, None, None).unwrap();
    run.finish(Outcome::Pass).unwrap();
    c.flush().unwrap();

    let records = db.snapshot().unwrap().records().unwrap();
    let stored = records
        .iter()
        .find_map(|r| match &r.body {
            Body::State(s) if s.repro == ReproLevel::Live => Some(s.clone()),
            _ => None,
        })
        .unwrap();
    assert_eq!(stored.non_reproducible, ["the live web page"]);
    // A counterfactual is only as trustworthy as the weakest state under it.
    assert_eq!(
        ReproLevel::weakest([ReproLevel::Exact, ReproLevel::Live, ReproLevel::Snapshot]),
        Some(ReproLevel::Live)
    );
    assert_eq!(ReproLevel::weakest([]), None);
}
