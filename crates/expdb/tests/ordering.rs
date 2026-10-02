// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: what several writers wrote is listed in the order it was written,
//! by the time on the record, and never in the order of their random writer
//! ids. The order is the same for every reader and every run.
#![allow(clippy::unwrap_used)]

mod common;

use std::sync::Arc;

use common::Scratch;
use splinter_expdb::analyze::EvalFilter;
use splinter_expdb::backend::PosixBackend;
use splinter_expdb::model::{Entity, Evaluation, EvaluatorRef, Target};
use splinter_expdb::{Config, Database, StepClock, WriterIdentity};

/// A handle on the database whose clock starts at `start`.
fn handle(scratch: &Scratch, start: u64) -> Database {
    Database::with_backend(
        Arc::new(PosixBackend::new(scratch.dir.path())),
        Config::default(),
        Arc::new(StepClock::new(start, 10)),
    )
    .unwrap()
}

#[test]
fn entities_are_listed_in_the_order_their_records_were_stamped() {
    let scratch = Scratch::new();
    // Three writers, whose ids are unrelated to when they wrote. Later in
    // real time does not mean later on the record: the middle one's clock is
    // the earliest.
    let clocks = [(0u32, 3_000u64), (1, 1_000), (2, 2_000)];
    let mut expected = Vec::new();
    for (rank, start) in clocks {
        let db = handle(&scratch, start);
        let mut collector = db
            .collector(&WriterIdentity::new("exp", "job", "node", rank))
            .unwrap();
        let id = collector
            .put_entity(&Entity::new("thing", serde_json::json!({ "rank": rank })))
            .unwrap();
        collector.flush().unwrap();
        expected.push((start, id));
    }
    expected.sort();
    let want: Vec<_> = expected.into_iter().map(|(_, id)| id).collect();

    let db = handle(&scratch, 9_000);
    let listed: Vec<_> = db
        .snapshot()
        .unwrap()
        .entity_ids("thing")
        .unwrap()
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    assert_eq!(listed, want);
    let entities: Vec<_> = db
        .snapshot()
        .unwrap()
        .entities("thing")
        .unwrap()
        .into_iter()
        .map(|e| e.id)
        .collect();
    assert_eq!(entities, want, "the same order whichever way they are read");
}

#[test]
fn evaluations_come_back_in_the_order_they_were_stamped() {
    let scratch = Scratch::new();
    let target = Target::Entity(splinter_expdb::ContentId::of(b"subject"));
    let mut by_time = Vec::new();
    for (rank, start) in [(0u32, 3_000u64), (1, 1_000), (2, 2_000)] {
        let db = handle(&scratch, start);
        let mut collector = db
            .collector(&WriterIdentity::new("exp", "job", "node", rank))
            .unwrap();
        let name = format!("grader-{rank}");
        collector
            .evaluate(Evaluation::new(
                target,
                EvaluatorRef::new(&name, "1"),
                "c",
                1.0,
                1.0,
            ))
            .unwrap();
        collector.flush().unwrap();
        by_time.push((start, name));
    }
    by_time.sort();
    let want: Vec<String> = by_time.into_iter().map(|(_, n)| n).collect();

    let found: Vec<String> = handle(&scratch, 9_000)
        .snapshot()
        .unwrap()
        .evaluations(&EvalFilter::new().target(target))
        .unwrap()
        .into_iter()
        .map(|v| v.evaluation.evaluator.name)
        .collect();
    assert_eq!(found, want);
}
