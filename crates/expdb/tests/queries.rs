// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: a query is answered from the cheapest structure that can answer it.
//! Zone maps let a scan skip blocks, a persisted index avoids scanning at all,
//! the answer is the same either way, and semantic and text search merge
//! results across immutable shards.
#![allow(clippy::unwrap_used)]

mod common;

use common::{attempt, coding_task, rid, Scratch};
use splinter_expdb::model::{family_key, Body, Epistemic, Outcome, RecordKind, Skill};
use splinter_expdb::query::{Plan, Query};
use splinter_expdb::{Config, ContentId, Database, RecordId, WriterIdentity};

/// Twelve task instances, one attempt of four decisions each, in one segment
/// of many small blocks.
fn populated(scratch: &Scratch) -> Database {
    let db = scratch.open_with(Config {
        block_records: 10,
        ..Config::default()
    });
    let mut c = db
        .collector(&WriterIdentity::new("exp", "job", "node", 0))
        .unwrap();
    for task in 0..12 {
        attempt(
            &mut c,
            task,
            "p",
            4,
            if task % 2 == 0 {
                Outcome::Pass
            } else {
                Outcome::Fail
            },
        );
    }
    c.flush().unwrap();
    db
}

fn family_of(task: u64) -> ContentId {
    let (_, instance, initial) = coding_task(task);
    family_key(&instance.id().unwrap(), &initial.id().unwrap())
}

#[test]
fn a_scan_skips_blocks_whose_zone_cannot_match() {
    let scratch = Scratch::new();
    let db = populated(&scratch);
    let result = db
        .snapshot()
        .unwrap()
        .query(&Query::all().family(family_of(5)))
        .unwrap();

    assert_eq!(result.plan, Plan::Scan);
    assert!(!result.records.is_empty());
    assert!(result
        .records
        .iter()
        .all(|r| r.family == Some(family_of(5))));
    assert!(
        result.blocks_read * 2 < result.blocks_total,
        "read {} of {} blocks for one family of twelve",
        result.blocks_read,
        result.blocks_total
    );
}

#[test]
fn a_kind_predicate_skips_blocks_that_hold_none_of_that_kind() {
    let scratch = Scratch::new();
    let db = scratch.open_with(Config {
        block_records: 8,
        ..Config::default()
    });
    let mut c = db
        .collector(&WriterIdentity::new("exp", "job", "node", 0))
        .unwrap();
    for task in 0..6 {
        attempt(&mut c, task, "p", 3, Outcome::Pass);
    }
    // A run of skills lands in blocks of its own.
    for n in 0..40u64 {
        c.record(Body::Skill(Skill {
            name: format!("skill {n}"),
            description: String::new(),
            trigger: String::new(),
            action_pattern: String::new(),
            expected_effect: String::new(),
            parents: vec![],
            prerequisites: vec![],
        }))
        .unwrap();
    }
    c.flush().unwrap();

    let result = db
        .snapshot()
        .unwrap()
        .query(&Query::all().kind(RecordKind::Skill))
        .unwrap();
    assert_eq!(result.records.len(), 40);
    assert!(result.blocks_read < result.blocks_total);
}

#[test]
fn a_time_range_skips_blocks_outside_it() {
    let scratch = Scratch::new();
    let clock = std::sync::Arc::new(splinter_expdb::StepClock::new(1_000, 10));
    let db = Database::with_backend(
        std::sync::Arc::new(splinter_expdb::backend::PosixBackend::new(
            scratch.dir.path(),
        )),
        Config {
            block_records: 10,
            ..Config::default()
        },
        clock,
    )
    .unwrap();
    let mut c = db
        .collector(&WriterIdentity::new("exp", "job", "node", 0))
        .unwrap();
    for task in 0..8 {
        attempt(&mut c, task, "p", 3, Outcome::Pass);
    }
    c.flush().unwrap();
    let all = db.snapshot().unwrap().records().unwrap();
    let late = all.iter().map(|r| r.timestamp_ns).max().unwrap();

    let result = db
        .snapshot()
        .unwrap()
        .query(&Query::all().between(late - 50, late + 1))
        .unwrap();
    assert!(result.records.iter().all(|r| r.timestamp_ns >= late - 50));
    assert!(result.blocks_read < result.blocks_total);
}

#[test]
fn a_persisted_index_answers_a_query_by_reading_only_the_blocks_it_needs() {
    let scratch = Scratch::new();
    let db = populated(&scratch);
    db.build_indexes().unwrap();
    let snapshot = db.snapshot().unwrap();
    let result = snapshot.query(&Query::all().family(family_of(5))).unwrap();

    assert_eq!(result.plan, Plan::Index);
    assert!(result.blocks_read <= result.blocks_total / 2);
    assert!(result
        .records
        .iter()
        .all(|r| r.family == Some(family_of(5))));
}

#[test]
fn the_scan_and_the_index_give_the_same_answer() {
    let scratch = Scratch::new();
    let db = populated(&scratch);
    let queries = [
        Query::all().family(family_of(3)),
        Query::all().kind(RecordKind::Decision),
        Query::all()
            .kind(RecordKind::Decision)
            .kind(RecordKind::Transition)
            .family(family_of(7)),
        Query::all()
            .task_instance(coding_task(2).1.id().unwrap())
            .kind(RecordKind::AttemptEnd),
    ];
    let scanned: Vec<_> = {
        let snapshot = db.snapshot().unwrap();
        queries.iter().map(|q| snapshot.query(q).unwrap()).collect()
    };
    db.build_indexes().unwrap();
    let snapshot = db.snapshot().unwrap();
    for (query, expected) in queries.iter().zip(scanned) {
        let indexed = snapshot.query(query).unwrap();
        assert_eq!(indexed.plan, Plan::Index);
        assert_eq!(indexed.records, expected.records);
    }
}

#[test]
fn a_limit_returns_the_first_records_in_id_order() {
    let scratch = Scratch::new();
    let db = populated(&scratch);
    let result = db
        .snapshot()
        .unwrap()
        .query(&Query::all().kind(RecordKind::Decision).limit(3))
        .unwrap();
    assert_eq!(result.records.len(), 3);
    let ids: Vec<_> = result.records.iter().map(|r| r.id).collect();
    let mut sorted = ids.clone();
    sorted.sort();
    assert_eq!(ids, sorted);
}

#[test]
fn facts_and_hypotheses_are_told_apart_in_a_query() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = db
        .collector(&WriterIdentity::new("exp", "job", "node", 0))
        .unwrap();
    attempt(&mut c, 1, "p", 2, Outcome::Pass);
    c.record(Body::Skill(Skill {
        name: "inspect first".into(),
        description: String::new(),
        trigger: String::new(),
        action_pattern: String::new(),
        expected_effect: String::new(),
        parents: vec![],
        prerequisites: vec![],
    }))
    .unwrap();
    c.flush().unwrap();
    let snapshot = db.snapshot().unwrap();

    let hypotheses = snapshot
        .query(&Query::all().epistemic(Epistemic::Hypothesis))
        .unwrap();
    assert_eq!(hypotheses.records.len(), 1);
    let facts = snapshot
        .query(&Query::all().epistemic(Epistemic::Fact))
        .unwrap();
    assert!(facts
        .records
        .iter()
        .all(|r| r.body.epistemic() == Epistemic::Fact));
    assert!(facts.records.len() > 5);
}

fn ids(n: u64) -> Vec<RecordId> {
    (0..n).map(|i| rid(1, i)).collect()
}

#[test]
fn a_semantic_search_returns_the_nearest_records_across_every_shard() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = db
        .collector(&WriterIdentity::new("exp", "job", "node", 0))
        .unwrap();
    // Real records for the vectors to point at.
    let targets: Vec<RecordId> = (0..9).map(|n| c.record(skill_body(n)).unwrap()).collect();
    c.flush().unwrap();
    let vector = |n: usize| vec![(n as f32).cos(), (n as f32).sin()];
    for shard in 0..3 {
        let items: Vec<_> = (0..3)
            .map(|i| (targets[shard * 3 + i], vector(shard * 3 + i)))
            .collect();
        db.add_vectors("embed-v1", &items).unwrap();
    }

    let snapshot = db.snapshot().unwrap();
    let query = vec![(4.2f32).cos(), (4.2f32).sin()];
    let hits = snapshot.search_vectors("embed-v1", &query, 3).unwrap();
    assert_eq!(hits.len(), 3);
    assert_eq!(hits[0].0, targets[4]);
    assert!((hits[0].1 - (0.2f32).cos()).abs() < 1e-4);
    // Exact search: the same as ranking every vector by hand.
    let mut expected: Vec<_> = (0..9)
        .map(|n| (targets[n], (n as f32 - 4.2).cos()))
        .collect();
    expected.sort_by(|a, b| b.1.total_cmp(&a.1));
    let found: Vec<_> = hits.iter().map(|h| h.0).collect();
    assert_eq!(
        found,
        expected.iter().take(3).map(|e| e.0).collect::<Vec<_>>()
    );
}

fn skill_body(n: u64) -> Body {
    Body::Skill(Skill {
        name: format!("skill {n}"),
        description: String::new(),
        trigger: String::new(),
        action_pattern: String::new(),
        expected_effect: String::new(),
        parents: vec![],
        prerequisites: vec![],
    })
}

#[test]
fn vectors_are_kept_per_model_and_checked_for_size() {
    let scratch = Scratch::new();
    let db = scratch.open();
    db.add_vectors("a", &[(ids(1)[0], vec![1.0, 0.0])]).unwrap();
    assert!(
        db.add_vectors("a", &[(ids(2)[1], vec![1.0, 0.0, 0.0])])
            .is_err(),
        "mixed sizes in one batch of a model"
    );
    assert!(
        db.add_vectors("a", &[(ids(1)[0], vec![0.0, 0.0])]).is_err(),
        "a zero vector has no direction"
    );
    assert!(db
        .snapshot()
        .unwrap()
        .search_vectors("b", &[1.0, 0.0], 1)
        .unwrap()
        .is_empty());
}

#[test]
fn a_vector_for_a_record_that_is_not_in_the_snapshot_is_never_returned() {
    let scratch = Scratch::new();
    let db = scratch.open();
    db.add_vectors("a", &[(rid(9, 9), vec![1.0, 0.0])]).unwrap();
    assert!(db
        .snapshot()
        .unwrap()
        .search_vectors("a", &[1.0, 0.0], 5)
        .unwrap()
        .is_empty());
}

#[test]
fn merging_vector_shards_changes_no_answer() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = db
        .collector(&WriterIdentity::new("exp", "job", "node", 0))
        .unwrap();
    let targets: Vec<RecordId> = (0..6).map(|n| c.record(skill_body(n)).unwrap()).collect();
    c.flush().unwrap();
    for (n, id) in targets.iter().enumerate() {
        db.add_vectors("m", &[(*id, vec![n as f32 + 1.0, 1.0])])
            .unwrap();
    }
    let before = db
        .snapshot()
        .unwrap()
        .search_vectors("m", &[5.0, 1.0], 6)
        .unwrap();
    assert!(db.compact_vectors("m").unwrap().is_some());
    let after = db.snapshot().unwrap();
    assert_eq!(after.search_vectors("m", &[5.0, 1.0], 6).unwrap(), before);
}

#[test]
fn a_text_search_finds_records_by_their_words_across_shards() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = db
        .collector(&WriterIdentity::new("exp", "job", "node", 0))
        .unwrap();
    let targets: Vec<RecordId> = (0..4).map(|n| c.record(skill_body(n)).unwrap()).collect();
    c.flush().unwrap();
    db.add_text(&[(targets[0], "deadlock between two mutexes".to_owned())])
        .unwrap();
    db.add_text(&[(
        targets[1],
        "interrupt handler starves the scheduler".to_owned(),
    )])
    .unwrap();
    db.add_text(&[(
        targets[2],
        "Deadlock in the interrupt handler, deadlock again".to_owned(),
    )])
    .unwrap();

    let snapshot = db.snapshot().unwrap();
    let hits: Vec<RecordId> = snapshot
        .search_text("deadlock", 10)
        .unwrap()
        .into_iter()
        .map(|h| h.0)
        .collect();
    assert_eq!(
        hits,
        [targets[2], targets[0]],
        "more occurrences rank first"
    );
    let both: Vec<RecordId> = snapshot
        .search_text("deadlock interrupt", 1)
        .unwrap()
        .into_iter()
        .map(|h| h.0)
        .collect();
    assert_eq!(
        both,
        [targets[2]],
        "a record with both words beats records with one"
    );
    assert!(snapshot
        .search_text("nothing like this", 5)
        .unwrap()
        .is_empty());
}
