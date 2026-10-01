// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: an agent run is recorded as a graph through a small API; writers
//! need no coordination; memory is bounded; nothing is visible until it is
//! flushed; and a fork shares everything before the decision it branches at.
#![allow(clippy::unwrap_used)]

mod common;

use std::collections::HashSet;

use common::{coding_task, state, Scratch};
use splinter_expdb::model::{
    Action, Body, Content, Evaluation, EvaluatorRef, Outcome, PolicyRef, Record, RecordKind, Rel,
    Target,
};
use splinter_expdb::{Config, WriterIdentity};

fn identity(rank: u32) -> WriterIdentity {
    WriterIdentity::new("exp1", "job-7", "node-a", rank)
}

fn kinds(records: &[Record]) -> Vec<RecordKind> {
    records.iter().map(Record::kind).collect()
}

#[test]
fn one_attempt_is_recorded_as_task_decision_transition_and_outcome() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut collector = db.collector(&identity(0)).unwrap();
    let (definition, instance, initial) = coding_task(742);
    let policy = PolicyRef::new("qwen3", "ckpt-4");

    let mut run = collector
        .start_attempt(&definition, &instance, &initial, &policy, Some(11))
        .unwrap();
    let seen = run.observe(Content::text("3 tests failing")).unwrap();
    let decision = run
        .decision()
        .observation(seen)
        .commit(Action::new("grep", serde_json::json!({"q": "foo"})))
        .unwrap();
    let transition = run
        .transition(&decision, &state("repo@abc124"), Some(0.25), None)
        .unwrap();
    let ended = run.finish(Outcome::Pass).unwrap();
    collector.flush().unwrap();

    let records = db.snapshot().unwrap().records().unwrap();
    let kinds = kinds(&records);
    for expected in [
        RecordKind::TaskDefinition,
        RecordKind::TaskInstance,
        RecordKind::Family,
        RecordKind::Attempt,
        RecordKind::Observation,
        RecordKind::Decision,
        RecordKind::Transition,
        RecordKind::AttemptEnd,
    ] {
        assert!(kinds.contains(&expected), "missing {expected:?}");
    }
    let by_id = |id| records.iter().find(|r| r.id == id).unwrap();
    assert_eq!(by_id(decision.id).parent, Some(seen));
    assert_eq!(by_id(transition).parent, Some(decision.id));
    assert_eq!(by_id(ended).parent, Some(transition));
    // Every record of the attempt carries its family and instance.
    let attempt_records: Vec<_> = records.iter().filter(|r| r.attempt.is_some()).collect();
    assert!(attempt_records
        .iter()
        .all(|r| r.family.is_some() && r.task_instance == Some(instance.id().unwrap())));
}

#[test]
fn a_run_chains_decisions_through_its_transitions() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut collector = db.collector(&identity(0)).unwrap();
    let (definition, instance, initial) = coding_task(1);
    let mut run = collector
        .start_attempt(
            &definition,
            &instance,
            &initial,
            &PolicyRef::new("p", "1"),
            None,
        )
        .unwrap();
    let first = run
        .decision()
        .commit(Action::new("a", serde_json::Value::Null))
        .unwrap();
    let t1 = run.transition(&first, &state("s1"), None, None).unwrap();
    let second = run
        .decision()
        .commit(Action::new("b", serde_json::Value::Null))
        .unwrap();
    run.finish(Outcome::Fail).unwrap();
    collector.flush().unwrap();

    let records = db.snapshot().unwrap().records().unwrap();
    let second_record = records.iter().find(|r| r.id == second.id).unwrap();
    assert_eq!(second_record.parent, Some(t1));
    assert_eq!(second.state, state("s1").id().unwrap());
}

#[test]
fn attempts_at_one_instance_from_one_state_share_a_family_and_others_do_not() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let (definition, instance, initial) = coding_task(5);
    let (_, other_instance, _) = coding_task(6);
    let policy = PolicyRef::new("p", "1");
    // Two writers, as two nodes would run, with no coordination.
    let families: Vec<_> = (0..2)
        .map(|rank| {
            let mut c = db.collector(&identity(rank)).unwrap();
            let run = c
                .start_attempt(&definition, &instance, &initial, &policy, None)
                .unwrap();
            let family = run.family();
            run.finish(Outcome::Pass).unwrap();
            c.flush().unwrap();
            family
        })
        .collect();
    assert_eq!(families[0], families[1]);

    let mut c = db.collector(&identity(9)).unwrap();
    let other = c
        .start_attempt(&definition, &other_instance, &initial, &policy, None)
        .unwrap()
        .family();
    assert_ne!(other, families[0]);
}

#[test]
fn restarting_a_writer_with_the_same_identity_never_reuses_a_record_id() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut ids = HashSet::new();
    for _ in 0..3 {
        let mut c = db.collector(&identity(0)).unwrap();
        for n in 0..20 {
            assert!(ids.insert(c.record(Body::Skill(skill(n))).unwrap()));
        }
        c.flush().unwrap();
    }
    assert_eq!(db.snapshot().unwrap().records().unwrap().len(), 60);
}

fn skill(n: u64) -> splinter_expdb::model::Skill {
    splinter_expdb::model::Skill {
        name: format!("skill {n}"),
        description: String::new(),
        trigger: String::new(),
        action_pattern: String::new(),
        expected_effect: String::new(),
        parents: vec![],
        prerequisites: vec![],
    }
}

#[test]
fn nothing_is_visible_before_a_flush_and_everything_after() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = db.collector(&identity(0)).unwrap();
    for n in 0..5 {
        c.record(Body::Skill(skill(n))).unwrap();
    }
    assert!(db.snapshot().unwrap().records().unwrap().is_empty());
    c.flush().unwrap();
    assert_eq!(db.snapshot().unwrap().records().unwrap().len(), 5);
}

#[test]
fn a_full_buffer_is_sealed_and_published_instead_of_growing() {
    let scratch = Scratch::new();
    let db = scratch.open_with(Config {
        max_buffered_records: 10,
        ..Config::default()
    });
    let mut c = db.collector(&identity(0)).unwrap();
    for n in 0..25 {
        c.record(Body::Skill(skill(n))).unwrap();
    }
    assert_eq!(c.buffered(), 5);
    assert_eq!(db.snapshot().unwrap().records().unwrap().len(), 20);
}

#[test]
fn small_content_stays_in_the_record_and_large_content_goes_to_the_blob_store() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = db.collector(&identity(0)).unwrap();
    let small = c.content(b"short").unwrap();
    let large = c.content(&vec![b'x'; 100_000]).unwrap();
    assert!(matches!(small, Content::Text { .. }));
    let Content::Blob { blob } = large else {
        panic!("large content must be a blob")
    };
    c.flush().unwrap();
    let reader = splinter_expdb::blob::BlobStore::open(&db).unwrap();
    assert_eq!(reader.get(&blob).unwrap().len(), 100_000);
}

#[test]
fn a_fork_records_the_alternative_as_a_counterfactual_that_shares_the_prefix() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = db.collector(&identity(0)).unwrap();
    let (definition, instance, initial) = coding_task(9);
    let policy = PolicyRef::new("p", "1");

    let mut run = c
        .start_attempt(&definition, &instance, &initial, &policy, None)
        .unwrap();
    let seen = run.observe(Content::text("failing")).unwrap();
    let original = run
        .decision()
        .observation(seen)
        .commit(Action::new("edit", serde_json::Value::Null))
        .unwrap();
    run.finish(Outcome::Fail).unwrap();

    let mut branch = c.fork(&original, &policy).unwrap();
    let alternative = branch
        .decision()
        .observation(seen)
        .commit(Action::new("inspect", serde_json::Value::Null))
        .unwrap();
    branch.finish(Outcome::Pass).unwrap();
    c.flush().unwrap();

    let snapshot = db.snapshot().unwrap();
    let records = snapshot.records().unwrap();
    let set = records
        .iter()
        .find_map(|r| match &r.body {
            Body::CounterfactualSet(set) => Some((r.id, set.clone())),
            _ => None,
        })
        .unwrap();
    assert_eq!(set.1.origin_decision, original.id);
    assert_eq!(set.1.origin_state, original.state);
    assert_eq!(set.1.shared_prefix, Some(seen));
    // Both the original and the alternative decision hang off the shared prefix.
    let parent_of = |id| records.iter().find(|r| r.id == id).unwrap().parent;
    assert_eq!(parent_of(original.id), parent_of(alternative.id));
    let edges = snapshot.edges().unwrap();
    for decision in [original.id, alternative.id] {
        assert!(edges
            .iter()
            .any(|e| e.from == set.0 && e.rel == Rel::Alternative && e.to == decision));
    }
}

#[test]
fn an_evaluation_is_a_record_about_a_target_not_a_field_of_it() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = db.collector(&identity(0)).unwrap();
    let (definition, instance, initial) = coding_task(3);
    let run = c
        .start_attempt(
            &definition,
            &instance,
            &initial,
            &PolicyRef::new("p", "1"),
            None,
        )
        .unwrap();
    let attempt = run.attempt();
    run.finish(Outcome::Pass).unwrap();
    let evaluation = Evaluation::new(
        Target::Record(attempt),
        EvaluatorRef::new("pytest", "8"),
        "task_completion",
        1.0,
        1.0,
    );
    let id = c.evaluate(evaluation).unwrap();
    c.flush().unwrap();

    let records = db.snapshot().unwrap().records().unwrap();
    let stored = records.iter().find(|r| r.id == id).unwrap();
    let Body::Evaluation(found) = &stored.body else {
        panic!("not an evaluation")
    };
    assert_eq!(found.target, Target::Record(attempt));
    assert_eq!(found.evaluator.name, "pytest");
}
