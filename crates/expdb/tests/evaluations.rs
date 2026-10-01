// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: an evaluation is a record about a target, not a field of it. Many
//! evaluators can judge one thing, a withdrawn evaluator stops counting
//! without anything being deleted, and what counts as an attempt's reward
//! follows the evaluations that still stand.
#![allow(clippy::unwrap_used)]

mod common;

use common::{attempt, coding_task, Scratch};
use splinter_expdb::analyze::EvalFilter;
use splinter_expdb::model::{family_key, Epistemic, Evaluation, EvaluatorRef, Outcome, Target};
use splinter_expdb::{Database, RecordId, WriterIdentity};

fn collector(db: &Database) -> splinter_expdb::ingest::Collector {
    db.collector(&WriterIdentity::new("exp", "job", "node", 0))
        .unwrap()
}

fn judge(attempt: RecordId, who: (&str, &str), score: f64, confidence: f64) -> Evaluation {
    Evaluation::new(
        Target::Record(attempt),
        EvaluatorRef::new(who.0, who.1),
        "task_completion",
        score,
        confidence,
    )
}

#[test]
fn several_evaluators_can_judge_one_target_and_each_judgement_is_kept() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = collector(&db);
    let (decisions, attempt_id) = attempt(&mut c, 1, "p", 2, Outcome::Pass);
    c.evaluate(judge(attempt_id, ("pytest", "8"), 1.0, 1.0).epistemic(Epistemic::Fact))
        .unwrap();
    c.evaluate(Evaluation::new(
        Target::Record(decisions[0].id),
        EvaluatorRef::new("critic", "2"),
        "reasoning_quality",
        0.73,
        0.61,
    ))
    .unwrap();
    c.evaluate(judge(attempt_id, ("human", "1"), 0.9, 0.8))
        .unwrap();
    c.flush().unwrap();
    let snapshot = db.snapshot().unwrap();

    let on_attempt = snapshot
        .evaluations(&EvalFilter::new().target(Target::Record(attempt_id)))
        .unwrap();
    assert_eq!(on_attempt.len(), 2);
    let on_decision = snapshot
        .evaluations(&EvalFilter::new().target(Target::Record(decisions[0].id)))
        .unwrap();
    assert_eq!(on_decision[0].evaluation.criterion, "reasoning_quality");
    let by_criterion = snapshot
        .evaluations(&EvalFilter::new().criterion("task_completion"))
        .unwrap();
    assert_eq!(by_criterion.len(), 2);
}

#[test]
fn a_withdrawn_evaluator_version_stops_counting_but_nothing_is_deleted() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = collector(&db);
    let (_, attempt_id) = attempt(&mut c, 1, "p", 1, Outcome::Fail);
    c.evaluate(judge(attempt_id, ("critic", "1"), 0.9, 0.5))
        .unwrap();
    c.evaluate(judge(attempt_id, ("critic", "2"), 0.2, 0.9))
        .unwrap();
    c.flush().unwrap();
    let before = db.snapshot().unwrap();
    let records_before = before.records().unwrap().len();

    c.retract(
        EvaluatorRef::new("critic", "1"),
        "it rated debugging sessions it never read",
    )
    .unwrap();
    c.flush().unwrap();
    let after = db.snapshot().unwrap();

    let standing = after
        .evaluations(&EvalFilter::new().target(Target::Record(attempt_id)))
        .unwrap();
    assert_eq!(standing.len(), 1);
    assert_eq!(standing[0].evaluation.evaluator.version, "2");
    // The withdrawn judgement is still there for audit, and the experience is untouched.
    let audit = after
        .evaluations(
            &EvalFilter::new()
                .target(Target::Record(attempt_id))
                .include_retracted(),
        )
        .unwrap();
    assert_eq!(audit.len(), 2);
    assert_eq!(
        after.records().unwrap().len(),
        records_before + 1,
        "only the retraction was added"
    );
    // The earlier snapshot still reads as it did.
    assert_eq!(before.evaluations(&EvalFilter::new()).unwrap().len(), 2);
}

#[test]
fn a_withdrawal_covers_judgements_written_after_it_too() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = collector(&db);
    let (_, attempt_id) = attempt(&mut c, 1, "p", 1, Outcome::Pass);
    c.retract(EvaluatorRef::new("critic", "1"), "bad").unwrap();
    c.evaluate(judge(attempt_id, ("critic", "1"), 1.0, 1.0))
        .unwrap();
    c.flush().unwrap();
    assert!(db
        .snapshot()
        .unwrap()
        .evaluations(&EvalFilter::new())
        .unwrap()
        .is_empty());
}

#[test]
fn evaluations_can_be_filtered_by_evaluator_confidence_and_how_far_they_are_believed() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = collector(&db);
    let (_, attempt_id) = attempt(&mut c, 1, "p", 1, Outcome::Pass);
    c.evaluate(judge(attempt_id, ("pytest", "8"), 1.0, 1.0).epistemic(Epistemic::Fact))
        .unwrap();
    c.evaluate(judge(attempt_id, ("critic", "2"), 0.7, 0.4).epistemic(Epistemic::Annotation))
        .unwrap();
    c.flush().unwrap();
    let snapshot = db.snapshot().unwrap();

    assert_eq!(
        snapshot
            .evaluations(&EvalFilter::new().evaluator("pytest"))
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        snapshot
            .evaluations(&EvalFilter::new().min_confidence(0.9))
            .unwrap()
            .len(),
        1
    );
    let facts = snapshot
        .evaluations(&EvalFilter::new().epistemic(Epistemic::Fact))
        .unwrap();
    assert_eq!(facts[0].evaluation.evaluator.name, "pytest");
}

#[test]
fn a_verifier_decides_an_attempts_reward_until_it_is_withdrawn() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = collector(&db);
    // The environment says the attempt failed; a verifier disagrees.
    let (_, attempt_id) = attempt(&mut c, 1, "p", 1, Outcome::Fail);
    c.evaluate(judge(attempt_id, ("verifier", "1"), 1.0, 1.0))
        .unwrap();
    c.flush().unwrap();
    let (_, instance, initial) = coding_task(1);
    let key = family_key(&instance.id().unwrap(), &initial.id().unwrap());

    let family = db.snapshot().unwrap().family(&key).unwrap().unwrap();
    assert_eq!(family.attempts[0].reward, Some(1.0));
    assert_eq!(
        family.attempts[0].outcome,
        Some(Outcome::Fail),
        "what the environment said is still reported"
    );

    c.retract(
        EvaluatorRef::new("verifier", "1"),
        "it was checking the wrong file",
    )
    .unwrap();
    c.flush().unwrap();
    let family = db.snapshot().unwrap().family(&key).unwrap().unwrap();
    assert_eq!(family.attempts[0].reward, Some(0.0));
}
