// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: what an attempt is worth is what the strongest evidence decides,
//! not an average of opinions. A recipe can require evidence of at least a
//! rank, pair the passing and failing attempts of one task, and follow the
//! relations recorded between attempts.
#![allow(clippy::unwrap_used)]

mod common;

use common::{attempt, Scratch};
use splinter_expdb::analyze::TASK_COMPLETION;
use splinter_expdb::ingest::Collector;
use splinter_expdb::model::{Evaluation, EvaluatorRef, Outcome, Rel, Ruling, Target, Verdict};
use splinter_expdb::train::{DataRef, Recipe, SampleBody};
use splinter_expdb::{Database, RecordId, WriterIdentity};

fn collector(db: &Database) -> Collector {
    db.collector(&WriterIdentity::new("exp", "job", "node", 0))
        .unwrap()
}

/// A ranked verdict on an attempt, as a grader at `rank` would give it.
fn rule(attempt: RecordId, who: &str, ruling: Ruling, rank: u8) -> Evaluation {
    let score = if ruling == Ruling::Pass { 1.0 } else { 0.0 };
    Evaluation::new(
        Target::Record(attempt),
        EvaluatorRef::new(who, "1"),
        TASK_COMPLETION,
        score,
        1.0,
    )
    .with_verdict(Verdict::new(ruling, rank))
}

#[test]
fn an_attempts_reward_is_what_its_strongest_evidence_decides() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = collector(&db);
    // Judged a pass by two weak graders, failed by one strong one.
    let (_, outvoted) = attempt(&mut c, 1, "p", 2, Outcome::Pass);
    for who in ["judge-a", "judge-b"] {
        c.evaluate(rule(outvoted, who, Ruling::Pass, 1)).unwrap();
    }
    c.evaluate(rule(outvoted, "tests", Ruling::Fail, 3))
        .unwrap();
    // Two equally strong graders disagree: nothing is decided.
    let (_, tied) = attempt(&mut c, 1, "p", 2, Outcome::Pass);
    c.evaluate(rule(tied, "a", Ruling::Pass, 2)).unwrap();
    c.evaluate(rule(tied, "b", Ruling::Fail, 2)).unwrap();
    // Passed by the strongest grader.
    let (_, sound) = attempt(&mut c, 1, "p", 2, Outcome::Pass);
    c.evaluate(rule(sound, "tests", Ruling::Pass, 3)).unwrap();
    c.flush().unwrap();

    let snapshot = db.snapshot().unwrap();
    let rewards: std::collections::HashMap<_, _> = snapshot
        .families()
        .unwrap()
        .into_iter()
        .flat_map(|f| f.attempts)
        .map(|a| (a.attempt, a.reward))
        .collect();
    assert_eq!(
        rewards[&outvoted],
        Some(0.0),
        "the strong fail outweighs two weak passes"
    );
    assert_eq!(
        rewards[&tied], None,
        "a tie at the top rank is unmeasured, not zero"
    );
    assert_eq!(rewards[&sound], Some(1.0));
}

#[test]
fn a_recipe_can_require_evidence_of_a_rank() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = collector(&db);
    let (_, weak) = attempt(&mut c, 1, "p", 1, Outcome::Pass);
    let (_, strong) = attempt(&mut c, 1, "p", 1, Outcome::Pass);
    let (_, unjudged) = attempt(&mut c, 1, "p", 1, Outcome::Pass);
    c.evaluate(rule(weak, "judge", Ruling::Pass, 1)).unwrap();
    c.evaluate(rule(strong, "tests", Ruling::Pass, 3)).unwrap();
    c.flush().unwrap();
    let snapshot = db.snapshot().unwrap();

    let at_least = |rank: u8| {
        let plan = snapshot
            .compile(&Recipe::sft().episodes().min_rank(rank))
            .unwrap();
        plan.samples
            .iter()
            .map(|s| s.provenance)
            .collect::<Vec<_>>()
    };
    assert_eq!(at_least(3), vec![strong]);
    assert_eq!(
        at_least(1),
        vec![weak, strong],
        "stronger evidence also qualifies"
    );
    let without = snapshot.compile(&Recipe::sft().episodes()).unwrap();
    assert!(
        without.samples.iter().any(|s| s.provenance == unjudged),
        "with no rank required, an attempt the environment passed still counts"
    );
}

fn pair_of(sample: &splinter_expdb::train::Sample) -> (RecordId, RecordId) {
    match &sample.body {
        SampleBody::Preference {
            chosen: DataRef::Episode { attempt: c },
            rejected: DataRef::Episode { attempt: r },
            ..
        } => (*c, *r),
        other => panic!("not an episode preference: {other:?}"),
    }
}

#[test]
fn passing_and_failing_attempts_of_one_task_are_paired_at_equal_rank() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = collector(&db);
    let (_, pass_a) = attempt(&mut c, 1, "p", 1, Outcome::Pass);
    let (_, pass_b) = attempt(&mut c, 1, "p", 1, Outcome::Pass);
    let (_, fail_strong) = attempt(&mut c, 1, "p", 1, Outcome::Fail);
    let (_, fail_weak) = attempt(&mut c, 1, "p", 1, Outcome::Fail);
    let (_, other_task) = attempt(&mut c, 2, "p", 1, Outcome::Fail);
    c.evaluate(rule(pass_a, "tests", Ruling::Pass, 3)).unwrap();
    c.evaluate(rule(pass_b, "tests", Ruling::Pass, 3)).unwrap();
    c.evaluate(rule(fail_strong, "tests", Ruling::Fail, 3))
        .unwrap();
    c.evaluate(rule(fail_weak, "judge", Ruling::Fail, 1))
        .unwrap();
    c.evaluate(rule(other_task, "tests", Ruling::Fail, 3))
        .unwrap();
    c.flush().unwrap();
    let snapshot = db.snapshot().unwrap();

    let plan = snapshot.compile(&Recipe::dpo().by_task()).unwrap();
    let mut pairs: Vec<_> = plan.samples.iter().map(pair_of).collect();
    pairs.sort();
    assert_eq!(
        pairs,
        vec![(pass_a, fail_strong), (pass_b, fail_strong)],
        "each pass against each fail of the same rank, within one task"
    );

    let capped = snapshot
        .compile(&Recipe::dpo().by_task().max_pairs_per_task(1))
        .unwrap();
    assert_eq!(capped.samples.len(), 1);
    assert_eq!(
        capped.id().unwrap(),
        snapshot
            .compile(&Recipe::dpo().by_task().max_pairs_per_task(1))
            .unwrap()
            .id()
            .unwrap(),
        "the cap picks the same pair every time"
    );
}

#[test]
fn relations_recorded_between_attempts_become_samples() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = collector(&db);
    let (_, failed) = attempt(&mut c, 1, "p", 1, Outcome::Fail);
    let (_, retry) = attempt(&mut c, 1, "p", 1, Outcome::Pass);
    let (_, unrelated) = attempt(&mut c, 1, "p", 1, Outcome::Pass);
    c.link(retry, Rel::RetryOf, failed).unwrap();
    c.link(unrelated, Rel::VariantOf, failed).unwrap();
    c.flush().unwrap();
    let snapshot = db.snapshot().unwrap();

    let plan = snapshot.compile(&Recipe::relations(Rel::RetryOf)).unwrap();
    assert_eq!(plan.samples.len(), 1);
    match &plan.samples[0].body {
        SampleBody::Related { rel, from, to } => {
            assert_eq!(*rel, Rel::RetryOf);
            assert_eq!(*from, DataRef::Episode { attempt: retry });
            assert_eq!(*to, DataRef::Episode { attempt: failed });
        }
        other => panic!("not a relation sample: {other:?}"),
    }
    assert!(snapshot
        .compile(&Recipe::relations(Rel::CritiqueOf))
        .unwrap()
        .samples
        .is_empty());
}
