// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: attempts at one task instance from one starting world form an
//! episode family, so group-relative questions are lookups rather than joins.
#![allow(clippy::unwrap_used)]

mod common;

use common::{attempt, coding_task, state, Scratch};
use splinter_expdb::model::{family_key, Outcome};
use splinter_expdb::WriterIdentity;

fn identity(rank: u32) -> WriterIdentity {
    WriterIdentity::new("exp", "job", "node", rank)
}

#[test]
fn every_attempt_at_an_instance_is_in_its_family_whichever_writer_ran_it() {
    let scratch = Scratch::new();
    let db = scratch.open();
    for (rank, (policy, outcome)) in [
        ("a", Outcome::Fail),
        ("b", Outcome::Pass),
        ("c", Outcome::Pass),
        ("d", Outcome::Fail),
    ]
    .into_iter()
    .enumerate()
    {
        let mut c = db.collector(&identity(rank as u32)).unwrap();
        attempt(&mut c, 42, policy, 2, outcome);
        c.flush().unwrap();
    }
    let (_, instance, initial) = coding_task(42);
    let key = family_key(&instance.id().unwrap(), &initial.id().unwrap());

    let family = db.snapshot().unwrap().family(&key).unwrap().unwrap();
    assert_eq!(family.attempts.len(), 4);
    assert_eq!(family.task_instance, instance.id().unwrap());
    assert_eq!(family.initial_state, state("repo@abc123").id().unwrap());
    let mut policies: Vec<_> = family
        .attempts
        .iter()
        .map(|a| a.policy.name.as_str())
        .collect();
    policies.sort_unstable();
    assert_eq!(policies, ["a", "b", "c", "d"]);
}

#[test]
fn pass_rate_and_reward_variance_describe_how_informative_a_family_is() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = db.collector(&identity(0)).unwrap();
    for outcome in [Outcome::Pass, Outcome::Pass, Outcome::Pass, Outcome::Fail] {
        attempt(&mut c, 1, "p", 1, outcome);
    }
    // A solved-every-time family and an aborted attempt, which proves nothing.
    attempt(&mut c, 2, "p", 1, Outcome::Pass);
    attempt(&mut c, 2, "p", 1, Outcome::Pass);
    attempt(&mut c, 3, "p", 1, Outcome::Aborted);
    c.flush().unwrap();

    let families = db.snapshot().unwrap().families().unwrap();
    let by_instance = |n: u64| {
        let (_, instance, _) = coding_task(n);
        families
            .iter()
            .find(|f| f.task_instance == instance.id().unwrap())
            .unwrap()
    };
    let mixed = by_instance(1);
    assert_eq!(mixed.pass_rate(), Some(0.75));
    assert!((mixed.reward_variance().unwrap() - 0.1875).abs() < 1e-12);
    assert!(mixed.has_both_outcomes());
    assert_eq!(by_instance(2).reward_variance(), Some(0.0));
    assert!(!by_instance(2).has_both_outcomes());
    // Nothing was measured, so nothing is reported - not a zero.
    assert_eq!(by_instance(3).pass_rate(), None);
    assert_eq!(by_instance(3).reward_variance(), None);
}

#[test]
fn families_where_some_attempts_pass_and_some_fail_can_be_listed() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = db.collector(&identity(0)).unwrap();
    attempt(&mut c, 1, "p", 1, Outcome::Pass);
    attempt(&mut c, 1, "q", 1, Outcome::Fail);
    attempt(&mut c, 2, "p", 1, Outcome::Pass);
    attempt(&mut c, 2, "q", 1, Outcome::Pass);
    c.flush().unwrap();

    let mixed = db.snapshot().unwrap().mixed_outcome_families().unwrap();
    assert_eq!(mixed.len(), 1);
    assert_eq!(mixed[0].attempts.len(), 2);
}
