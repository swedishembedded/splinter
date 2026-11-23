// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements restart-safe controllers for long-running
// automation, with fenced leases and an auditable event trail, for its
// clients. If your team needs expertise in durable workflow state, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The factory's run state: leases and fencing, optimistic stage
//! transitions, and the slot and attempt ledger of rollouts.

#![allow(clippy::unwrap_used)]

use splinter_core::digest::Digest;
use splinter_store::factory::runstate::{
    AttemptState, RunState, RunStateError, SlotKey, SlotState, StageOutcome, StageState,
};

fn digest(label: &str) -> Digest {
    Digest::sha256_of(label.as_bytes())
}

#[test]
fn a_second_controller_cannot_take_a_live_lease_and_the_first_is_fenced_off_after_expiry() {
    let dir = tempfile::tempdir().unwrap();
    let first = RunState::open(dir.path()).unwrap();
    let second = RunState::open(dir.path()).unwrap();

    let fence1 = first.acquire("alpha", 10, 100).unwrap();
    assert!(matches!(
        second.acquire("beta", 10, 105),
        Err(RunStateError::LeaseHeld { .. })
    ));
    let renewed = first.acquire("alpha", 10, 105).unwrap();
    assert_eq!(renewed.token, fence1.token, "renewal keeps the token");

    let fence2 = second.acquire("beta", 10, 200).unwrap();
    assert!(fence2.token > fence1.token);

    let stale = first.upsert_pending(&fence1, &digest("s"), "s", "train", 201);
    assert!(matches!(stale, Err(RunStateError::StaleFence { .. })));
    second
        .upsert_pending(&fence2, &digest("s"), "s", "train", 201)
        .unwrap();
}

#[test]
fn stage_transitions_use_optimistic_versions() {
    let dir = tempfile::tempdir().unwrap();
    let state = RunState::open(dir.path()).unwrap();
    let fence = state.acquire("alpha", 10, 0).unwrap();
    let id = digest("stage");
    let pending = state
        .upsert_pending(&fence, &id, "train", "train", 1)
        .unwrap();
    assert_eq!((pending.state, pending.version), (StageState::Pending, 0));
    assert_eq!(
        state
            .upsert_pending(&fence, &id, "train", "train", 2)
            .unwrap(),
        pending
    );

    let running = state
        .transition(&fence, &id, 0, StageState::Running, StageOutcome::None, 3)
        .unwrap();
    assert_eq!(running.version, 1);
    let conflict = state.transition(
        &fence,
        &id,
        0,
        StageState::Failed,
        StageOutcome::Error("x".into()),
        4,
    );
    assert!(matches!(
        conflict,
        Err(RunStateError::Conflict {
            expected: 0,
            actual: 1,
            ..
        })
    ));

    let product = digest("product");
    let done = state
        .transition(
            &fence,
            &id,
            1,
            StageState::Completed,
            StageOutcome::Product(product.clone()),
            5,
        )
        .unwrap();
    assert_eq!(done.product, Some(product));
    assert_eq!(state.get(&id).unwrap().unwrap(), done);
    assert_eq!(state.list().unwrap().len(), 1);
}

#[test]
fn slots_are_ensured_idempotently_and_finish_once() {
    let dir = tempfile::tempdir().unwrap();
    let state = RunState::open(dir.path()).unwrap();
    let fence = state.acquire("alpha", 10, 0).unwrap();
    let tasks = vec!["t1".to_string(), "t2".to_string()];
    let stage = Digest::of(b"stage");
    assert_eq!(
        state
            .ensure_slots(&fence, &stage, "pass-a", &tasks, 2, 1)
            .unwrap(),
        4
    );
    assert_eq!(
        state
            .ensure_slots(&fence, &stage, "pass-a", &tasks, 2, 2)
            .unwrap(),
        0
    );
    assert_eq!(state.list_slots("pass-a").unwrap().len(), 4);
    assert_eq!(state.slot_namespaces(&stage).unwrap(), ["pass-a"]);

    let attempt = state
        .claim_next(&fence, "pass-a", r#"{"pid":7}"#, 3)
        .unwrap()
        .unwrap();
    assert_eq!(attempt.slot, SlotKey::new("pass-a", "t1", 0));
    assert_eq!(attempt.state, AttemptState::Launched);
    let slot = state.slot(&attempt.slot).unwrap().unwrap();
    assert_eq!(slot.state, SlotState::Running);

    let trial = digest("trial");
    state
        .record_attempt(
            &fence,
            attempt.id,
            AttemptState::Succeeded,
            Some(trial.clone()),
            4,
        )
        .unwrap();
    // A failed attempt returns the slot; the slot's every attempt is kept.
    let retry_slot = SlotKey::new("pass-a", "t1", 1);
    let failed = state
        .claim_next(&fence, "pass-a", "{}", 4)
        .unwrap()
        .unwrap();
    assert_eq!(failed.slot, retry_slot);
    state
        .record_attempt(&fence, failed.id, AttemptState::Failed, None, 4)
        .unwrap();
    let again = state
        .claim_next(&fence, "pass-a", "{}", 4)
        .unwrap()
        .unwrap();
    assert_eq!(again.slot, retry_slot);
    let ids: Vec<_> = state
        .attempts(&retry_slot)
        .unwrap()
        .iter()
        .map(|a| a.id)
        .collect();
    assert_eq!(ids, [failed.id, again.id]);
    state.finish_slot(&fence, &attempt.slot, &trial, 5).unwrap();
    state.finish_slot(&fence, &attempt.slot, &trial, 6).unwrap();
    let other = state.finish_slot(&fence, &attempt.slot, &digest("other"), 7);
    assert!(matches!(other, Err(RunStateError::SlotFinished { .. })));
    assert_eq!(
        state.slot(&attempt.slot).unwrap().unwrap().trial,
        Some(trial)
    );
}

#[test]
fn unfinished_attempts_are_listed_after_reopen_and_events_are_appended() {
    let dir = tempfile::tempdir().unwrap();
    let key;
    {
        let state = RunState::open(dir.path()).unwrap();
        let fence = state.acquire("alpha", 10, 0).unwrap();
        state
            .ensure_slots(&fence, &Digest::of(b"stage"), "p", &["t".to_string()], 1, 1)
            .unwrap();
        let attempt = state
            .claim_next(&fence, "p", r#"{"container":"c1"}"#, 2)
            .unwrap()
            .unwrap();
        key = attempt.slot.clone();
        state
            .record_attempt(&fence, attempt.id, AttemptState::Running, None, 3)
            .unwrap();
    }
    let state = RunState::open(dir.path()).unwrap();
    let unfinished = state.unfinished_attempts().unwrap();
    assert_eq!(unfinished.len(), 1);
    assert_eq!(unfinished[0].slot, key);
    assert_eq!(unfinished[0].backend, r#"{"container":"c1"}"#);

    let log = std::fs::read_to_string(dir.path().join("state/events.jsonl")).unwrap();
    let lines: Vec<serde_json::Value> = log
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert!(lines.len() >= 4, "lease, slots, claim and attempt events");
    assert!(lines
        .iter()
        .all(|l| l.get("kind").is_some() && l.get("at").is_some()));
}

#[test]
fn a_released_lease_is_taken_at_once_and_a_dead_holders_is_taken_over() {
    let dir = tempfile::tempdir().unwrap();
    let state = RunState::open(dir.path()).unwrap();
    let first = state.acquire("alpha", 600, 10).unwrap();
    state.release(&first, 11).unwrap();
    let second = state.acquire("beta", 600, 11).unwrap();
    assert!(second.token > first.token);
    // beta is gone (its caller knows); gamma takes over before the expiry.
    let third = state.take_over("gamma", "beta", 600, 12).unwrap();
    assert!(third.token > second.token);
    assert!(matches!(
        state.take_over("delta", "beta", 600, 13),
        Err(RunStateError::LeaseHeld { .. })
    ));
    assert!(matches!(
        state.release(&second, 14),
        Err(RunStateError::StaleFence { .. })
    ));
}
