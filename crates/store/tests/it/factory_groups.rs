// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements restart-safe controllers for long-running
// automation, with fenced leases and an auditable event trail, for its
// clients. If your team needs expertise in durable workflow state, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The group ledger of an online run: every transition durable and fenced,
//! a finished rollout never started again, records kept exactly until the
//! group is terminal.

#![allow(clippy::unwrap_used)]

use splinter_core::factory::group::{
    GroupKey, GroupState, RolloutOutcome, RolloutState, VersionSpan,
};
use splinter_store::factory::runstate::{NewGroup, RunState, RunStateError};

fn outcome(reward: f64) -> RolloutOutcome {
    RolloutOutcome {
        reward: Some(reward),
        tokens: 10,
        span: VersionSpan::at(0),
        exact: true,
        truncated: false,
        truncation: None,
        behaviour: None,
        session: None,
    }
}

#[test]
fn a_group_is_recorded_through_its_life_and_reopened_as_it_was_left() {
    let dir = tempfile::tempdir().unwrap();
    let state = RunState::open(dir.path()).unwrap();
    let fence = state.acquire("alpha", 60, 0).unwrap();
    let key = GroupKey::numbered("pass", 0).unwrap();

    assert!(state
        .create_group(
            &fence,
            &NewGroup {
                key: &key,
                seq: 0,
                task: "task-a",
                size: 2,
                epoch: 0
            },
            1
        )
        .unwrap());
    assert!(
        !state
            .create_group(
                &fence,
                &NewGroup {
                    key: &key,
                    seq: 0,
                    task: "task-a",
                    size: 2,
                    epoch: 0
                },
                2
            )
            .unwrap(),
        "creating it again is a no-op"
    );
    assert!(
        state
            .create_group(
                &fence,
                &NewGroup {
                    key: &key,
                    seq: 0,
                    task: "task-b",
                    size: 2,
                    epoch: 0
                },
                2
            )
            .is_err(),
        "the same key for another group is refused"
    );

    state.start_rollout(&fence, &key, 0, 3).unwrap();
    state.start_rollout(&fence, &key, 1, 3).unwrap();
    state
        .fail_rollout(&fence, &key, 1, false, "worker lost", 4)
        .unwrap();
    state.start_rollout(&fence, &key, 1, 5).unwrap();
    state
        .finish_rollout(&fence, &key, 0, &outcome(1.0), "{\"r\":0}\n", 6)
        .unwrap();
    state
        .finish_rollout(&fence, &key, 1, &outcome(0.0), "{\"r\":1}\n", 6)
        .unwrap();
    assert!(
        matches!(
            state.start_rollout(&fence, &key, 0, 7),
            Err(RunStateError::IllegalTransition { .. })
        ),
        "a finished rollout is never started again"
    );
    assert!(state
        .finish_rollout(&fence, &key, 0, &outcome(0.5), "", 7)
        .is_err());

    for to in [GroupState::Scored, GroupState::Sent] {
        state
            .set_group_state(&fence, &key, to, None, Some(2), 8)
            .unwrap();
    }
    assert!(
        state
            .set_group_state(
                &fence,
                &key,
                GroupState::Filtered,
                Some("zero-variance"),
                None,
                9
            )
            .is_err(),
        "sent cannot become filtered"
    );
    drop(state);

    let reopened = RunState::open(dir.path()).unwrap();
    let entries = reopened.group_entries().unwrap();
    assert_eq!(entries.len(), 1);
    let entry = &entries[0];
    assert_eq!(
        (
            entry.group.state,
            entry.group.seq,
            entry.group.task.as_str(),
            entry.group.staleness
        ),
        (GroupState::Sent, 0, "task-a", Some(2))
    );
    let rollouts: Vec<_> = entry
        .rollouts
        .iter()
        .map(|r| {
            (
                r.index,
                r.state,
                r.failures,
                r.outcome.as_ref().and_then(|o| o.reward),
            )
        })
        .collect();
    assert_eq!(
        rollouts,
        [
            (0, RolloutState::Finished, 0, Some(1.0)),
            (1, RolloutState::Finished, 1, Some(0.0))
        ]
    );
    assert_eq!(
        reopened.group_records(&key).unwrap(),
        [
            (0, "{\"r\":0}\n".to_string()),
            (1, "{\"r\":1}\n".to_string())
        ]
    );

    let fence = reopened.acquire("alpha", 60, 10).unwrap();
    for to in [GroupState::Applied, GroupState::Committed] {
        reopened
            .set_group_state(&fence, &key, to, None, None, 11)
            .unwrap();
    }
    assert!(
        reopened.group_records(&key).unwrap().is_empty(),
        "a committed group's records are dropped"
    );
    assert_eq!(
        reopened.group_entries().unwrap()[0].rollouts[0].outcome,
        Some(outcome(1.0))
    );
}

#[test]
fn a_rollout_abandoned_past_its_budget_ends_without_an_outcome_and_a_stale_fence_changes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let state = RunState::open(dir.path()).unwrap();
    let old = state.acquire("alpha", 10, 0).unwrap();
    let key = GroupKey::numbered("pass", 3).unwrap();
    state
        .create_group(
            &old,
            &NewGroup {
                key: &key,
                seq: 3,
                task: "t",
                size: 2,
                epoch: 0,
            },
            1,
        )
        .unwrap();
    state.start_rollout(&old, &key, 0, 1).unwrap();
    state
        .fail_rollout(&old, &key, 0, true, "verifier down", 2)
        .unwrap();
    let entry = &state.group_entries().unwrap()[0];
    assert_eq!(entry.group.state, GroupState::Running);
    assert_eq!(
        (
            entry.rollouts[0].state,
            entry.rollouts[0].failures,
            entry.rollouts[0].outcome.clone()
        ),
        (RolloutState::Abandoned, 1, None)
    );

    let new = state.acquire("beta", 10, 100).unwrap();
    assert!(matches!(
        state.start_rollout(&old, &key, 1, 101),
        Err(RunStateError::StaleFence { .. })
    ));
    assert_eq!(
        state.group_entries().unwrap()[0].rollouts[1].state,
        RolloutState::Pending
    );
    state.start_rollout(&new, &key, 1, 101).unwrap();
    state.start_rollout(&new, &key, 1, 102).unwrap(); // a resumed controller starts a running rollout again
}

#[test]
fn a_ledger_written_before_epochs_reads_its_groups_as_epoch_zero_and_records_epochs_after() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("state")).unwrap();
    {
        // The ledger's table as a run before epochs wrote it, with one group in it.
        let conn = rusqlite::Connection::open(dir.path().join("state/controller.db")).unwrap();
        conn.execute_batch(
            "CREATE TABLE groups (key TEXT PRIMARY KEY, seq INTEGER NOT NULL UNIQUE, task TEXT NOT NULL,
               size INTEGER NOT NULL, state TEXT NOT NULL, reason TEXT, staleness INTEGER, updated_at INTEGER NOT NULL);
             INSERT INTO groups (key, seq, task, size, state, updated_at) VALUES ('pass:0', 0, 't', 0, 'pending', 1);",
        )
        .unwrap();
    }
    let state = RunState::open(dir.path()).unwrap();
    let fence = state.acquire("alpha", 60, 0).unwrap();
    let later = GroupKey::numbered("pass", 1).unwrap();
    state
        .create_group(
            &fence,
            &NewGroup {
                key: &later,
                seq: 1,
                task: "t",
                size: 2,
                epoch: 3,
            },
            2,
        )
        .unwrap();
    let entries = state.group_entries().unwrap();
    let epochs: Vec<u64> = entries.iter().map(|e| e.group.epoch).collect();
    assert_eq!(epochs, [0, 3]);
    state
        .set_group_state(
            &fence,
            &later,
            GroupState::RolledBack,
            Some("epoch"),
            None,
            4,
        )
        .unwrap();
    assert!(
        state
            .set_group_state(&fence, &later, GroupState::Sent, None, None, 5)
            .is_err(),
        "a rolled-back group is never sent"
    );
}
