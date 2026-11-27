// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements online reinforcement-learning controllers
// that never train a group twice or lose one they produced, for its clients.
// If your team needs expertise in asynchronous policy optimisation, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: version pinning and epochs. A run that pins versions trains only
//! rollouts one policy version decoded from first request to last; a rollout
//! that straddles a publication is scored and counted, never trained. Every
//! group belongs to the epoch it was created in, and a group of another
//! epoch (one a rollback ended) is never sent: a controller resumed in a new
//! epoch rolls every such group that is not terminal back, records why, and
//! keeps the selector's statistics to its own epoch.

use splinter_core::factory::group::{GroupState, RolloutState, VersionSpan};
use splinter_factory::online::control::{reasons, Change, Controller};
use splinter_factory::online::seams::{LearnerState, RoundRobin, TaskSelector};
use splinter_store::factory::runstate::{GroupEntry, GroupRecord, GroupRolloutRecord};

use super::{admit_all, finished, key, outcome, settings, states, tasks};

#[test]
fn a_run_that_pins_versions_trains_no_rollout_that_straddles_a_publication() {
    for pinned in [false, true] {
        let mut selector = RoundRobin::new(tasks(1), 1).unwrap();
        let mut control = Controller::new(
            splinter_factory::online::Settings {
                pin_versions: pinned,
                max_staleness: 10,
                ..settings()
            },
            2,
        )
        .unwrap();
        admit_all(&mut control, 0, &mut selector);
        let mut straddles = outcome(Some(1.0), 30, 1, true);
        straddles.span = VersionSpan::new(1, 2).unwrap();
        control
            .rollout_ended(finished(0, 0, straddles), &mut selector)
            .unwrap();
        let changes = control
            .rollout_ended(
                finished(0, 1, outcome(Some(0.0), 30, 2, true)),
                &mut selector,
            )
            .unwrap();
        let stats = control.stats(0);
        if pinned {
            assert_eq!(
                states(&changes).last(),
                Some(&(GroupState::Filtered, Some(reasons::VERSION_MIXED))),
                "its one admitted rollout leaves no reward to compare"
            );
            assert_eq!(
                (stats.rollouts_version_mixed, stats.filtered_version_mixed),
                (1, 1)
            );
            assert!(control.next_send(0).is_none());
        } else {
            assert_eq!(states(&changes), [(GroupState::Scored, None)]);
            assert_eq!(control.next_send(0).unwrap().members.len(), 2);
            assert_eq!(stats.rollouts_version_mixed, 0);
        }
    }
}

/// A group of `epoch` in `state`, its two rollouts finished with rewards 0
/// and 1 at version 0.
fn entry(seq: u64, state: GroupState, epoch: u64) -> GroupEntry {
    GroupEntry {
        group: GroupRecord {
            key: key(seq),
            seq,
            task: "task-0".into(),
            size: 2,
            state,
            reason: None,
            staleness: matches!(state, GroupState::Sent | GroupState::Committed).then_some(0),
            updated_at: 0,
            epoch,
        },
        rollouts: (0..2)
            .map(|index| GroupRolloutRecord {
                index,
                state: RolloutState::Finished,
                failures: 0,
                outcome: Some(outcome(Some(f64::from(index)), 30, 0, true)),
                updated_at: 0,
            })
            .collect(),
    }
}

/// Counts what it observes.
struct Counting(RoundRobin, usize);

impl TaskSelector for Counting {
    fn select(&mut self, seq: u64) -> Option<String> {
        self.0.select(seq)
    }
    fn observe(&mut self, _task: &str, _rewards: &[f64]) {
        self.1 += 1;
    }
}

#[test]
fn a_controller_resumed_in_a_new_epoch_rolls_back_the_earlier_epochs_groups_and_never_sends_them() {
    let entries = vec![
        entry(0, GroupState::Committed, 0),
        entry(1, GroupState::Scored, 0),
        entry(2, GroupState::Sent, 0),
        entry(3, GroupState::Scored, 1),
    ];
    let mut selector = Counting(RoundRobin::new(tasks(1), 4).unwrap(), 0);
    let learner = LearnerState::default();
    let (mut control, changes) = Controller::restore(
        splinter_factory::online::Settings {
            epoch: 1,
            ..settings()
        },
        entries,
        &learner,
        &mut selector,
    )
    .unwrap();
    let rolled: Vec<_> = changes
        .iter()
        .filter_map(|c| match c {
            Change::State {
                key,
                to: GroupState::RolledBack,
                reason,
                ..
            } => Some((key.clone(), *reason)),
            _ => None,
        })
        .collect();
    assert_eq!(
        rolled,
        [
            (key(1), Some(reasons::ROLLED_BACK)),
            (key(2), Some(reasons::ROLLED_BACK))
        ],
        "the earlier epoch's groups not yet trained are rolled back; its committed group stays"
    );
    let draft = control.next_send(0).expect("the new epoch's group is sent");
    assert_eq!(draft.key, key(3));
    assert_eq!(control.stats(0).groups_rolled_back, 2);
    assert_eq!(
        selector.1, 1,
        "the selector learns only from its own epoch's groups"
    );
}

#[test]
fn another_stages_groups_in_the_runs_ledger_only_move_the_order_of_groups_on() {
    let mut other = entry(5, GroupState::Committed, 0);
    other.group.key = splinter_core::factory::group::GroupKey::numbered("other", 5).unwrap();
    let mut selector = RoundRobin::new(tasks(1), 7).unwrap();
    let (mut control, changes) = Controller::restore(
        settings(),
        vec![other],
        &LearnerState::default(),
        &mut selector,
    )
    .unwrap();
    assert!(changes.is_empty());
    assert_eq!(control.stats(0).groups_started, 0, "not this stage's group");
    let created = control.admit(0, &mut selector).unwrap();
    assert!(
        matches!(&created[0], Change::Create { seq: 6, .. }),
        "{created:?}"
    );
}
