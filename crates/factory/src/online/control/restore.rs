// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements online reinforcement-learning controllers
// that never train a group twice or lose one they produced, for its clients.
// If your team needs expertise in asynchronous policy optimisation, you can
// procure our services by sending an email to info@swedishembedded.com.

//! A controller rebuilt from its ledger and the learner's state.

use std::collections::HashSet;

use splinter_core::factory::group::{GroupKey, GroupState, RolloutState};
use splinter_store::factory::runstate::GroupEntry;

use super::super::seams::{LearnerState, TaskSelector};
use super::super::Settings;
use super::{reasons, scored_rewards, trainable, Change, ControlError, Controller, Live, Rollout};

impl Controller {
    /// The controller of a run resumed from its ledger and the learner's
    /// state, with the changes that bring the ledger up to date (groups the
    /// learner committed, groups completed or dropped while it was down).
    /// Running rollouts are started again; scored, sent and applied groups
    /// the learner has not committed are sent again. Reward and rollout
    /// statistics are rebuilt from the ledger; backpressure time and the
    /// step estimate start afresh.
    ///
    /// # Errors
    /// The settings are invalid, or the ledger contradicts itself.
    pub fn restore(
        settings: Settings,
        entries: Vec<GroupEntry>,
        learner: &LearnerState,
        selector: &mut dyn TaskSelector,
    ) -> Result<(Self, Vec<Change>), ControlError> {
        let mut control =
            Self::new(settings, learner.policy_version).map_err(|e| ControlError(e.to_string()))?;
        let committed: HashSet<&GroupKey> = learner.committed.iter().collect();
        let mut changes = Vec::new();
        for entry in entries {
            control.restore_group(entry, &committed, selector, &mut changes)?;
        }
        Ok((control, changes))
    }

    fn restore_group(
        &mut self,
        entry: GroupEntry,
        committed: &HashSet<&GroupKey>,
        selector: &mut dyn TaskSelector,
        changes: &mut Vec<Change>,
    ) -> Result<(), ControlError> {
        let GroupEntry { group, rollouts } = entry;
        let size = usize::try_from(group.size).unwrap_or(usize::MAX);
        if rollouts.len() != size || rollouts.iter().zip(0u32..).any(|(r, i)| r.index != i) {
            return Err(ControlError(format!(
                "group {} records {} rollouts for size {}",
                group.key,
                rollouts.len(),
                group.size
            )));
        }
        self.next_seq = self.next_seq.max(group.seq + 1);
        self.stats.groups_started += 1;
        let mut live = Live {
            key: group.key.clone(),
            task: group.task.clone(),
            state: group.state,
            rollouts: Vec::with_capacity(size),
            tokens: 0,
            queued: false,
        };
        for record in rollouts {
            self.stats.rollout_failures += u64::from(record.failures);
            if record.state == RolloutState::Abandoned {
                self.stats.rollouts_abandoned += 1;
            }
            if let Some(outcome) = &record.outcome {
                self.count_finished(outcome);
                live.tokens += outcome.tokens;
            }
            live.rollouts.push(Rollout {
                // A rollout running when the controller stopped runs again.
                state: match record.state {
                    RolloutState::Running => RolloutState::Pending,
                    other => other,
                },
                failures: record.failures,
                outcome: record.outcome,
            });
        }
        let seq = group.seq;
        let state = group.state;
        if state != GroupState::Pending && state != GroupState::Running {
            self.stats.groups_scored += 1;
            selector.observe(&live.task, &scored_rewards(&live.rollouts));
            self.ingest_rewards(&live.rollouts);
        }
        if let Some(staleness) = group.staleness {
            self.stats.groups_sent += 1;
            *self.stats.staleness.entry(staleness).or_default() += 1;
        }
        match state {
            GroupState::Filtered => {
                self.count_filtered(group.reason.as_deref().unwrap_or(""));
                return Ok(());
            }
            GroupState::Stale => {
                if group.reason.as_deref() == Some(reasons::STALE_AT_LEARNER) {
                    self.stats.dropped_by_learner += 1;
                } else {
                    self.stats.dropped_stale += 1;
                }
                return Ok(());
            }
            GroupState::Committed => {
                self.stats.groups_applied += 1;
                self.stats.groups_committed += 1;
                return Ok(());
            }
            GroupState::Applied => self.stats.groups_applied += 1,
            _ => {}
        }
        let complete = live.rollouts.iter().all(|r| r.state.is_terminal());
        if matches!(state, GroupState::Sent | GroupState::Applied) {
            live.tokens = trainable(&live.rollouts).map(|(_, o)| o.tokens).sum();
        }
        // A group recorded sent may be queued, applied or lost at the
        // learner; its resend's answer decides whether it counts.
        let queued = matches!(
            state,
            GroupState::Pending | GroupState::Running | GroupState::Scored
        );
        if queued {
            live.queued = true;
            self.stats.queue_tokens += live.tokens;
        }
        for (index, rollout) in (0u32..).zip(&live.rollouts) {
            if !rollout.state.is_terminal() {
                self.ready.push_back((seq, index));
            }
        }
        self.seq_of.insert(live.key.clone(), seq);
        self.live.insert(seq, live);
        match state {
            GroupState::Pending | GroupState::Running if complete => {
                changes.extend(self.score(seq, selector));
            }
            GroupState::Scored if committed.contains(&group.key) => {
                self.stats.groups_sent += 1;
                self.walk(seq, GroupState::Committed, None, changes);
            }
            GroupState::Scored => changes.extend(self.decide(seq)),
            GroupState::Sent | GroupState::Applied if committed.contains(&group.key) => {
                self.walk(seq, GroupState::Committed, None, changes);
            }
            GroupState::Sent | GroupState::Applied => {
                self.send_queue.insert(seq);
                self.resend.insert(seq);
            }
            _ => {}
        }
        Ok(())
    }
}
