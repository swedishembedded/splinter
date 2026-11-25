// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements online reinforcement-learning controllers
// that never train a group twice or lose one they produced, for its clients.
// If your team needs expertise in asynchronous policy optimisation, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The controller core: no I/O, no clock of its own. Each call takes the
//! time and returns the [`Change`]s the caller must make durable, in order,
//! before acting on them.
//!
//! The rules:
//!
//! * **Admission.** A rollout starts only while fewer than `rollout_slots`
//!   run. A new group starts only while the tokens produced and not yet
//!   applied, plus the rollouts admitted and not ended at the mean tokens of
//!   a finished rollout, stay below the allowance `max(max_staleness, 1) *
//!   tokens_per_step` (before any rollout has finished, a rollout is assumed
//!   to sample `rollout_tokens`). A started group's remaining rollouts need
//!   only a slot, so a partial group never waits on the allowance it counts
//!   in. A running rollout is never stopped by backpressure.
//! * **Tokens per step** are the learner's step budget at first, then the
//!   mean of its last `step_window` step reports, never below the budget.
//! * **Scoring.** A group is scored when every rollout has ended. A rollout
//!   with no reward, or abandoned after its infrastructure failed more than
//!   `max_failures` times, is left out of the baseline, never read as zero.
//!   No reward at all: filtered `unscored`. Rewards that do not differ
//!   ([`splinter_core::rewards::differ`]): filtered `zero-variance`, counted in
//!   the reward statistics. Rewards of the trainable rollouts (exact ids and
//!   behaviour log-probabilities, not cut at a length limit) that do not
//!   differ: filtered `not-exact`; an inexact or truncated rollout is never
//!   sent.
//! * **Staleness.** A group whose oldest policy version among its exact
//!   rollouts is more than `max_staleness` behind the learner is dropped
//!   (checked when it is scored and whenever the learner publishes).
//! * **Tokens queued** are what the learner will hold: a group's finished
//!   rollouts while it runs, its exact scored rollouts once it is queued to
//!   send and once the learner accepted it, nothing once it is applied,
//!   filtered or dropped, nor while the learner may already have applied it
//!   (it answered `duplicate`, or a resumed controller has not yet heard
//!   from it about a group recorded sent).
//! * **Sending.** Scored groups go to the learner oldest first; a full queue
//!   holds every send for `send_retry_ms` or until the learner applies
//!   something, then the same group is sent again. Any other answer means the
//!   learner holds the group: it is `sent` until the learner reports it
//!   applied, dropped or committed (an applied group sent again after a
//!   resume goes back to `sent`, since the learner may hold it queued again).

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

use splinter_core::factory::group::{GroupKey, GroupState, RolloutOutcome, RolloutState};
use splinter_core::rewards::differ;

use super::seams::{LearnerEvent, RolloutEnd, SendOutcome, TaskSelector};
use super::stats::Stats;
use super::{Settings, SettingsError};

/// Why a group was filtered: stored as the group's reason.
pub mod reasons {
    /// No rollout of the group was scored.
    pub const UNSCORED: &str = "unscored";
    /// Its rewards do not differ.
    pub const ZERO_VARIANCE: &str = "zero-variance";
    /// Its exact rollouts' rewards do not differ.
    pub const NOT_EXACT: &str = "not-exact";
    /// Dropped by this controller as too old.
    pub const STALE_HERE: &str = "stale";
    /// Dropped by the learner as too old.
    pub const STALE_AT_LEARNER: &str = "stale-at-learner";
}

/// One durable change, in the order the controller made it.
#[derive(Clone, Debug, PartialEq)]
pub enum Change {
    /// A new group of `size` pending rollouts of `task`.
    Create {
        /// Its key.
        key: GroupKey,
        /// Its place in the run.
        seq: u64,
        /// The task.
        task: String,
        /// Its rollouts.
        size: u32,
    },
    /// A rollout starts (its group with it); start it after recording.
    Start {
        /// The group.
        key: GroupKey,
        /// The task.
        task: String,
        /// The rollout.
        index: u32,
    },
    /// A rollout finished.
    Finish {
        /// The group.
        key: GroupKey,
        /// The rollout.
        index: u32,
        /// How it ended.
        outcome: RolloutOutcome,
        /// Its records.
        records: String,
    },
    /// A rollout's infrastructure failed; retried, or abandoned.
    Fail {
        /// The group.
        key: GroupKey,
        /// The rollout.
        index: u32,
        /// Its retry budget is spent.
        abandon: bool,
        /// What failed.
        reason: String,
    },
    /// The group moves along its lifecycle.
    State {
        /// The group.
        key: GroupKey,
        /// Its new state.
        to: GroupState,
        /// Why, for a filtered or dropped group.
        reason: Option<&'static str>,
        /// Its staleness, when sent.
        staleness: Option<u64>,
    },
}

/// A group ready to send: its scored, exact members' rewards; the caller
/// adds their records.
#[derive(Clone, Debug, PartialEq)]
pub struct Draft {
    /// The group.
    pub key: GroupKey,
    /// The task.
    pub task: String,
    /// `(index, reward)` of each member, by index.
    pub members: Vec<(u32, f64)>,
}

/// A seam broke its contract (an end for a rollout not running, an answer
/// for a group not being sent).
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("online controller: {0}")]
pub struct ControlError(pub String);

#[derive(Clone, Debug)]
struct Rollout {
    state: RolloutState,
    failures: u32,
    outcome: Option<RolloutOutcome>,
}

#[derive(Clone, Debug)]
struct Live {
    key: GroupKey,
    task: String,
    state: GroupState,
    rollouts: Vec<Rollout>,
    /// Tokens of its finished rollouts.
    tokens: u64,
    /// Whether `tokens` count toward the queue.
    queued: bool,
}

mod restore;
mod rules;

use rules::{scored_rewards, trainable, verdict, Verdict};

/// The online controller's state.
#[derive(Debug)]
pub struct Controller {
    settings: Settings,
    learner_version: u64,
    next_seq: u64,
    exhausted: bool,
    live: BTreeMap<u64, Live>,
    seq_of: HashMap<GroupKey, u64>,
    ready: VecDeque<(u64, u32)>,
    send_queue: BTreeSet<u64>,
    resend: HashSet<u64>,
    full_until: Option<u64>,
    step_tokens: VecDeque<u64>,
    blocked_since: Option<u64>,
    stats: Stats,
}

impl Controller {
    /// A controller for a fresh run.
    ///
    /// # Errors
    /// The settings are invalid.
    pub fn new(settings: Settings, learner_version: u64) -> Result<Self, SettingsError> {
        settings.validate()?;
        Ok(Self {
            settings,
            learner_version,
            next_seq: 0,
            exhausted: false,
            live: BTreeMap::new(),
            seq_of: HashMap::new(),
            ready: VecDeque::new(),
            send_queue: BTreeSet::new(),
            resend: HashSet::new(),
            full_until: None,
            step_tokens: VecDeque::new(),
            blocked_since: None,
            stats: Stats::default(),
        })
    }

    /// The next rollout to start, with the changes to record first; `None`
    /// when no slot is free, the allowance holds new groups back, or the
    /// run has every group it asks for.
    pub fn admit(&mut self, now: u64, selector: &mut dyn TaskSelector) -> Option<Vec<Change>> {
        if self.stats.in_flight >= u64::from(self.settings.rollout_slots) {
            self.unblock(now);
            return None;
        }
        if let Some((seq, index)) = self.ready.pop_front() {
            return Some(self.start(seq, index));
        }
        if self.exhausted {
            self.unblock(now);
            return None;
        }
        if !self.fits() {
            self.blocked_since.get_or_insert(now);
            return None;
        }
        self.unblock(now);
        let seq = self.next_seq;
        let Some(task) = selector.select(seq) else {
            self.exhausted = true;
            return None;
        };
        let Ok(key) = GroupKey::numbered(&self.settings.namespace, seq) else {
            // The namespace was validated with the settings.
            self.exhausted = true;
            return None;
        };
        self.next_seq += 1;
        let size = self.settings.group_size;
        self.stats.groups_started += 1;
        self.seq_of.insert(key.clone(), seq);
        self.live.insert(
            seq,
            Live {
                key: key.clone(),
                task: task.clone(),
                state: GroupState::Pending,
                rollouts: vec![
                    Rollout {
                        state: RolloutState::Pending,
                        failures: 0,
                        outcome: None,
                    };
                    usize::try_from(size).unwrap_or(0)
                ],
                tokens: 0,
                queued: true,
            },
        );
        self.ready.extend((1..size).map(|index| (seq, index)));
        let mut changes = vec![Change::Create {
            key,
            seq,
            task,
            size,
        }];
        changes.extend(self.start(seq, 0));
        Some(changes)
    }

    fn start(&mut self, seq: u64, index: u32) -> Vec<Change> {
        let Some(group) = self.live.get_mut(&seq) else {
            return Vec::new();
        };
        let Some(rollout) = group.rollouts.get_mut(index as usize) else {
            return Vec::new();
        };
        rollout.state = RolloutState::Running;
        if group.state == GroupState::Pending {
            group.state = GroupState::Running;
        }
        self.stats.in_flight += 1;
        vec![Change::Start {
            key: group.key.clone(),
            task: group.task.clone(),
            index,
        }]
    }

    fn unblock(&mut self, now: u64) {
        if let Some(since) = self.blocked_since.take() {
            self.stats.backpressure_ms += now.saturating_sub(since);
        }
    }

    /// Tokens consumed per optimiser step, as estimated now.
    fn tokens_per_step(&self) -> u64 {
        let budget = self.settings.tokens_per_step;
        if self.step_tokens.is_empty() {
            return budget;
        }
        let window = self.step_tokens.len() as u64;
        (self.step_tokens.iter().sum::<u64>() / window).max(budget)
    }

    /// The token allowance in force.
    fn allowance(&self) -> u64 {
        self.settings
            .max_staleness
            .max(1)
            .saturating_mul(self.tokens_per_step())
    }

    /// Whether a new group fits in the allowance.
    fn fits(&self) -> bool {
        let admitted = self.stats.in_flight + self.ready.len() as u64;
        let finished = self.stats.rollouts_finished;
        let per_rollout = if finished == 0 {
            self.settings.rollout_tokens
        } else {
            self.stats.tokens_generated.div_ceil(finished)
        };
        let expected = self.stats.queue_tokens + admitted.saturating_mul(per_rollout);
        expected < self.allowance()
    }

    fn count_finished(&mut self, outcome: &RolloutOutcome) {
        let stats = &mut self.stats;
        stats.rollouts_finished += 1;
        stats.tokens_generated += outcome.tokens;
        stats.rollouts_unscored += u64::from(outcome.reward.is_none());
        stats.rollouts_not_exact += u64::from(!outcome.exact);
        stats.rollouts_truncated += u64::from(outcome.truncated);
    }

    fn ingest_rewards(&mut self, rollouts: &[Rollout]) {
        let rewards = scored_rewards(rollouts);
        for reward in &rewards {
            self.stats.reward.add(*reward);
        }
        if !rewards.is_empty() && !differ(rewards.iter().copied()) {
            #[allow(clippy::cast_precision_loss)]
            let mean = rewards.iter().sum::<f64>() / rewards.len() as f64;
            self.stats.zero_std_reward.add(mean);
        }
    }

    fn count_filtered(&mut self, reason: &str) {
        match reason {
            reasons::UNSCORED => self.stats.filtered_unscored += 1,
            reasons::ZERO_VARIANCE => self.stats.filtered_zero_variance += 1,
            _ => self.stats.filtered_not_exact += 1,
        }
    }

    /// A rollout ended: record it and, when its group is complete, score
    /// the group.
    ///
    /// # Errors
    /// The rollout was not running, or its outcome is invalid.
    pub fn rollout_ended(
        &mut self,
        end: RolloutEnd,
        selector: &mut dyn TaskSelector,
    ) -> Result<Vec<Change>, ControlError> {
        let RolloutEnd {
            group: key,
            index,
            result,
        } = end;
        let seq = *self
            .seq_of
            .get(&key)
            .ok_or_else(|| ControlError(format!("an end for {key}/{index}, which is not live")))?;
        let max_failures = self.settings.max_failures;
        let group = self
            .live
            .get_mut(&seq)
            .ok_or_else(|| ControlError(format!("{key} lost")))?;
        let rollout = group
            .rollouts
            .get_mut(index as usize)
            .filter(|r| r.state == RolloutState::Running)
            .ok_or_else(|| {
                ControlError(format!("an end for {key}/{index}, which is not running"))
            })?;
        let mut changes = Vec::new();
        let mut finished = None;
        match result {
            Ok(done) => {
                done.outcome
                    .validate()
                    .map_err(|e| ControlError(format!("{key}/{index}: {e}")))?;
                rollout.state = RolloutState::Finished;
                rollout.outcome = Some(done.outcome.clone());
                group.tokens += done.outcome.tokens;
                finished = Some(done.outcome.clone());
                changes.push(Change::Finish {
                    key: key.clone(),
                    index,
                    outcome: done.outcome,
                    records: done.records,
                });
            }
            Err(reason) => {
                rollout.failures += 1;
                let abandon = rollout.failures > max_failures;
                rollout.state = if abandon {
                    RolloutState::Abandoned
                } else {
                    RolloutState::Pending
                };
                changes.push(Change::Fail {
                    key: key.clone(),
                    index,
                    abandon,
                    reason,
                });
                self.stats.rollout_failures += 1;
                if abandon {
                    self.stats.rollouts_abandoned += 1;
                } else {
                    self.ready.push_front((seq, index));
                }
            }
        }
        let complete = group.rollouts.iter().all(|r| r.state.is_terminal());
        self.stats.in_flight -= 1;
        if let Some(outcome) = finished {
            self.stats.queue_tokens += outcome.tokens;
            self.count_finished(&outcome);
        }
        if complete {
            changes.extend(self.score(seq, selector));
        }
        Ok(changes)
    }

    fn score(&mut self, seq: u64, selector: &mut dyn TaskSelector) -> Vec<Change> {
        let Some(group) = self.live.get_mut(&seq) else {
            return Vec::new();
        };
        group.state = GroupState::Scored;
        let key = group.key.clone();
        let rollouts = group.rollouts.clone();
        selector.observe(&group.task, &scored_rewards(&rollouts));
        self.stats.groups_scored += 1;
        self.ingest_rewards(&rollouts);
        let mut changes = vec![Change::State {
            key,
            to: GroupState::Scored,
            reason: None,
            staleness: None,
        }];
        changes.extend(self.decide(seq));
        changes
    }

    /// Filter, drop or queue a scored group.
    fn decide(&mut self, seq: u64) -> Vec<Change> {
        let Some(group) = self.live.get(&seq) else {
            return Vec::new();
        };
        let mut changes = Vec::new();
        match verdict(
            &group.rollouts,
            self.learner_version,
            self.settings.max_staleness,
        ) {
            Verdict::Filtered(reason) => {
                self.count_filtered(reason);
                self.walk(seq, GroupState::Filtered, Some(reason), &mut changes);
            }
            Verdict::Stale(_) => {
                self.stats.dropped_stale += 1;
                self.walk(
                    seq,
                    GroupState::Stale,
                    Some(reasons::STALE_HERE),
                    &mut changes,
                );
            }
            Verdict::Send(_) => {
                self.hold_trainable(seq);
                self.send_queue.insert(seq);
            }
        }
        changes
    }

    /// A group queued to send holds only what the learner will receive: its
    /// exact scored rollouts' tokens.
    fn hold_trainable(&mut self, seq: u64) {
        let Some(group) = self.live.get_mut(&seq) else {
            return;
        };
        let held = trainable(&group.rollouts)
            .map(|(_, o)| o.tokens)
            .sum::<u64>();
        if group.queued {
            self.stats.queue_tokens = self.stats.queue_tokens - group.tokens + held;
        }
        group.tokens = held;
    }

    /// Moves a group to `target`, through every state on the way
    /// (`scored -> sent -> applied -> committed`), recording each step and
    /// releasing what it holds as it goes.
    fn walk(
        &mut self,
        seq: u64,
        target: GroupState,
        reason: Option<&'static str>,
        changes: &mut Vec<Change>,
    ) {
        let Some(group) = self.live.get_mut(&seq) else {
            return;
        };
        while group.state != target {
            let next = if group.state.may_become(target) {
                target
            } else {
                match group.state {
                    GroupState::Scored => GroupState::Sent,
                    GroupState::Sent => GroupState::Applied,
                    _ => break,
                }
            };
            group.state = next;
            changes.push(Change::State {
                key: group.key.clone(),
                to: next,
                reason: next.is_terminal().then_some(reason).flatten(),
                staleness: None,
            });
            match next {
                GroupState::Applied => self.stats.groups_applied += 1,
                GroupState::Committed => self.stats.groups_committed += 1,
                _ => {}
            }
        }
        if matches!(group.state, GroupState::Applied) || group.state.is_terminal() {
            if group.queued {
                group.queued = false;
                self.stats.queue_tokens -= group.tokens;
            }
            // Sending again after an applied answer would train nothing new.
            if group.state.is_terminal() {
                self.send_queue.remove(&seq);
                self.resend.remove(&seq);
                let key = group.key.clone();
                self.seq_of.remove(&key);
                self.live.remove(&seq);
            }
        }
    }

    /// The group to send next, oldest first; `None` while the learner's
    /// queue is full or nothing waits.
    pub fn next_send(&mut self, now: u64) -> Option<Draft> {
        if let Some(until) = self.full_until {
            if now < until {
                return None;
            }
            self.full_until = None;
        }
        let seq = *self.send_queue.first()?;
        let group = self.live.get(&seq)?;
        Some(Draft {
            key: group.key.clone(),
            task: group.task.clone(),
            members: trainable(&group.rollouts)
                .filter_map(|(i, o)| o.reward.map(|r| (i, r)))
                .collect(),
        })
    }

    /// The learner answered the draft of `key`.
    ///
    /// # Errors
    /// `key` was not being sent.
    pub fn sent(
        &mut self,
        now: u64,
        key: &GroupKey,
        answer: &SendOutcome,
    ) -> Result<Vec<Change>, ControlError> {
        let seq = self
            .seq_of
            .get(key)
            .copied()
            .filter(|seq| self.send_queue.contains(seq))
            .ok_or_else(|| ControlError(format!("an answer for {key}, which is not being sent")))?;
        if let SendOutcome::Full { allowance_tokens } = answer {
            self.stats.learner_full += 1;
            self.stats.learner_allowance = Some(*allowance_tokens);
            self.full_until = Some(now.saturating_add(self.settings.send_retry_ms));
            return Ok(Vec::new());
        }
        self.send_queue.remove(&seq);
        if self.resend.remove(&seq) {
            self.stats.resends += 1;
        }
        let learner_version = self.learner_version;
        let Some(group) = self.live.get_mut(&seq) else {
            return Ok(Vec::new());
        };
        let staleness = trainable(&group.rollouts)
            .map(|(_, o)| o.span.staleness(learner_version))
            .max()
            .unwrap_or(0);
        // Whatever the learner answered, it holds the group now. `accepted`
        // means it queued the group on this send; `duplicate` that it held it
        // already, queued or applied (a send whose answer was lost, or a
        // resend after a restart that took it back): its tokens then leave the
        // queue, since counting a group the learner may have applied could
        // hold admission back until a step that never comes. Only the
        // learner's report ends the group (applied, dropped or committed).
        let queue = *answer == SendOutcome::Accepted;
        if group.queued != queue {
            group.queued = queue;
            if queue {
                self.stats.queue_tokens += group.tokens;
            } else {
                self.stats.queue_tokens -= group.tokens;
            }
        }
        match group.state {
            GroupState::Scored => {
                self.stats.groups_sent += 1;
                *self.stats.staleness.entry(staleness).or_default() += 1;
            }
            GroupState::Applied => {}
            _ => return Ok(Vec::new()),
        }
        group.state = GroupState::Sent;
        Ok(vec![Change::State {
            key: key.clone(),
            to: GroupState::Sent,
            reason: None,
            staleness: Some(staleness),
        }])
    }

    /// The learner reported `event`.
    pub fn learner_event(&mut self, event: LearnerEvent) -> Vec<Change> {
        let mut changes = Vec::new();
        match event {
            LearnerEvent::Step { tokens } => {
                self.step_tokens.push_back(tokens);
                while self.step_tokens.len() > self.settings.step_window {
                    self.step_tokens.pop_front();
                }
            }
            LearnerEvent::Published { version } => {
                self.learner_version = self.learner_version.max(version);
                let waiting: Vec<u64> = self
                    .send_queue
                    .iter()
                    .copied()
                    .filter(|seq| {
                        self.live
                            .get(seq)
                            .is_some_and(|g| g.state == GroupState::Scored)
                    })
                    .collect();
                for seq in waiting {
                    let rollouts = &self.live[&seq].rollouts;
                    let max_staleness = self.settings.max_staleness;
                    if let Verdict::Stale(_) =
                        verdict(rollouts, self.learner_version, max_staleness)
                    {
                        self.stats.dropped_stale += 1;
                        self.walk(
                            seq,
                            GroupState::Stale,
                            Some(reasons::STALE_HERE),
                            &mut changes,
                        );
                    }
                }
            }
            LearnerEvent::Applied { groups } => {
                self.full_until = None;
                self.acknowledge(&groups, GroupState::Applied, None, &mut changes);
            }
            LearnerEvent::Dropped { groups } => {
                let before = changes.len();
                self.acknowledge(
                    &groups,
                    GroupState::Stale,
                    Some(reasons::STALE_AT_LEARNER),
                    &mut changes,
                );
                self.stats.dropped_by_learner += changes[before..]
                    .iter()
                    .filter(|c| {
                        matches!(
                            c,
                            Change::State {
                                to: GroupState::Stale,
                                ..
                            }
                        )
                    })
                    .count() as u64;
            }
            LearnerEvent::Committed { groups } => {
                self.acknowledge(&groups, GroupState::Committed, None, &mut changes);
            }
        }
        changes
    }

    fn acknowledge(
        &mut self,
        groups: &[GroupKey],
        target: GroupState,
        reason: Option<&'static str>,
        changes: &mut Vec<Change>,
    ) {
        for key in groups {
            let Some(&seq) = self.seq_of.get(key) else {
                self.stats.stray_acks += 1;
                continue;
            };
            let state = self.live[&seq].state;
            let reachable = match target {
                GroupState::Applied => matches!(state, GroupState::Scored | GroupState::Sent),
                GroupState::Stale => matches!(state, GroupState::Scored | GroupState::Sent),
                _ => matches!(
                    state,
                    GroupState::Scored | GroupState::Sent | GroupState::Applied
                ),
            };
            if !reachable {
                self.stats.stray_acks += 1;
                continue;
            }
            if state == GroupState::Scored {
                // The learner holds a group this controller had not recorded
                // as sent (it stopped between sending and recording).
                self.stats.groups_sent += 1;
            }
            self.send_queue.remove(&seq);
            self.resend.remove(&seq);
            self.walk(seq, target, reason, changes);
        }
    }

    /// Whether every group the run asks for is terminal.
    #[must_use]
    pub fn done(&self) -> bool {
        self.exhausted && self.live.is_empty()
    }

    /// Whether only the learner can move the run on: every group asked for
    /// is sent or applied, nothing runs or waits to be sent. The learner is
    /// then told to finish, so a last partial step is trained and committed.
    #[must_use]
    pub fn waits_on_learner(&self) -> bool {
        self.exhausted
            && self.stats.in_flight == 0
            && self.ready.is_empty()
            && self.send_queue.is_empty()
            && !self.live.is_empty()
    }

    /// The statistics as of `now`.
    #[must_use]
    pub fn stats(&self, now: u64) -> Stats {
        let mut stats = self.stats.clone();
        if let Some(since) = self.blocked_since {
            stats.backpressure_ms += now.saturating_sub(since);
        }
        stats.learner_version = self.learner_version;
        stats.tokens_per_step = self.tokens_per_step();
        stats.allowance = self.allowance();
        stats
    }

    /// How many groups are not terminal.
    #[must_use]
    pub fn live_groups(&self) -> usize {
        self.live.len()
    }
}
