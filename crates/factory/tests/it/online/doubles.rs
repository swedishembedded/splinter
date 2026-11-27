// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements online reinforcement-learning controllers
// that never train a group twice or lose one they produced, for its clients.
// If your team needs expertise in asynchronous policy optimisation, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Scripted doubles of an online run's seams, on one manual clock: an
//! executor whose rollouts end at scripted times, a learner that trains
//! token-budgeted steps, publishes and commits, and an in-memory ledger that
//! keeps the run state's rules.

use std::cell::Cell;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet, VecDeque};
use std::rc::Rc;

use splinter_core::factory::group::{
    GroupKey, GroupState, RolloutOutcome, RolloutState, VersionSpan,
};
use splinter_factory::online::control::Change;
use splinter_factory::online::seams::{
    ExecutorError, FinishedRollout, GroupLedger, Learner, LearnerError, LearnerEvent, LearnerState,
    LedgerError, OutgoingGroup, RolloutEnd, RolloutExecutor, RolloutRequest, RunClock, SendOutcome,
};
use splinter_store::factory::runstate::{GroupEntry, GroupRecord, GroupRolloutRecord};

/// The run's time, shared by every double.
#[derive(Clone, Default)]
pub struct Clock(pub Rc<Cell<u64>>);

impl Clock {
    pub fn now(&self) -> u64 {
        self.0.get()
    }
    pub fn advance_to(&self, at: u64) {
        self.0.set(self.0.get().max(at));
    }
}

impl RunClock for Clock {
    fn now_ms(&self) -> u64 {
        self.now()
    }
}

/// How one start of a rollout goes: after `duration_ms`, this result.
pub struct Plan {
    pub duration_ms: u64,
    /// `Ok((reward, tokens, exact))`, or an infrastructure failure.
    pub result: Result<(Option<f64>, u64, bool), String>,
}

/// Decides each start: `(request, how many times this rollout started before)`.
pub type Script = Box<dyn FnMut(&RolloutRequest, u32) -> Plan>;

/// The policy version the learner serves, read by the executor.
pub type Served = Rc<Cell<u64>>;

pub struct Executor {
    clock: Clock,
    served: Served,
    script: Script,
    heap: BinaryHeap<Reverse<(u64, u64)>>,
    pending: HashMap<u64, (RolloutRequest, u64, Plan)>,
    counter: u64,
    starts_of: HashMap<(GroupKey, u32), u32>,
    /// Every start, in order.
    pub starts: Vec<RolloutRequest>,
    /// Every end delivered, with its time.
    pub ends: Vec<(u64, RolloutEnd)>,
    /// Fail with an executor error once this many ends were delivered.
    pub die_after: Option<usize>,
    /// Rollouts stopped by a cancel.
    pub cancelled: usize,
    peak: usize,
}

impl Executor {
    pub fn new(clock: Clock, served: Served, script: Script) -> Self {
        Self {
            clock,
            served,
            script,
            heap: BinaryHeap::new(),
            pending: HashMap::new(),
            counter: 0,
            starts_of: HashMap::new(),
            starts: Vec::new(),
            ends: Vec::new(),
            die_after: None,
            cancelled: 0,
            peak: 0,
        }
    }

    pub fn running(&self) -> usize {
        self.pending.len()
    }

    /// The most rollouts that ran at once.
    pub fn peak_running(&self) -> usize {
        self.peak
    }
}

impl RolloutExecutor for Executor {
    fn start(&mut self, request: RolloutRequest) -> Result<(), ExecutorError> {
        let nth = self
            .starts_of
            .entry((request.group.clone(), request.index))
            .or_default();
        let plan = (self.script)(&request, *nth);
        *nth += 1;
        let at = self.clock.now() + plan.duration_ms.max(1);
        self.counter += 1;
        self.heap.push(Reverse((at, self.counter)));
        self.pending
            .insert(self.counter, (request.clone(), self.served.get(), plan));
        self.starts.push(request);
        self.peak = self.peak.max(self.pending.len());
        Ok(())
    }

    fn wait(&mut self, until_ms: u64) -> Result<Vec<RolloutEnd>, ExecutorError> {
        if self.die_after.is_some_and(|n| self.ends.len() >= n) {
            return Err(ExecutorError("killed".into()));
        }
        let Some(Reverse((at, _))) = self.heap.peek().copied() else {
            self.clock.advance_to(until_ms);
            return Ok(Vec::new());
        };
        if at > until_ms {
            self.clock.advance_to(until_ms);
            return Ok(Vec::new());
        }
        self.clock.advance_to(at);
        let mut out = Vec::new();
        while let Some(Reverse((t, id))) = self.heap.peek().copied() {
            if t > at {
                break;
            }
            self.heap.pop();
            let (request, version_at_start, plan) = self.pending.remove(&id).unwrap();
            let result = plan.result.map(|(reward, tokens, exact)| FinishedRollout {
                outcome: RolloutOutcome {
                    reward,
                    tokens,
                    span: VersionSpan::new(version_at_start, self.served.get()).unwrap(),
                    exact,
                    truncated: false,
                    truncation: None,
                },
                records: format!("{tokens},{version_at_start}"),
            });
            let end = RolloutEnd {
                group: request.group,
                index: request.index,
                result,
            };
            self.ends.push((at, end.clone()));
            out.push(end);
        }
        Ok(out)
    }

    fn cancel(&mut self) {
        self.cancelled += self.pending.len();
        self.pending.clear();
        self.heap.clear();
    }
}

/// How the learner double trains.
#[derive(Clone, Copy)]
pub struct LearnerPlan {
    pub step_tokens: u64,
    pub step_ms: u64,
    pub publish_every: u64,
    pub commit_every: u64,
    pub capacity: u64,
    pub max_staleness: u64,
}

pub struct ScriptedLearner {
    clock: Clock,
    pub served: Served,
    pub plan: LearnerPlan,
    queue: VecDeque<(GroupKey, u64, u64)>,
    queued_tokens: u64,
    held: HashSet<GroupKey>,
    applied: Vec<GroupKey>,
    pub committed: HashSet<GroupKey>,
    /// How often each group's update was made durable.
    pub commits: HashMap<GroupKey, u32>,
    /// Every send and its answer, with its time.
    pub sends: Vec<(u64, GroupKey, SendOutcome)>,
    steps: u64,
    next_step_at: u64,
    finishing: bool,
    events: Vec<LearnerEvent>,
    /// Fail right after accepting this many sends (the connection dropped
    /// with the group already taken).
    pub die_after_accepts: Option<usize>,
    pub accepts: usize,
}

impl ScriptedLearner {
    pub fn new(clock: Clock, plan: LearnerPlan) -> Self {
        Self {
            clock,
            served: Rc::new(Cell::new(0)),
            plan,
            queue: VecDeque::new(),
            queued_tokens: 0,
            held: HashSet::new(),
            applied: Vec::new(),
            committed: HashSet::new(),
            commits: HashMap::new(),
            sends: Vec::new(),
            steps: 0,
            next_step_at: 0,
            finishing: false,
            events: Vec::new(),
            die_after_accepts: None,
            accepts: 0,
        }
    }

    /// The learner process restarts from its last checkpoint: what was
    /// queued or applied and not committed is gone.
    pub fn restart(&mut self) {
        self.queue.clear();
        self.queued_tokens = 0;
        self.applied.clear();
        self.held = self.committed.clone();
        self.finishing = false;
        self.events.clear();
    }

    pub fn queued_tokens(&self) -> u64 {
        self.queued_tokens
    }

    fn commit(&mut self) {
        if self.applied.is_empty() {
            return;
        }
        let groups = std::mem::take(&mut self.applied);
        for key in &groups {
            *self.commits.entry(key.clone()).or_default() += 1;
            self.committed.insert(key.clone());
        }
        self.events.push(LearnerEvent::Committed { groups });
    }

    fn train(&mut self) {
        let now = self.clock.now();
        while now >= self.next_step_at
            && !self.queue.is_empty()
            && (self.queued_tokens >= self.plan.step_tokens || self.finishing)
        {
            let (mut tokens, mut applied, mut dropped) = (0, Vec::new(), Vec::new());
            while tokens < self.plan.step_tokens {
                let Some((key, t, oldest)) = self.queue.pop_front() else {
                    break;
                };
                self.queued_tokens -= t;
                if self.served.get().saturating_sub(oldest) > self.plan.max_staleness {
                    self.held.remove(&key);
                    dropped.push(key);
                } else {
                    tokens += t;
                    applied.push(key);
                }
            }
            if !dropped.is_empty() {
                self.events.push(LearnerEvent::Dropped { groups: dropped });
            }
            if applied.is_empty() {
                continue;
            }
            self.steps += 1;
            self.next_step_at = now + self.plan.step_ms;
            self.events.push(LearnerEvent::Step { tokens });
            self.events.push(LearnerEvent::Applied {
                groups: applied.clone(),
            });
            self.applied.extend(applied);
            if self.steps.is_multiple_of(self.plan.publish_every) {
                self.served.set(self.served.get() + 1);
                self.events.push(LearnerEvent::Published {
                    version: self.served.get(),
                });
            }
            if self.steps.is_multiple_of(self.plan.commit_every) {
                self.commit();
            }
        }
        if self.finishing && self.queue.is_empty() {
            self.commit();
        }
    }
}

fn parse_member(records: &str) -> (u64, u64) {
    let (tokens, version) = records.split_once(',').unwrap();
    (tokens.parse().unwrap(), version.parse().unwrap())
}

impl Learner for ScriptedLearner {
    fn state(&mut self) -> Result<LearnerState, LearnerError> {
        let mut committed: Vec<_> = self.committed.iter().cloned().collect();
        committed.sort();
        Ok(LearnerState {
            policy_version: self.served.get(),
            committed,
        })
    }

    fn send(&mut self, group: &OutgoingGroup) -> Result<SendOutcome, LearnerError> {
        let now = self.clock.now();
        let answer = if self.held.contains(&group.key) {
            SendOutcome::Duplicate
        } else {
            let members: Vec<_> = group
                .members
                .iter()
                .map(|m| parse_member(&m.records))
                .collect();
            let tokens = members.iter().map(|m| m.0).sum::<u64>();
            let oldest = members.iter().map(|m| m.1).min().unwrap();
            // Full only while it holds a step's worth: a learner that cannot
            // train yet must take more, or the run could never move.
            if self.queued_tokens >= self.plan.step_tokens
                && self.queued_tokens + tokens > self.plan.capacity
            {
                SendOutcome::Full {
                    allowance_tokens: self.plan.capacity,
                }
            } else {
                self.queue.push_back((group.key.clone(), tokens, oldest));
                self.queued_tokens += tokens;
                self.held.insert(group.key.clone());
                self.finishing = false;
                SendOutcome::Accepted
            }
        };
        self.sends.push((now, group.key.clone(), answer.clone()));
        if answer == SendOutcome::Accepted {
            self.accepts += 1;
            if self.die_after_accepts == Some(self.accepts) {
                return Err(LearnerError::new("connection lost"));
            }
        }
        Ok(answer)
    }

    fn poll(&mut self) -> Result<Vec<LearnerEvent>, LearnerError> {
        self.train();
        Ok(std::mem::take(&mut self.events))
    }

    fn finish(&mut self) -> Result<(), LearnerError> {
        self.finishing = true;
        self.train();
        Ok(())
    }
}

/// The run state's ledger rules in memory, for runs too long for a disk.
#[derive(Default)]
pub struct MemoryLedger {
    pub groups: BTreeMap<u64, GroupEntry>,
    /// Changes applied so far, readable while the run holds the ledger.
    pub changes: Rc<Cell<u64>>,
    seq_of: HashMap<GroupKey, u64>,
    records: HashMap<(GroupKey, u32), String>,
}

impl MemoryLedger {
    fn entry(&mut self, key: &GroupKey) -> Result<&mut GroupEntry, LedgerError> {
        let seq = self
            .seq_of
            .get(key)
            .ok_or_else(|| LedgerError(format!("unknown {key}")))?;
        Ok(self.groups.get_mut(seq).unwrap())
    }

    fn rollout(
        &mut self,
        key: &GroupKey,
        index: u32,
    ) -> Result<&mut GroupRolloutRecord, LedgerError> {
        self.entry(key)?
            .rollouts
            .get_mut(index as usize)
            .ok_or_else(|| LedgerError(format!("unknown {key}/{index}")))
    }
}

impl GroupLedger for MemoryLedger {
    fn load(&mut self) -> Result<Vec<GroupEntry>, LedgerError> {
        Ok(self.groups.values().cloned().collect())
    }

    fn apply(&mut self, now_ms: u64, change: &Change) -> Result<(), LedgerError> {
        self.changes.set(self.changes.get() + 1);
        let at = now_ms / 1000;
        match change {
            Change::Create {
                key,
                seq,
                task,
                size,
            } => {
                if self.seq_of.contains_key(key) {
                    return Ok(());
                }
                self.seq_of.insert(key.clone(), *seq);
                let rollouts = (0..*size)
                    .map(|index| GroupRolloutRecord {
                        index,
                        state: RolloutState::Pending,
                        failures: 0,
                        outcome: None,
                        updated_at: at,
                    })
                    .collect();
                self.groups.insert(
                    *seq,
                    GroupEntry {
                        group: GroupRecord {
                            key: key.clone(),
                            seq: *seq,
                            task: task.clone(),
                            size: *size,
                            state: GroupState::Pending,
                            reason: None,
                            staleness: None,
                            updated_at: at,
                        },
                        rollouts,
                    },
                );
            }
            Change::Start { key, index, .. } => {
                let rollout = self.rollout(key, *index)?;
                if rollout.state.is_terminal() {
                    return Err(LedgerError(format!("{key}/{index} started again")));
                }
                rollout.state = RolloutState::Running;
                let group = &mut self.entry(key)?.group;
                if group.state == GroupState::Pending {
                    group.state = GroupState::Running;
                }
            }
            Change::Finish {
                key,
                index,
                outcome,
                records,
            } => {
                let rollout = self.rollout(key, *index)?;
                if rollout.state != RolloutState::Running {
                    return Err(LedgerError(format!(
                        "{key}/{index} finished from {}",
                        rollout.state
                    )));
                }
                rollout.state = RolloutState::Finished;
                rollout.outcome = Some(outcome.clone());
                self.records.insert((key.clone(), *index), records.clone());
            }
            Change::Fail {
                key,
                index,
                abandon,
                ..
            } => {
                let rollout = self.rollout(key, *index)?;
                if rollout.state != RolloutState::Running {
                    return Err(LedgerError(format!(
                        "{key}/{index} failed from {}",
                        rollout.state
                    )));
                }
                rollout.failures += 1;
                rollout.state = if *abandon {
                    RolloutState::Abandoned
                } else {
                    RolloutState::Pending
                };
            }
            Change::State {
                key,
                to,
                reason,
                staleness,
            } => {
                let entry = self.entry(key)?;
                let group = &mut entry.group;
                if !group.state.may_become(*to) {
                    return Err(LedgerError(format!("{key}: {} -> {to}", group.state)));
                }
                group.state = *to;
                if let Some(reason) = reason {
                    group.reason = Some((*reason).to_string());
                }
                if staleness.is_some() {
                    group.staleness = *staleness;
                }
                if to.is_terminal() {
                    let size = group.size;
                    for index in 0..size {
                        self.records.remove(&(key.clone(), index));
                    }
                }
            }
        }
        Ok(())
    }

    fn records(&mut self, key: &GroupKey) -> Result<Vec<(u32, String)>, LedgerError> {
        let size = self.entry(key)?.group.size;
        Ok((0..size)
            .filter_map(|i| self.records.get(&(key.clone(), i)).map(|r| (i, r.clone())))
            .collect())
    }
}
