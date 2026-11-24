// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fenced, crash-safe controller state for
// long-running automation, checked against a reference model, for its
// clients. If your team needs expertise in concurrency control for
// distributed workflows, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec, by model (plan 10A and 10C): the controller's run state against a
//! reference model of its documented rules, over random sequences of what
//! two controllers do, each through its own connection as two processes
//! would: take, renew and give up the lease while the clock moves past its
//! expiry; make, claim, move, finish and abandon slots and attempts. Every
//! operation's outcome (done, or refused and why) and the slots and attempts
//! it leaves are the model's. Over every sequence:
//!
//! * a controller holding an older fence than the newest one given never
//!   changes anything (a stale controller cannot replace a newer result);
//! * a finished slot's trial never changes;
//! * no attempt once claimed is lost, and none appears that was not claimed.
//!
//! The model is written from the rules the run state's documentation states,
//! never from its queries.
#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;

use proptest::prelude::*;
use splinter_conformance::fuzz;
use splinter_core::digest::Digest;
use splinter_store::factory::runstate::{
    AttemptId, AttemptState, Fence, RunState, RunStateError, SlotKey, SlotState,
};

const TTL: u64 = 10;
const PASS: &str = "pass-a";
const SLOTS: u32 = 2;

/// One of the two controllers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum C {
    A,
    B,
}

impl C {
    fn name(self) -> &'static str {
        match self {
            C::A => "controller-a",
            C::B => "controller-b",
        }
    }
    fn at(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, Debug)]
enum Op {
    Advance(u64),
    Acquire(C),
    Renew(C),
    Release(C),
    Ensure(C),
    Claim(C),
    /// An attempt (by position, modulo how many there are) moved to a state.
    Move(C, usize, AttemptState),
    /// A slot finished with trial x (`true`) or y.
    Finish(C, u32, bool),
    Abandon(C, u32),
}

/// How an operation ended.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Out {
    Done,
    NothingToClaim,
    LeaseHeld,
    Stale,
    Illegal,
    SlotFinished,
    Unknown,
    /// Not run: the controller holds no fence yet, or there is no attempt.
    Skipped,
}

fn classify(error: &RunStateError) -> Out {
    match error {
        RunStateError::LeaseHeld { .. } => Out::LeaseHeld,
        RunStateError::StaleFence { .. } => Out::Stale,
        RunStateError::IllegalTransition { .. } => Out::Illegal,
        RunStateError::SlotFinished { .. } => Out::SlotFinished,
        RunStateError::UnknownSlot(_) | RunStateError::UnknownAttempt(_) => Out::Unknown,
        other => panic!("an outcome the model has no rule for: {other}"),
    }
}

/// What a run state shows: each slot's state and trial, each attempt's slot
/// and state, in order.
type Shown = (Vec<(SlotState, Option<bool>)>, Vec<(u32, AttemptState)>);

/// The reference model.
#[derive(Default)]
struct Model {
    now: u64,
    /// Holder, token, expiry.
    lease: Option<(C, u64, u64)>,
    fences: [Option<u64>; 2],
    slots: Vec<(SlotState, Option<bool>)>,
    attempts: Vec<(u32, AttemptState)>,
}

impl Model {
    fn fenced(&self, c: C) -> Result<(), Out> {
        let fence = self.fences[c.at()].ok_or(Out::Skipped)?;
        match self.lease {
            Some((holder, token, _)) if holder == c && token == fence => Ok(()),
            _ => Err(Out::Stale),
        }
    }

    fn apply(&mut self, op: Op) -> Out {
        match self.step(op) {
            Ok(out) | Err(out) => out,
        }
    }

    fn step(&mut self, op: Op) -> Result<Out, Out> {
        match op {
            Op::Advance(dt) => self.now += dt,
            Op::Acquire(c) => {
                let token = match self.lease {
                    Some((holder, token, until)) if until > self.now && holder == c => token,
                    Some((_, _, until)) if until > self.now => return Err(Out::LeaseHeld),
                    Some((_, token, _)) => token + 1,
                    None => 1,
                };
                self.lease = Some((c, token, self.now + TTL));
                self.fences[c.at()] = Some(token);
            }
            Op::Renew(c) => {
                self.fenced(c)?;
                if let Some(lease) = &mut self.lease {
                    lease.2 = self.now + TTL;
                }
            }
            Op::Release(c) => {
                self.fenced(c)?;
                if let Some(lease) = &mut self.lease {
                    lease.2 = self.now;
                }
            }
            Op::Ensure(c) => {
                self.fenced(c)?;
                while self.slots.len() < SLOTS as usize {
                    self.slots.push((SlotState::Pending, None));
                }
            }
            Op::Claim(c) => {
                self.fenced(c)?;
                let Some(slot) = self.slots.iter().position(|s| s.0 == SlotState::Pending) else {
                    return Ok(Out::NothingToClaim);
                };
                self.slots[slot].0 = SlotState::Running;
                self.attempts.push((slot as u32, AttemptState::Launched));
            }
            Op::Move(c, at, to) => {
                if self.attempts.is_empty() {
                    return Err(Out::Skipped);
                }
                self.fenced(c)?;
                let at = at % self.attempts.len();
                let (slot, from) = self.attempts[at];
                let forward = match from {
                    AttemptState::Launched => to != AttemptState::Launched,
                    AttemptState::Running => to.is_terminal(),
                    _ => false,
                };
                if !forward {
                    return Err(Out::Illegal);
                }
                self.attempts[at].1 = to;
                let slot = &mut self.slots[slot as usize];
                if matches!(to, AttemptState::Failed | AttemptState::Lost)
                    && slot.0 == SlotState::Running
                {
                    slot.0 = SlotState::Pending;
                }
            }
            Op::Finish(c, slot, x) => {
                self.fenced(c)?;
                let slot = self.slots.get_mut(slot as usize).ok_or(Out::Unknown)?;
                match *slot {
                    (SlotState::Finished, Some(trial)) if trial == x => {}
                    (SlotState::Finished, _) => return Err(Out::SlotFinished),
                    (SlotState::Abandoned, _) => return Err(Out::Illegal),
                    _ => *slot = (SlotState::Finished, Some(x)),
                }
            }
            Op::Abandon(c, slot) => {
                self.fenced(c)?;
                let slot = self.slots.get_mut(slot as usize).ok_or(Out::Unknown)?;
                match slot.0 {
                    SlotState::Abandoned => {}
                    SlotState::Finished => return Err(Out::Illegal),
                    _ => slot.0 = SlotState::Abandoned,
                }
            }
        }
        Ok(Out::Done)
    }

    fn shown(&self) -> Shown {
        (self.slots.clone(), self.attempts.clone())
    }
}

/// The run state under test: one connection per controller.
struct Real {
    _dir: tempfile::TempDir,
    connections: [RunState; 2],
    fences: [Option<Fence>; 2],
    attempts: Vec<AttemptId>,
    now: u64,
}

fn trial(x: bool) -> Digest {
    Digest::sha256_of(if x { b"x" } else { b"y" })
}

impl Real {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let connections = [
            RunState::open(dir.path()).unwrap(),
            RunState::open(dir.path()).unwrap(),
        ];
        Self {
            _dir: dir,
            connections,
            fences: [None, None],
            attempts: Vec::new(),
            now: 0,
        }
    }

    fn apply(&mut self, op: Op) -> Out {
        let fenced = |c: C| self.fences[c.at()].clone().ok_or(Out::Skipped);
        let done = |r: Result<(), RunStateError>| r.map_or_else(|e| classify(&e), |()| Out::Done);
        match op {
            Op::Advance(dt) => {
                self.now += dt;
                Out::Done
            }
            Op::Acquire(c) => match self.connections[c.at()].acquire(c.name(), TTL, self.now) {
                Ok(fence) => {
                    self.fences[c.at()] = Some(fence);
                    Out::Done
                }
                Err(e) => classify(&e),
            },
            Op::Renew(c) => match fenced(c) {
                Ok(f) => done(self.connections[c.at()].renew(&f, TTL, self.now)),
                Err(out) => out,
            },
            Op::Release(c) => match fenced(c) {
                Ok(f) => done(self.connections[c.at()].release(&f, self.now)),
                Err(out) => out,
            },
            Op::Ensure(c) => match fenced(c) {
                Ok(f) => done(
                    self.connections[c.at()]
                        .ensure_slots(
                            &f,
                            &Digest::sha256_of(b"stage"),
                            PASS,
                            &["t1".into()],
                            SLOTS,
                            self.now,
                        )
                        .map(drop),
                ),
                Err(out) => out,
            },
            Op::Claim(c) => match fenced(c) {
                Ok(f) => match self.connections[c.at()].claim_next(&f, PASS, "{}", self.now) {
                    Ok(Some(attempt)) => {
                        self.attempts.push(attempt.id);
                        Out::Done
                    }
                    Ok(None) => Out::NothingToClaim,
                    Err(e) => classify(&e),
                },
                Err(out) => out,
            },
            Op::Move(c, at, to) => {
                if self.attempts.is_empty() {
                    return Out::Skipped;
                }
                match fenced(c) {
                    Ok(f) => {
                        let id = self.attempts[at % self.attempts.len()];
                        let result = (to == AttemptState::Succeeded).then(|| trial(true));
                        done(
                            self.connections[c.at()]
                                .record_attempt(&f, id, to, result, self.now)
                                .map(drop),
                        )
                    }
                    Err(out) => out,
                }
            }
            Op::Finish(c, slot, x) => match fenced(c) {
                Ok(f) => done(
                    self.connections[c.at()]
                        .finish_slot(&f, &SlotKey::new(PASS, "t1", slot), &trial(x), self.now)
                        .map(drop),
                ),
                Err(out) => out,
            },
            Op::Abandon(c, slot) => match fenced(c) {
                Ok(f) => done(
                    self.connections[c.at()]
                        .abandon_slot(&f, &SlotKey::new(PASS, "t1", slot), self.now)
                        .map(drop),
                ),
                Err(out) => out,
            },
        }
    }

    fn shown(&self) -> Shown {
        let state = &self.connections[0];
        let slots = state
            .list_slots(PASS)
            .unwrap()
            .into_iter()
            .map(|s| (s.state, s.trial.map(|t| t == trial(true))))
            .collect();
        let mut attempts: BTreeMap<AttemptId, (u32, AttemptState)> = BTreeMap::new();
        for slot in 0..SLOTS {
            for a in state.attempts(&SlotKey::new(PASS, "t1", slot)).unwrap() {
                attempts.insert(a.id, (a.slot.slot, a.state));
            }
        }
        (slots, attempts.into_values().collect())
    }
}

fn op() -> impl Strategy<Value = Op> {
    let c = prop_oneof![Just(C::A), Just(C::B)];
    let state = prop::sample::select(vec![
        AttemptState::Launched,
        AttemptState::Running,
        AttemptState::Succeeded,
        AttemptState::Failed,
        AttemptState::Lost,
    ]);
    prop_oneof![
        prop::sample::select(vec![1u64, 6, 11]).prop_map(Op::Advance),
        c.clone().prop_map(Op::Acquire),
        c.clone().prop_map(Op::Renew),
        c.clone().prop_map(Op::Release),
        c.clone().prop_map(Op::Ensure),
        c.clone().prop_map(Op::Claim),
        (c.clone(), any::<usize>(), state).prop_map(|(c, a, s)| Op::Move(c, a, s)),
        (c.clone(), 0..=SLOTS, any::<bool>()).prop_map(|(c, s, x)| Op::Finish(c, s, x)),
        (c, 0..=SLOTS).prop_map(|(c, s)| Op::Abandon(c, s)),
    ]
}

/// Runs `ops` on both and checks each step and the properties.
fn check(ops: &[Op]) -> Result<(), TestCaseError> {
    let mut model = Model::default();
    let mut real = Real::new();
    let mut newest_token = 0;
    let mut finished: BTreeMap<usize, Option<bool>> = BTreeMap::new();
    for (step, &op) in ops.iter().enumerate() {
        let before = model.shown();
        let stale = match op {
            Op::Advance(_) | Op::Acquire(_) => false,
            Op::Renew(c)
            | Op::Release(c)
            | Op::Ensure(c)
            | Op::Claim(c)
            | Op::Move(c, ..)
            | Op::Finish(c, ..)
            | Op::Abandon(c, _) => model.fences[c.at()].is_some_and(|t| t < newest_token),
        };
        let want = model.apply(op);
        let seen = real.apply(op);
        prop_assert_eq!(&seen, &want, "step {} {:?} of {:?}", step, op, ops);
        prop_assert_eq!(
            real.shown(),
            model.shown(),
            "after step {} {:?} of {:?}",
            step,
            op,
            ops
        );
        if let Some((_, token, _)) = model.lease {
            newest_token = newest_token.max(token);
        }
        // A stale controller changes nothing.
        if stale {
            prop_assert!(
                matches!(want, Out::Stale | Out::Skipped),
                "a stale {:?} was not refused: {:?}",
                op,
                want
            );
            prop_assert_eq!(model.shown(), before.clone());
        }
        // A finished slot's trial never changes; no attempt is lost.
        for (i, slot) in model.slots.iter().enumerate() {
            if slot.0 == SlotState::Finished {
                let first = *finished.entry(i).or_insert(slot.1);
                prop_assert_eq!(first, slot.1, "slot {}'s trial changed", i);
            }
        }
        prop_assert!(model.attempts.len() >= before.1.len());
    }
    Ok(())
}

proptest! {
    // Each case is a database on disk, synced at every commit.
    #![proptest_config(ProptestConfig { cases: 384, ..fuzz::config() })]

    #[test]
    fn two_controllers_do_only_what_the_rules_allow(ops in prop::collection::vec(op(), 1..18)) {
        check(&ops)?;
    }
}

#[test]
fn a_controller_whose_lease_was_taken_over_cannot_finish_a_slot() {
    // The case the properties exist for, written out: A's lease expires,
    // B takes it and finishes the slot; A, paused meanwhile, wakes and tries
    // to finish it another way.
    let ops = [
        Op::Acquire(C::A),
        Op::Ensure(C::A),
        Op::Claim(C::A),
        Op::Advance(11),
        Op::Acquire(C::B),
        Op::Finish(C::B, 0, true),
        Op::Finish(C::A, 0, false),
        Op::Move(C::A, 0, AttemptState::Succeeded),
    ];
    check(&ops).unwrap();
    let mut model = Model::default();
    let outs: Vec<Out> = ops.iter().map(|&op| model.apply(op)).collect();
    assert_eq!(outs[6], Out::Stale);
    assert_eq!(outs[7], Out::Stale);
}
