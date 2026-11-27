// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements experiment controllers that keep a fleet
// busy by running every independent step of a pipeline at once, for its
// clients. If your team needs expertise in concurrent, resumable
// machine-learning pipelines, you can procure our services by sending an
// email to info@swedishembedded.com.

//! The controller's schedule: the plan's ready set, run at once.
//!
//! A stage is ready once every stage it consumes has completed in this
//! invocation; the ready stages start in plan order, each on a thread of its
//! own, until the bound is reached. A stage that trains (`model.train`, an
//! online stage) holds the run's trainer while it runs, so no other training
//! starts beside it: brain places a second run on another card only where
//! one has room and refuses it (HTTP 429) past its admission deadline
//! otherwise, and a run spread over several cards or servers holds all of
//! them. A trainer that serves one adapter at a time (a release manifest it
//! follows) cannot serve two candidates at once, so its run takes one stage
//! at a time.
//!
//! After a failure no stage starts; the running ones finish and commit, as
//! a fan-out's items do. Every failure is reported. A cancellation is seen
//! by every running stage at its next boundary; each stays running for the
//! resume, which continues it from its own slots.
//!
//! What a stage computes does not depend on the schedule: its identity is
//! the products it consumes, all committed before it starts, and a stage
//! identity two stages produce at once (a shared baseline arm, a recipe
//! input) is produced by one while the other waits, then reused.

use std::collections::BTreeSet;
use std::num::NonZeroUsize;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::mpsc;

use super::compute::panic_message;
use super::{online, stage_error, Factory, RunError, StageEntry, StagesAtOnce};
use crate::ops::names;
use crate::plan::PlannedStage;

/// How many stages run at once unless the controller is told otherwise.
///
/// Three: the widest ready sets the factory's generational recipes produce
/// are three stages - the three evaluations of one candidate (exam, unseen
/// tasks, retention), and a generation's champion, teacher and second
/// teacher attempts - so three takes the whole of that win. Each stage
/// already runs as many trials at once as its own plan says, sized as if it
/// ran alone, so the run's sandboxes, worker processes and requests in
/// flight grow with this bound: a wider one is a choice for a host with the
/// room, made on the command line.
pub const DEFAULT_CONCURRENT_STAGES: NonZeroUsize = match NonZeroUsize::new(3) {
    Some(bound) => bound,
    None => unreachable!(),
};

/// The stack a stage's thread gets: a process's main thread's, which every
/// stage ran on before stages ran at once.
const STAGE_STACK_BYTES: usize = 8 << 20;

/// Whether `stage` holds the run's trainer while it runs.
fn trains(stage: &PlannedStage) -> bool {
    stage.operation == names::TRAIN || stage.operation == online::ONLINE
}

/// What the schedule knows about the plan's stages, by plan index.
struct Ledger<'p> {
    stages: &'p [PlannedStage],
    started: Vec<bool>,
    completed: BTreeSet<&'p str>,
    running: BTreeSet<usize>,
}

impl<'p> Ledger<'p> {
    fn new(stages: &'p [PlannedStage]) -> Self {
        Self {
            stages,
            started: vec![false; stages.len()],
            completed: BTreeSet::new(),
            running: BTreeSet::new(),
        }
    }

    /// The first stage in plan order that has not started, whose inputs are
    /// all committed, and that does not need a trainer another stage holds.
    fn next_ready(&self) -> Option<usize> {
        let trainer_held = self.running.iter().any(|&i| trains(&self.stages[i]));
        self.stages.iter().enumerate().position(|(i, stage)| {
            !self.started[i]
                && stage
                    .depends_on
                    .iter()
                    .all(|d| self.completed.contains(d.as_str()))
                && !(trainer_held && trains(stage))
        })
    }
}

impl Factory<'_> {
    /// The bound this run keeps: the one it was given, unless its trainer
    /// can serve only one adapter at a time.
    fn stage_bound(&self) -> NonZeroUsize {
        let one_adapter = self
            .services
            .trainer
            .is_some_and(crate::training::Trainer::serves_one_adapter_at_a_time);
        if one_adapter && self.concurrent_stages.get() > 1 {
            eprintln!(
                "splinter: stages run one at a time: the trainer's servers follow a release \
                 manifest and serve one adapter at a time (serve adapters from a directory \
                 to run stages at once)"
            );
            return NonZeroUsize::MIN;
        }
        self.concurrent_stages
    }

    /// Every stage of the plan, each once its inputs are committed, at most
    /// the bound at once; the entries in plan order.
    pub(super) fn run_stages(&self) -> Result<(Vec<StageEntry>, StagesAtOnce), RunError> {
        let stages = self.resolved.plan.stages.as_slice();
        let bound = self.stage_bound().get();
        let mut ledger = Ledger::new(stages);
        let mut entries: Vec<Option<StageEntry>> = vec![None; stages.len()];
        let mut failures: Vec<(String, RunError)> = Vec::new();
        let mut cancelled = false;
        let mut most = 0;
        std::thread::scope(|scope| {
            let (sent, results) = mpsc::channel();
            loop {
                while failures.is_empty() && !cancelled && ledger.running.len() < bound {
                    if let Err(e) = self.cancelled() {
                        debug_assert!(matches!(e, RunError::Cancelled));
                        cancelled = true;
                        break;
                    }
                    let Some(next) = ledger.next_ready() else {
                        break;
                    };
                    let stage = &stages[next];
                    let sent = sent.clone();
                    let spawned = std::thread::Builder::new()
                        .name(format!("stage {}", stage.key))
                        .stack_size(STAGE_STACK_BYTES)
                        .spawn_scoped(scope, move || {
                            // A panicking stage is a failed stage: its result
                            // is sent like any other, or the loop below would
                            // wait for it for ever.
                            let result = catch_unwind(AssertUnwindSafe(|| self.stage(stage)))
                                .unwrap_or_else(|panic| {
                                    Err(stage_error(
                                        &stage.key,
                                        format!("panicked: {}", panic_message(&*panic)),
                                    ))
                                });
                            let _ = sent.send((next, result));
                        });
                    ledger.started[next] = true;
                    match spawned {
                        Ok(_) => {
                            ledger.running.insert(next);
                            most = most.max(ledger.running.len());
                        }
                        Err(e) => failures.push((
                            stage.key.clone(),
                            stage_error(&stage.key, format!("cannot start its thread: {e}")),
                        )),
                    }
                }
                if ledger.running.is_empty() {
                    break;
                }
                // Every running stage's thread sends exactly once.
                let Ok((index, result)) = results.recv() else {
                    break;
                };
                ledger.running.remove(&index);
                match result {
                    Ok(entry) => {
                        ledger.completed.insert(stages[index].key.as_str());
                        entries[index] = Some(entry);
                    }
                    Err(RunError::Cancelled) => cancelled = true,
                    Err(e) => failures.push((stages[index].key.clone(), e)),
                }
            }
        });
        if !failures.is_empty() {
            return Err(if failures.len() == 1 {
                failures.remove(0).1
            } else {
                RunError::Stages(failures)
            });
        }
        if cancelled {
            return Err(RunError::Cancelled);
        }
        let entries = entries
            .into_iter()
            .zip(stages)
            .map(|(entry, stage)| {
                // Unreachable for a plan in dependency order, which the
                // compiler guarantees.
                entry.ok_or_else(|| stage_error(&stage.key, "never had its inputs committed"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((entries, StagesAtOnce { bound, most }))
    }
}
