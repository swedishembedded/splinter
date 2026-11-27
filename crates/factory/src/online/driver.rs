// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements online reinforcement-learning controllers
// that never train a group twice or lose one they produced, for its clients.
// If your team needs expertise in asynchronous policy optimisation, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The loop of an online run over its seams.
//!
//! Every change the controller makes is recorded in the ledger before it
//! takes effect: a rollout is recorded running before it starts, its outcome
//! before its group is scored, a group sent only after the learner took it.
//! A controller stopped anywhere therefore resumes from the ledger and the
//! learner's state ([`super::control::Controller::restore`]): it starts no
//! finished rollout again, and sends again only what the learner has not
//! committed, which the learner deduplicates by group key.

use splinter_core::factory::group::GroupKey;
use splinter_store::factory::runstate::{Fence, GroupEntry, NewGroup, RunState};

use super::control::{Change, ControlError, Controller};
use super::seams::{
    ExecutorError, GroupLedger, Learner, LearnerError, LearnerEvent, LedgerError, Member,
    OutgoingGroup, RolloutExecutor, RolloutRequest, RunClock, SendOutcome, TaskSelector,
};
use super::stats::Stats;
use super::watchdog::Verdict;
use super::Settings;

/// Why an online run stopped short.
#[derive(Debug, thiserror::Error)]
pub enum OnlineError {
    /// The executor failed.
    #[error(transparent)]
    Executor(#[from] ExecutorError),
    /// The learner failed.
    #[error(transparent)]
    Learner(#[from] LearnerError),
    /// The ledger failed.
    #[error(transparent)]
    Ledger(#[from] LedgerError),
    /// A seam broke its contract, or the ledger contradicts itself.
    #[error(transparent)]
    Control(#[from] ControlError),
}

/// How a run ended.
#[derive(Clone, Debug, PartialEq)]
pub enum Ended {
    /// Every group it asked for is committed, filtered or dropped.
    Finished,
    /// It was cancelled; resuming continues it.
    Cancelled,
    /// Its watchdog rolled it back: the epoch ends here, and the run goes on
    /// in the next one from the anchor.
    RolledBack(Box<super::watchdog::Audit>),
}

/// What a run did.
#[derive(Clone, Debug, PartialEq)]
pub struct Report {
    /// How it ended.
    pub ended: Ended,
    /// Its statistics at the end.
    pub stats: Stats,
}

/// What a run is carried out with.
pub struct Seams<'a> {
    /// Runs rollouts.
    pub executor: &'a mut dyn RolloutExecutor,
    /// Trains on groups.
    pub learner: &'a mut dyn Learner,
    /// Picks each group's task.
    pub selector: &'a mut dyn TaskSelector,
    /// Records every change.
    pub ledger: &'a mut dyn GroupLedger,
    /// The run's time.
    pub clock: &'a dyn RunClock,
    /// Asked once per turn of the loop: true cancels the run.
    pub cancel: &'a dyn Fn() -> bool,
}

/// Runs (or resumes) an online run until every group it asks for is
/// terminal or it is cancelled.
///
/// # Errors
/// A seam failed or broke its contract; what is recorded stays, and a
/// resume continues from it.
pub fn run(settings: Settings, seams: Seams<'_>) -> Result<Report, OnlineError> {
    run_watched(settings, seams, None)
}

/// [`run`], judged by `watchdog` after every turn: a breached invariant
/// holds new groups back while it lasts, and a rollback stops every running
/// rollout and ends the run [`Ended::RolledBack`].
///
/// # Errors
/// As [`run`].
pub fn run_watched(
    settings: Settings,
    seams: Seams<'_>,
    mut watchdog: Option<&mut super::watchdog::Watchdog<'_>>,
) -> Result<Report, OnlineError> {
    let Seams {
        executor,
        learner,
        selector,
        ledger,
        clock,
        cancel,
    } = seams;
    let poll_ms = settings.poll_ms;
    let learner_state = learner.state()?;
    let entries = ledger.load()?;
    let (mut control, changes) = Controller::restore(settings, entries, &learner_state, selector)?;
    apply(ledger, executor, clock.now_ms(), changes)?;
    let mut finishing = false;
    loop {
        if cancel() {
            executor.cancel();
            return Ok(Report {
                ended: Ended::Cancelled,
                stats: control.stats(clock.now_ms()),
            });
        }
        let now = clock.now_ms();
        for event in learner.poll()? {
            if let LearnerEvent::Published { version } = event {
                ledger.learner_version(version);
            }
            let changes = control.learner_event(event);
            apply(ledger, executor, now, changes)?;
        }
        while let Some(draft) = control.next_send(now) {
            let records = ledger.records(&draft.key)?;
            let group = outgoing(draft, records)?;
            let answer = learner.send(&group)?;
            let changes = control.sent(now, &group.key, &answer)?;
            apply(ledger, executor, now, changes)?;
            if matches!(answer, SendOutcome::Full { .. }) {
                break;
            }
        }
        let paused = match watchdog.as_deref_mut() {
            Some(watchdog) => {
                let empty = super::health::VersionSeries::default();
                match watchdog.verdict(&learner.telemetry(), ledger.series().unwrap_or(&empty)) {
                    Verdict::Continue => false,
                    Verdict::Pause => true,
                    Verdict::Rollback(audit) => {
                        executor.cancel();
                        return Ok(Report {
                            ended: Ended::RolledBack(audit),
                            stats: control.stats(clock.now_ms()),
                        });
                    }
                }
            }
            None => false,
        };
        while let Some(changes) = (!paused).then(|| control.admit(now, selector)).flatten() {
            apply(ledger, executor, now, changes)?;
        }
        if control.done() {
            return Ok(Report {
                ended: Ended::Finished,
                stats: control.stats(clock.now_ms()),
            });
        }
        if control.waits_on_learner() && !finishing {
            learner.finish()?;
            finishing = true;
        }
        for end in executor.wait(now.saturating_add(poll_ms))? {
            let changes = control.rollout_ended(end, selector)?;
            apply(ledger, executor, clock.now_ms(), changes)?;
        }
    }
}

/// Records each change, then starts the rollouts it asks for.
fn apply(
    ledger: &mut dyn GroupLedger,
    executor: &mut dyn RolloutExecutor,
    now: u64,
    changes: Vec<Change>,
) -> Result<(), OnlineError> {
    for change in changes {
        ledger.apply(now, &change)?;
        if let Change::Start { key, task, index } = change {
            executor.start(RolloutRequest {
                group: key,
                task,
                index,
            })?;
        }
    }
    Ok(())
}

/// The draft with its members' records.
fn outgoing(
    draft: super::control::Draft,
    records: Vec<(u32, String)>,
) -> Result<OutgoingGroup, ControlError> {
    let mut by_index: std::collections::HashMap<u32, String> = records.into_iter().collect();
    let members = draft
        .members
        .into_iter()
        .map(|(index, reward)| {
            by_index
                .remove(&index)
                .map(|records| Member {
                    index,
                    reward,
                    records,
                })
                .ok_or_else(|| {
                    ControlError(format!("{}/{index} has no records to send", draft.key))
                })
        })
        .collect::<Result<_, _>>()?;
    Ok(OutgoingGroup {
        key: draft.key,
        task: draft.task,
        members,
    })
}

/// The group ledger in a run directory's run state, under the controller's
/// fence, renewing the lease as the run goes.
pub struct RunLedger<'a> {
    state: &'a RunState,
    fence: Fence,
    ttl_secs: u64,
    renewed_at: u64,
}

impl<'a> RunLedger<'a> {
    /// The ledger of `state`, written under `fence`, whose lease lasts
    /// `ttl_secs` and is renewed when half of it has passed.
    #[must_use]
    pub fn new(state: &'a RunState, fence: Fence, ttl_secs: u64, now_secs: u64) -> Self {
        Self {
            state,
            fence,
            ttl_secs,
            renewed_at: now_secs,
        }
    }
}

fn ledger_error(error: impl std::fmt::Display) -> LedgerError {
    LedgerError(error.to_string())
}

impl GroupLedger for RunLedger<'_> {
    fn load(&mut self) -> Result<Vec<GroupEntry>, LedgerError> {
        self.state.group_entries().map_err(ledger_error)
    }

    fn apply(&mut self, now_ms: u64, change: &Change) -> Result<(), LedgerError> {
        let now = now_ms / 1000;
        if now.saturating_sub(self.renewed_at) * 2 >= self.ttl_secs {
            self.state
                .renew(&self.fence, self.ttl_secs, now)
                .map_err(ledger_error)?;
            self.renewed_at = now;
        }
        let (state, fence) = (self.state, &self.fence);
        match change {
            Change::Create {
                key,
                seq,
                task,
                size,
                epoch,
            } => state
                .create_group(
                    fence,
                    &NewGroup {
                        key,
                        seq: *seq,
                        task,
                        size: *size,
                        epoch: *epoch,
                    },
                    now,
                )
                .map(drop),
            Change::Start { key, index, .. } => state.start_rollout(fence, key, *index, now),
            Change::Finish {
                key,
                index,
                outcome,
                records,
            } => state.finish_rollout(fence, key, *index, outcome, records, now),
            Change::Fail {
                key,
                index,
                abandon,
                reason,
            } => state.fail_rollout(fence, key, *index, *abandon, reason, now),
            Change::State {
                key,
                to,
                reason,
                staleness,
            } => state.set_group_state(fence, key, *to, *reason, *staleness, now),
        }
        .map_err(ledger_error)
    }

    fn records(&mut self, key: &GroupKey) -> Result<Vec<(u32, String)>, LedgerError> {
        self.state.group_records(key).map_err(ledger_error)
    }
}
