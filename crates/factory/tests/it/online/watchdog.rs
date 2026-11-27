// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training loops that notice a model moving the
// wrong way and roll it back on evidence, for its clients. If your team needs
// expertise in safe continuous training, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Spec: a watched online run rolls back on evidence and recovers. The
//! learner publishes versions while it trains; from one version on the
//! policy regresses on the held-out tasks. The watchdog's paired evaluation
//! decides a rollback; the run ends its epoch there, every running rollout
//! stopped. Resumed in the next epoch with a fresh learner (from the anchor),
//! the controller rolls back every group of the earlier epoch that was not
//! committed, never sends one, and trains the new epoch's groups to the end.

use std::cell::RefCell;
use std::rc::Rc;

use splinter_core::factory::group::GroupState;
use splinter_factory::online::driver::{run, run_watched, Ended, RunLedger, Seams};
use splinter_factory::online::seams::{
    GroupLedger, Learner, LearnerError, LearnerEvent, LearnerState, LearnerTelemetry,
    OutgoingGroup, PublishedVersion, RoundRobin, SendOutcome,
};
use splinter_factory::online::watchdog::monitor::MonitorPool;
use splinter_factory::online::watchdog::sequential::{Decision, TestSettings};
use splinter_factory::online::watchdog::signals::{Bounds, DriftSettings};
use splinter_factory::online::watchdog::{Evaluator, Pair, Version, Watchdog, WatchdogSettings};
use splinter_factory::online::Settings;
use splinter_store::factory::runstate::RunState;

use super::doubles::{Clock, Executor, ScriptedLearner};
use super::{learner_plan, script, settings, tasks};

/// The scripted learner, reporting the versions it publishes.
struct Telemetered<'a> {
    inner: &'a mut ScriptedLearner,
    published: Vec<PublishedVersion>,
}

impl Learner for Telemetered<'_> {
    fn state(&mut self) -> Result<LearnerState, LearnerError> {
        self.inner.state()
    }
    fn send(&mut self, group: &OutgoingGroup) -> Result<SendOutcome, LearnerError> {
        self.inner.send(group)
    }
    fn poll(&mut self) -> Result<Vec<LearnerEvent>, LearnerError> {
        let events = self.inner.poll()?;
        for event in &events {
            if let LearnerEvent::Published { version } = event {
                self.published.push(PublishedVersion {
                    version: *version,
                    digest: format!("blake3:{version:064}"),
                    step: *version,
                    lineage: None,
                    version_id: None,
                });
            }
        }
        Ok(events)
    }
    fn finish(&mut self) -> Result<(), LearnerError> {
        self.inner.finish()
    }
    fn telemetry(&self) -> LearnerTelemetry {
        LearnerTelemetry {
            published: self.published.clone(),
            ..LearnerTelemetry::default()
        }
    }
}

/// Grades a version as its pass rate says: versions from `from` on pass
/// `bad` of the tasks, the others `good`, by a fixed pattern per task.
struct Graded {
    from: u64,
    queue: Vec<Pair>,
    started: Rc<RefCell<Vec<(u64, u64)>>>,
}

impl Evaluator for Graded {
    fn start(&mut self, candidate: &Version, anchor: &Version, tasks: Vec<String>) {
        self.started
            .borrow_mut()
            .push((candidate.version, anchor.version));
        let rate = |v: u64| if v >= self.from { 0.1 } else { 0.6 };
        for (i, task) in tasks.into_iter().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let (x, y) = (
                (i as f64 * 0.618_033_988_75).fract(),
                (i as f64 * 0.414_213_562_37 + 0.5).fract(),
            );
            self.queue.push(Pair {
                task,
                candidate: x < rate(candidate.version),
                anchor: y < rate(anchor.version),
            });
        }
    }
    fn poll(&mut self) -> Vec<Pair> {
        std::mem::take(&mut self.queue)
    }
    fn idle(&self) -> bool {
        self.queue.is_empty()
    }
}

fn watchdog_settings() -> WatchdogSettings {
    WatchdogSettings {
        bounds: Bounds {
            guard_skip_rate: 0.5,
            ratio_max: 100.0,
            truncation_rate: 0.9,
            infra_failure_rate: 0.9,
            window: 8,
            min_rollouts: 1_000,
        },
        drift: DriftSettings {
            slack: 0.05,
            threshold: 0.5,
            reference_window: 3,
        },
        test: TestSettings {
            delta: 0.25,
            alpha: 0.05,
            beta: 0.1,
            max_discordant: 200,
        },
        evaluate_every: 1,
        monitor_size: 100,
        max_uses: 100,
        max_rollbacks: 2,
        cooldown: 0.5,
    }
}

#[test]
fn a_watched_run_rolls_back_on_a_regression_and_its_next_epoch_trains_only_its_own_groups() {
    let dir = tempfile::tempdir().unwrap();
    let clock = Clock::default();
    let started = Rc::new(RefCell::new(Vec::new()));
    let mut selector = RoundRobin::new(tasks(3), 40).unwrap();
    let base = Settings {
        max_staleness: 100,
        ..settings()
    };

    // Epoch 0, watched: from version 3 on the policy regresses on held-out tasks.
    let state = RunState::open(dir.path()).unwrap();
    let fence = state.acquire("controller-0", 30, 0).unwrap();
    let mut ledger = RunLedger::new(&state, fence, 30, 0);
    let mut first = ScriptedLearner::new(clock.clone(), learner_plan());
    let mut executor = Executor::new(clock.clone(), first.served.clone(), script(99));
    let monitor: Vec<String> = (0..300).map(|i| format!("held-{i}")).collect();
    let confirm: Vec<String> = (0..100).map(|i| format!("sealed-{i}")).collect();
    let mut watchdog = Watchdog::new(
        watchdog_settings(),
        Box::new(Graded {
            from: 3,
            queue: Vec::new(),
            started: started.clone(),
        }),
        // seed: the monitor draws are checked by what they decide, not which tasks they are.
        MonitorPool::new(monitor, 100, 1),
        confirm,
        None,
        0,
    );
    let mut learner = Telemetered {
        inner: &mut first,
        published: Vec::new(),
    };
    let report = run_watched(
        base.clone(),
        Seams {
            executor: &mut executor,
            learner: &mut learner,
            selector: &mut selector,
            ledger: &mut ledger,
            clock: &clock,
            cancel: &|| false,
        },
        Some(&mut watchdog),
    )
    .unwrap();
    let Ended::RolledBack(audit) = report.ended else {
        panic!("the regression is rolled back: {:?}", report.ended);
    };
    assert_eq!(audit.evidence.decision, Decision::Regressed);
    assert!(audit.candidate.version >= 3, "{audit:?}");
    assert!(
        audit.anchor.version < 3,
        "the anchor is a good version: {audit:?}"
    );
    assert!(
        started.borrow().iter().any(|(c, _)| *c < 3),
        "a good version was evaluated (and promoted) before: {:?}",
        started.borrow()
    );
    assert_eq!(executor.running(), 0, "every running rollout was stopped");
    let epoch_zero: Vec<_> = ledger.load().unwrap();
    let untrained: Vec<_> = epoch_zero
        .iter()
        .filter(|e| !e.group.state.is_terminal())
        .map(|e| e.group.key.clone())
        .collect();
    assert!(!untrained.is_empty(), "the rollback left groups in flight");

    // Epoch 1: a fresh learner from the anchor; the earlier epoch's untrained groups are never sent.
    let state = RunState::open(dir.path()).unwrap();
    let fence = state
        .take_over("controller-1", "controller-0", 30, clock.now() / 1000)
        .unwrap();
    let mut ledger = RunLedger::new(&state, fence, 30, clock.now() / 1000);
    let mut second = ScriptedLearner::new(clock.clone(), learner_plan());
    let mut executor = Executor::new(clock.clone(), second.served.clone(), script(99));
    let report = run(
        Settings { epoch: 1, ..base },
        Seams {
            executor: &mut executor,
            learner: &mut second,
            selector: &mut selector,
            ledger: &mut ledger,
            clock: &clock,
            cancel: &|| false,
        },
    )
    .unwrap();
    assert_eq!(report.ended, Ended::Finished);
    assert_eq!(report.stats.groups_rolled_back, untrained.len() as u64);
    for (_, key, _) in &second.sends {
        assert!(
            !untrained.contains(key),
            "{key} of the rolled-back epoch was sent"
        );
    }
    let entries = state.group_entries().unwrap();
    assert!(entries.iter().all(|e| e.group.state.is_terminal()));
    for entry in &entries {
        if untrained.contains(&entry.group.key) {
            assert_eq!(entry.group.state, GroupState::RolledBack);
        }
    }
    assert!(
        entries
            .iter()
            .any(|e| e.group.epoch == 1 && e.group.state == GroupState::Committed),
        "the new epoch trained"
    );
}
