// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The watchdog's three layers against scripted telemetry and a scripted
//! evaluator.

use std::cell::RefCell;
use std::rc::Rc;

use super::*;
use crate::online::seams::PublishedVersion;

/// Grades every task with the pass probabilities of the version asked, by a
/// fixed pattern per task index: deterministic, no randomness.
struct Scripted {
    pass: Rc<RefCell<Vec<(u64, f64)>>>,
    queue: Vec<Pair>,
    started: Rc<RefCell<Vec<(u64, u64, usize)>>>,
}

impl Evaluator for Scripted {
    fn start(&mut self, candidate: &Version, anchor: &Version, tasks: Vec<String>) {
        self.started
            .borrow_mut()
            .push((candidate.version, anchor.version, tasks.len()));
        let rate = |v: u64| {
            self.pass
                .borrow()
                .iter()
                .rev()
                .find(|(from, _)| *from <= v)
                .map_or(0.5, |(_, p)| *p)
        };
        let (c, a) = (rate(candidate.version), rate(anchor.version));
        for (i, task) in tasks.into_iter().enumerate() {
            // A low-discrepancy pattern: task i passes when its place in [0, 1) is below the rate.
            #[allow(clippy::cast_precision_loss)]
            let at = (i as f64 * 0.618_033_988_75).fract();
            #[allow(clippy::cast_precision_loss)]
            let other = (i as f64 * 0.414_213_562_37 + 0.5).fract();
            self.queue.push(Pair {
                task,
                candidate: at < c,
                anchor: other < a,
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

fn settings() -> WatchdogSettings {
    WatchdogSettings {
        bounds: Bounds {
            guard_skip_rate: 0.5,
            ratio_max: 20.0,
            truncation_rate: 0.5,
            infra_failure_rate: 0.5,
            window: 4,
            min_rollouts: 4,
        },
        drift: DriftSettings {
            slack: 0.05,
            threshold: 0.3,
            reference_window: 3,
        },
        test: TestSettings {
            delta: 0.25,
            alpha: 0.05,
            beta: 0.1,
            max_discordant: 200,
        },
        evaluate_every: 2,
        monitor_size: 200,
        max_uses: 4,
        max_rollbacks: 2,
        cooldown: 0.5,
    }
}

struct World {
    watchdog: Watchdog<'static>,
    telemetry: LearnerTelemetry,
    series: VersionSeries,
    pass: Rc<RefCell<Vec<(u64, f64)>>>,
    started: Rc<RefCell<Vec<(u64, u64, usize)>>>,
}

fn world() -> World {
    let pass = Rc::new(RefCell::new(vec![(0, 0.6)]));
    let started = Rc::new(RefCell::new(Vec::new()));
    let evaluator = Scripted {
        pass: pass.clone(),
        queue: Vec::new(),
        started: started.clone(),
    };
    let monitor: Vec<String> = (0..400).map(|i| format!("m{i}")).collect();
    let confirm: Vec<String> = (0..200).map(|i| format!("c{i}")).collect();
    World {
        // seed: the monitor draws are checked by count; any seed serves.
        watchdog: Watchdog::new(
            settings(),
            Box::new(evaluator),
            MonitorPool::new(monitor, 4, 1),
            confirm,
            None,
            0,
        ),
        telemetry: LearnerTelemetry::default(),
        series: VersionSeries::default(),
        pass,
        started,
    }
}

impl World {
    fn publish(&mut self, version: u64) {
        self.telemetry.published.push(PublishedVersion {
            version,
            digest: format!("blake3:{version:064}"),
            step: version,
            lineage: None,
            version_id: None,
        });
    }

    /// `n` rollouts of `version` with this reward and false-completion share.
    fn rollouts(&mut self, version: u64, n: u32, reward: f64, false_completions: u32) {
        use splinter_core::factory::group::{Behaviour, RolloutOutcome, VersionSpan};
        let outcomes: Vec<RolloutOutcome> = (0..n)
            .map(|i| RolloutOutcome {
                reward: Some(if i < false_completions { 0.0 } else { reward }),
                tokens: 1,
                span: VersionSpan::at(version),
                exact: true,
                truncated: false,
                truncation: None,
                behaviour: Some(Behaviour {
                    stopped: true,
                    tool_calls: 1,
                    repeated_calls: 0,
                }),
            })
            .collect();
        self.series
            .add_group(&outcomes.iter().collect::<Vec<_>>(), 0);
    }

    fn turn(&mut self) -> Verdict {
        self.watchdog.verdict(&self.telemetry, &self.series)
    }
}

#[test]
fn a_regressed_candidate_is_rolled_back_with_its_audit_and_an_equal_one_is_promoted_after_its_confirm(
) {
    let mut w = world();
    w.publish(1);
    assert_eq!(
        w.turn(),
        Verdict::Continue,
        "the first version is the anchor"
    );
    w.publish(2);
    w.publish(3);
    assert_eq!(
        w.turn(),
        Verdict::Continue,
        "an equal candidate: confirm, then promote"
    );
    assert_eq!(w.turn(), Verdict::Continue);
    assert_eq!(w.watchdog.anchor().map(|a| a.version), Some(3), "promoted");
    assert_eq!(
        *w.started.borrow(),
        [(3, 1, 200), (3, 1, 200)],
        "the look, then the confirm set"
    );

    // From version 4 on the policy passes far less.
    w.pass.borrow_mut().push((4, 0.1));
    w.publish(4);
    w.publish(5);
    let Verdict::Rollback(audit) = w.turn() else {
        panic!("a clear regression rolls back");
    };
    assert_eq!((audit.candidate.version, audit.anchor.version), (5, 3));
    assert_eq!(audit.evidence.decision, Decision::Regressed);
    assert!(audit.evidence.anchor_wins > audit.evidence.candidate_wins);
    assert_eq!(audit.trigger, json!({"schedule": 5}));
}

#[test]
fn an_invariant_pauses_intake_and_never_rolls_back_and_drift_asks_for_an_evaluation_at_once() {
    let mut w = world();
    w.publish(1);
    w.rollouts(1, 8, 1.0, 1);
    assert_eq!(w.turn(), Verdict::Continue);
    w.publish(2);
    w.watchdog.last_evaluated = 2;
    w.telemetry.step_health = (1..=4)
        .map(|step| StepHealth {
            step,
            version: 1,
            ratio_max: Some(50.0),
            ..StepHealth::default()
        })
        .collect();
    assert_eq!(
        w.turn(),
        Verdict::Pause,
        "a ratio past its bound holds intake while an evaluation rules"
    );
    assert!(w.watchdog.report()["alarms"][0]["alarm"] == "invariant");
    assert_eq!(w.turn(), Verdict::Continue, "ruled on: intake goes on");
    assert_eq!(
        w.started.borrow().len(),
        2,
        "the breach asked for an evaluation (and its confirm), and rolled nothing back"
    );
    assert_eq!(
        w.turn(),
        Verdict::Continue,
        "a breach already ruled on holds nothing"
    );
    w.started.borrow_mut().clear();
    w.watchdog.anchor = Some(Version {
        version: 1,
        digest: String::new(),
        id: None,
    });

    // Version 3 is not yet due for an evaluation (every 2 after version 2),
    // but its false-completion rate jumped: drift asks for one now.
    w.publish(3);
    w.rollouts(3, 8, 1.0, 6);
    assert_eq!(w.turn(), Verdict::Continue);
    assert_eq!(
        w.started.borrow().len(),
        2,
        "evaluated at once (a look and its confirm)"
    );
    let report = w.watchdog.report();
    assert!(
        report["alarms"]
            .as_array()
            .unwrap()
            .iter()
            .any(|a| a["alarm"] == "drift" && a["detail"]["signal"] == "false_completion"),
        "{report:#}"
    );
}
