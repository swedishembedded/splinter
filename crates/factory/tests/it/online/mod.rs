// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements online reinforcement-learning controllers
// that never train a group twice or lose one they produced, for its clients.
// If your team needs expertise in asynchronous policy optimisation, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Online runs: the controller core against hand arithmetic, and the
//! driver over scripted doubles, killed and resumed.

mod brain;
mod doubles;
mod simulation;

use splinter_core::factory::group::{GroupKey, GroupState, RolloutOutcome, VersionSpan};
use splinter_factory::online::control::{reasons, Change, Controller};
use splinter_factory::online::driver::{run, Ended, RunLedger, Seams};
use splinter_factory::online::seams::{
    FinishedRollout, LearnerEvent, RolloutEnd, RoundRobin, SendOutcome,
};
use splinter_factory::online::Settings;
use splinter_store::factory::runstate::RunState;

use doubles::{Clock, Executor, LearnerPlan, MemoryLedger, Plan, ScriptedLearner};

fn settings() -> Settings {
    Settings {
        namespace: "online".into(),
        group_size: 2,
        max_staleness: 1,
        rollout_slots: 8,
        tokens_per_step: 100,
        rollout_tokens: 30,
        step_window: 2,
        max_failures: 0,
        send_retry_ms: 500,
        poll_ms: 50,
    }
}

fn tasks(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("task-{i}")).collect()
}

fn key(seq: u64) -> GroupKey {
    GroupKey::numbered("online", seq).unwrap()
}

fn outcome(reward: Option<f64>, tokens: u64, version: u64, exact: bool) -> RolloutOutcome {
    RolloutOutcome {
        reward,
        tokens,
        span: VersionSpan::at(version),
        exact,
        truncated: false,
    }
}

fn finished(seq: u64, index: u32, outcome: RolloutOutcome) -> RolloutEnd {
    RolloutEnd {
        group: key(seq),
        index,
        result: Ok(FinishedRollout {
            outcome,
            records: String::new(),
        }),
    }
}

fn failed(seq: u64, index: u32) -> RolloutEnd {
    RolloutEnd {
        group: key(seq),
        index,
        result: Err("worker lost".into()),
    }
}

/// Admits every rollout the controller lets start at `now`.
fn admit_all(control: &mut Controller, now: u64, selector: &mut RoundRobin) -> usize {
    let mut started = 0;
    while let Some(changes) = control.admit(now, selector) {
        started += changes
            .iter()
            .filter(|c| matches!(c, Change::Start { .. }))
            .count();
    }
    started
}

fn states(changes: &[Change]) -> Vec<(GroupState, Option<&'static str>)> {
    changes
        .iter()
        .filter_map(|c| match c {
            Change::State { to, reason, .. } => Some((*to, *reason)),
            _ => None,
        })
        .collect()
}

#[test]
fn reward_statistics_count_every_scored_group_and_filter_what_cannot_teach() {
    let mut selector = RoundRobin::new(tasks(1), 5).unwrap();
    let mut control = Controller::new(
        Settings {
            rollout_slots: 16,
            tokens_per_step: 1_000,
            ..settings()
        },
        0,
    )
    .unwrap();
    assert_eq!(
        admit_all(&mut control, 0, &mut selector),
        10,
        "five groups within 1000 tokens"
    );
    let decided = |c: &[Change]| states(c).last().copied();
    let mut end = |end: RolloutEnd| control.rollout_ended(end, &mut selector).unwrap();
    end(finished(0, 0, outcome(Some(1.0), 10, 0, true)));
    let a = end(finished(0, 1, outcome(Some(1.0), 10, 0, true)));
    assert_eq!(
        decided(&a),
        Some((GroupState::Filtered, Some(reasons::ZERO_VARIANCE)))
    );

    let mut end = |end: RolloutEnd| control.rollout_ended(end, &mut selector).unwrap();
    end(finished(1, 0, outcome(Some(0.0), 10, 0, true)));
    let b = end(finished(1, 1, outcome(Some(1.0), 10, 0, true)));
    assert_eq!(
        states(&b),
        [(GroupState::Scored, None)],
        "B waits to be sent"
    );
    end(finished(2, 0, outcome(None, 10, 0, true)));
    let c = end(finished(2, 1, outcome(Some(0.5), 10, 0, true)));
    assert_eq!(
        decided(&c),
        Some((GroupState::Filtered, Some(reasons::ZERO_VARIANCE))),
        "one score cannot differ"
    );
    end(finished(3, 0, outcome(Some(0.2), 10, 0, false)));
    let d = end(finished(3, 1, outcome(Some(0.8), 10, 0, true)));
    assert_eq!(
        decided(&d),
        Some((GroupState::Filtered, Some(reasons::NOT_EXACT)))
    );
    end(failed(4, 0));
    let e = end(failed(4, 1));
    assert_eq!(
        decided(&e),
        Some((GroupState::Filtered, Some(reasons::UNSCORED)))
    );

    let stats = control.stats(2);
    // Scored rewards 1, 1, 0, 1, 0.5, 0.2, 0.8: sum 4.5, sum of squares 3.93.
    assert_eq!(stats.reward.count(), 7);
    assert!((stats.reward.mean().unwrap() - 4.5 / 7.0).abs() < 1e-12);
    assert!((stats.reward.std().unwrap() - 7.26_f64.sqrt() / 7.0).abs() < 1e-12);
    // Zero-variance: A (mean 1) and C (mean 0.5) of the four groups with a score.
    assert_eq!(stats.frac_zero_std(), Some(0.5));
    assert_eq!(stats.zero_std_reward.mean(), Some(0.75));
    assert_eq!(
        (
            stats.filtered_zero_variance,
            stats.filtered_not_exact,
            stats.filtered_unscored
        ),
        (2, 1, 1)
    );
    assert_eq!(
        (
            stats.rollouts_unscored,
            stats.rollouts_not_exact,
            stats.rollouts_abandoned,
            stats.rollout_failures
        ),
        (1, 1, 2, 2)
    );
    assert_eq!(
        stats.queue_tokens, 20,
        "only B's tokens wait for the learner"
    );
    let draft = control.next_send(2).unwrap();
    assert_eq!(
        (draft.key, draft.members),
        (key(1), vec![(0, 0.0), (1, 1.0)])
    );
}

#[test]
fn a_group_more_than_max_staleness_behind_the_learner_is_dropped_not_sent() {
    let mut selector = RoundRobin::new(tasks(2), 2).unwrap();
    let mut control = Controller::new(settings(), 0).unwrap();
    admit_all(&mut control, 0, &mut selector);
    control
        .rollout_ended(
            finished(0, 0, outcome(Some(0.0), 10, 0, true)),
            &mut selector,
        )
        .unwrap();
    admit_all(&mut control, 0, &mut selector);
    // The learner moves two versions on while group 0's last rollout runs.
    assert!(control
        .learner_event(LearnerEvent::Published { version: 2 })
        .is_empty());
    let late = control
        .rollout_ended(
            finished(0, 1, outcome(Some(1.0), 10, 2, true)),
            &mut selector,
        )
        .unwrap();
    assert_eq!(
        states(&late).last(),
        Some(&(GroupState::Stale, Some(reasons::STALE_HERE))),
        "oldest request at 0, learner at 2"
    );

    // A scored group waiting to be sent is dropped when the learner publishes past it.
    control
        .rollout_ended(
            finished(1, 0, outcome(Some(0.0), 10, 1, true)),
            &mut selector,
        )
        .unwrap();
    control
        .rollout_ended(
            finished(1, 1, outcome(Some(1.0), 10, 2, true)),
            &mut selector,
        )
        .unwrap();
    assert!(control.next_send(0).is_some());
    let dropped = control.learner_event(LearnerEvent::Published { version: 3 });
    assert_eq!(
        states(&dropped),
        [(GroupState::Stale, Some(reasons::STALE_HERE))]
    );
    assert!(control.next_send(0).is_none());
    assert_eq!(control.stats(0).dropped_stale, 2);
}

#[test]
fn the_token_allowance_holds_new_groups_back_until_the_learner_applies_and_never_stops_a_running_rollout(
) {
    // max_staleness 1, step budget 100: the allowance is 100 tokens; a
    // rollout is assumed to sample 30 until one has finished.
    let mut selector = RoundRobin::new(tasks(3), 10).unwrap();
    let mut control = Controller::new(settings(), 0).unwrap();
    // Group 0 at 0 tokens, group 1 at 2 * 30 = 60; group 2 would be 120.
    assert_eq!(admit_all(&mut control, 0, &mut selector), 4);
    assert_eq!(control.stats(0).allowance, 100);
    for index in 0..2 {
        let reward = f64::from(index);
        control
            .rollout_ended(
                finished(0, index, outcome(Some(reward), 30, 0, true)),
                &mut selector,
            )
            .unwrap();
    }
    // 60 queued + 2 running at the mean of 30 = 120.
    assert!(control.admit(1_000, &mut selector).is_none());
    let draft = control.next_send(1_000).unwrap();
    control
        .sent(1_000, &draft.key, &SendOutcome::Accepted)
        .unwrap();
    for index in 0..2 {
        let reward = f64::from(index);
        control
            .rollout_ended(
                finished(1, index, outcome(Some(reward), 30, 0, true)),
                &mut selector,
            )
            .unwrap();
    }
    assert!(
        control.admit(2_000, &mut selector).is_none(),
        "120 tokens queued"
    );
    assert_eq!(control.stats(2_000).queue_tokens, 120);
    assert_eq!(control.stats(2_000).in_flight, 0);

    control.learner_event(LearnerEvent::Applied {
        groups: vec![key(0)],
    });
    assert_eq!(
        admit_all(&mut control, 3_000, &mut selector),
        2,
        "60 queued: group 2 starts, group 3 waits"
    );
    assert_eq!(
        control.stats(3_000).backpressure_ms,
        3_000,
        "held from 0 to 3000"
    );
    assert_eq!(
        control.stats(3_500).backpressure_ms,
        3_500,
        "and again since 3000"
    );
    assert_eq!(control.stats(3_500).in_flight, 2);

    // Steps of 300 tokens raise the estimate, and with it the allowance.
    control.learner_event(LearnerEvent::Step { tokens: 300 });
    assert_eq!(control.stats(4_000).allowance, 300);
    control.learner_event(LearnerEvent::Step { tokens: 50 });
    assert_eq!(
        control.stats(4_000).allowance,
        175,
        "mean of the last two steps"
    );
    control.learner_event(LearnerEvent::Step { tokens: 50 });
    assert_eq!(
        control.stats(4_000).allowance,
        100,
        "never below the step budget"
    );
}

#[test]
fn a_full_learner_queue_holds_sends_and_the_same_group_is_sent_again() {
    let mut selector = RoundRobin::new(tasks(1), 1).unwrap();
    let mut control = Controller::new(settings(), 0).unwrap();
    admit_all(&mut control, 0, &mut selector);
    for index in 0..2 {
        control
            .rollout_ended(
                finished(0, index, outcome(Some(f64::from(index)), 30, 0, true)),
                &mut selector,
            )
            .unwrap();
    }
    let draft = control.next_send(0).unwrap();
    let full = SendOutcome::Full {
        allowance_tokens: 64,
    };
    assert!(control.sent(0, &draft.key, &full).unwrap().is_empty());
    assert!(control.next_send(499).is_none(), "held for send_retry_ms");
    assert_eq!(control.next_send(500).map(|d| d.key), Some(key(0)));
    control.sent(500, &key(0), &full).unwrap();
    control.learner_event(LearnerEvent::Applied { groups: vec![] });
    assert_eq!(
        control.next_send(501).map(|d| d.key),
        Some(key(0)),
        "an application frees room"
    );
    let sent = control.sent(501, &key(0), &SendOutcome::Accepted).unwrap();
    assert_eq!(states(&sent), [(GroupState::Sent, None)]);
    assert!(control.next_send(501).is_none());
    let stats = control.stats(501);
    assert_eq!(
        (
            stats.learner_full,
            stats.learner_allowance,
            stats.groups_sent
        ),
        (2, Some(64), 1)
    );
}

fn script(slow_group: u64) -> doubles::Script {
    Box::new(move |request, _| {
        let seq: u64 = request
            .group
            .as_str()
            .rsplit(':')
            .next()
            .unwrap()
            .parse()
            .unwrap();
        Plan {
            duration_ms: if seq == slow_group {
                10_000
            } else {
                100 + 10 * u64::from(request.index)
            },
            result: Ok((Some(f64::from(request.index)), 40, true)),
        }
    })
}

fn learner_plan() -> LearnerPlan {
    LearnerPlan {
        step_tokens: 80,
        step_ms: 200,
        publish_every: 1,
        commit_every: 2,
        capacity: 400,
        max_staleness: 100,
    }
}

#[test]
fn groups_reach_the_learner_as_they_complete_with_no_barrier() {
    let clock = Clock::default();
    let mut learner = ScriptedLearner::new(clock.clone(), learner_plan());
    let mut executor = Executor::new(clock.clone(), learner.served.clone(), script(0));
    let mut selector = RoundRobin::new(tasks(4), 6).unwrap();
    let mut ledger = MemoryLedger::default();
    let report = run(
        Settings {
            max_staleness: 100,
            ..settings()
        },
        Seams {
            executor: &mut executor,
            learner: &mut learner,
            selector: &mut selector,
            ledger: &mut ledger,
            clock: &clock,
            cancel: &|| false,
        },
    )
    .unwrap();
    assert_eq!(report.ended, Ended::Finished);
    let accepted: Vec<_> = learner
        .sends
        .iter()
        .filter(|(_, _, a)| *a == SendOutcome::Accepted)
        .map(|(at, key, _)| (*at, key.clone()))
        .collect();
    let slow = accepted.iter().find(|(_, k)| *k == key(0)).unwrap().0;
    assert!(slow >= 10_000);
    assert_eq!(accepted.len(), 6);
    assert!(
        accepted.iter().filter(|(at, _)| *at < 10_000).count() == 5,
        "the five fast groups were sent while the slow one ran: {accepted:?}"
    );
    assert_eq!(learner.committed.len(), 6);
    assert_eq!(report.stats.groups_committed, 6);
    assert!(
        executor.peak_running() <= 8,
        "never more rollouts than slots"
    );
    assert_eq!(
        learner.queued_tokens(),
        0,
        "the learner trained everything it took"
    );
}

#[test]
fn a_killed_controller_resumes_without_repeating_a_finished_rollout_or_committing_a_group_twice() {
    let dir = tempfile::tempdir().unwrap();
    let clock = Clock::default();
    let mut learner = ScriptedLearner::new(clock.clone(), learner_plan());
    let mut selector = RoundRobin::new(tasks(3), 8).unwrap();
    let settings = Settings {
        max_staleness: 100,
        ..settings()
    };
    let mut first_ends = Vec::new();

    // Killed twice: the executor dies after 7 rollout ends, then the
    // learner's connection drops right after it took a group.
    for round in 0..3 {
        let state = RunState::open(dir.path()).unwrap();
        let now = clock.now() / 1000;
        let owner = format!("controller-{round}");
        let fence = if round == 0 {
            state.acquire(&owner, 30, now).unwrap()
        } else {
            state
                .take_over(&owner, &format!("controller-{}", round - 1), 30, now)
                .unwrap()
        };
        let mut ledger = RunLedger::new(&state, fence, 30, now);
        let mut executor = Executor::new(clock.clone(), learner.served.clone(), script(1));
        match round {
            0 => executor.die_after = Some(7),
            1 => learner.die_after_accepts = Some(learner.accepts + 1),
            _ => learner.die_after_accepts = None,
        }
        let result = run(
            settings.clone(),
            Seams {
                executor: &mut executor,
                learner: &mut learner,
                selector: &mut selector,
                ledger: &mut ledger,
                clock: &clock,
                cancel: &|| false,
            },
        );
        let finished_here: Vec<_> = executor
            .ends
            .iter()
            .filter(|(_, e)| e.result.is_ok())
            .map(|(_, e)| (e.group.clone(), e.index))
            .collect();
        for start in &executor.starts {
            assert!(
                !first_ends.contains(&(start.group.clone(), start.index)),
                "round {round} started {}/{}, which had finished",
                start.group,
                start.index
            );
        }
        assert_eq!(
            executor.starts.len(),
            executor.ends.len() + executor.running(),
            "every start ended or was running when the controller stopped"
        );
        first_ends.extend(finished_here);
        if round < 2 {
            assert!(result.is_err(), "round {round} was killed");
        } else {
            let report = result.unwrap();
            assert_eq!(report.ended, Ended::Finished);
            assert_eq!(report.stats.groups_committed, 8);
            assert!(
                report.stats.resends >= 1,
                "the group taken before the drop was sent again"
            );
        }
    }
    assert_eq!(first_ends.len(), 16, "every rollout finished exactly once");
    assert_eq!(learner.commits.len(), 8);
    assert!(
        learner.commits.values().all(|n| *n == 1),
        "{:?}",
        learner.commits
    );
    let state = RunState::open(dir.path()).unwrap();
    assert!(state
        .group_entries()
        .unwrap()
        .iter()
        .all(|e| e.group.state == GroupState::Committed));
}

#[test]
fn a_resumed_controller_sends_again_what_a_restarted_learner_had_not_committed() {
    let clock = Clock::default();
    let mut learner = ScriptedLearner::new(
        clock.clone(),
        LearnerPlan {
            commit_every: 1_000,
            ..learner_plan()
        },
    );
    let mut selector = RoundRobin::new(tasks(2), 4).unwrap();
    let mut ledger = MemoryLedger::default();
    let stop_at = std::cell::Cell::new(Some(400u64));
    let settings = Settings {
        max_staleness: 100,
        ..settings()
    };
    for _ in 0..2 {
        let mut executor = Executor::new(clock.clone(), learner.served.clone(), script(u64::MAX));
        let clock_ref = &clock;
        let cancel = || stop_at.get().is_some_and(|at| clock_ref.now() >= at);
        let report = run(
            settings.clone(),
            Seams {
                executor: &mut executor,
                learner: &mut learner,
                selector: &mut selector,
                ledger: &mut ledger,
                clock: &clock,
                cancel: &cancel,
            },
        )
        .unwrap();
        if stop_at.take().is_some() {
            assert_eq!(report.ended, Ended::Cancelled);
            assert!(report.stats.groups_applied > 0 && report.stats.groups_committed == 0);
            learner.restart();
        } else {
            assert_eq!(report.ended, Ended::Finished);
            assert_eq!(report.stats.groups_committed, 4);
        }
    }
    assert!(learner.commits.values().all(|n| *n == 1));
    assert_eq!(learner.commits.len(), 4);
}
