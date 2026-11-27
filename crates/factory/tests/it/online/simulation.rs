// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements online reinforcement-learning controllers
// that never train a group twice or lose one they produced, for its clients.
// If your team needs expertise in asynchronous policy optimisation, you can
// procure our services by sending an email to info@swedishembedded.com.

//! A discrete-event simulation of online runs: at least 100,000 rollouts
//! over slow and fast generation and training, heavy-tailed durations,
//! groups whose rewards are all equal, frequent publication, verifier
//! outages, worker failures, controller kills, learner restarts and
//! cancellations. Every run must finish (a run that records no change for
//! far longer than any rollout or step takes is a deadlock), keep its queues
//! bounded, commit no group twice, and leave every
//! group and rollout in an accounted terminal state.

use std::cell::Cell;

use splinter_core::factory::group::{GroupState, RolloutState};
use splinter_core::rng::SplitMix64;
use splinter_factory::online::driver::{run, Ended, OnlineError, Seams};
use splinter_factory::online::seams::RoundRobin;
use splinter_factory::online::Settings;

use super::doubles::{Clock, Executor, LearnerPlan, MemoryLedger, Plan, ScriptedLearner};

/// Simulated time without a recorded change after which a run counts as
/// deadlocked: ten times the longest rollout, far past any training step.
const STALL_MS: u64 = 100_000_000;

/// Rollouts the simulation must finish at least.
const EPISODES: usize = 100_000;

#[derive(Clone)]
struct World {
    settings: Settings,
    plan: LearnerPlan,
    groups: u64,
    tasks: usize,
    all_equal: bool,
    p_fail: f64,
    p_unscored: f64,
    p_inexact: f64,
    outage: Option<(u64, u64)>,
    base_ms: u64,
    max_tokens: u64,
}

fn unit(rng: &mut SplitMix64) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    let x = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
    x
}

fn world(rng: &mut SplitMix64) -> World {
    let max_staleness = rng.between(0, 4);
    let fast_training = rng.coin();
    let max_tokens = rng.between(50, 2_000);
    let step_tokens = rng.between(max_tokens, max_tokens * 8);
    World {
        settings: Settings {
            namespace: "sim".into(),
            group_size: u32::try_from(rng.between(2, 8)).unwrap(),
            max_staleness,
            rollout_slots: u32::try_from(rng.between(1, 64)).unwrap(),
            tokens_per_step: step_tokens,
            rollout_tokens: max_tokens,
            step_window: usize::try_from(rng.between(1, 8)).unwrap(),
            max_failures: u32::try_from(rng.between(0, 3)).unwrap(),
            send_retry_ms: rng.between(1, 2_000),
            poll_ms: rng.between(1, 500),
            epoch: 0,
            pin_versions: false,
        },
        plan: LearnerPlan {
            step_tokens,
            step_ms: if fast_training {
                rng.between(1, 50)
            } else {
                rng.between(1_000, 60_000)
            },
            publish_every: rng.between(1, 3),
            commit_every: rng.between(1, 5),
            capacity: step_tokens * rng.between(1, 4),
            max_staleness,
        },
        groups: rng.between(20, 200),
        tasks: usize::try_from(rng.between(1, 10)).unwrap(),
        all_equal: rng.below(5) == 0,
        p_fail: [0.0, 0.05, 0.3][rng.below(3)],
        p_unscored: [0.0, 0.1][rng.below(2)],
        p_inexact: [0.0, 0.1][rng.below(2)],
        outage: rng
            .coin()
            .then(|| (rng.between(10_000, 100_000), rng.between(1_000, 20_000))),
        base_ms: if rng.coin() {
            rng.between(1, 20)
        } else {
            rng.between(500, 5_000)
        },
        max_tokens,
    }
}

/// What the simulated runs went through, summed.
#[derive(Debug, Default)]
struct Tally {
    rollouts: usize,
    runs: usize,
    interruptions: usize,
    committed: u64,
    filtered: u64,
    stale: u64,
    failures: u64,
    abandoned: u64,
    learner_full: u64,
    resends: u64,
}

/// One simulated run, interrupted and resumed, added to `tally`.
fn simulate(seed: u64, tally: &mut Tally) {
    let mut rng = SplitMix64::new(seed);
    let w = world(&mut rng);
    let clock = Clock::default();
    let mut learner = ScriptedLearner::new(clock.clone(), w.plan);
    let mut selector =
        RoundRobin::new((0..w.tasks).map(|t| format!("t{t}")).collect(), w.groups).unwrap();
    let mut ledger = MemoryLedger::default();
    let difficulty: Vec<f64> = (0..w.tasks).map(|_| unit(&mut rng)).collect();
    let mut interruptions = rng.below(4);
    tally.runs += 1;
    loop {
        let mut script_rng = SplitMix64::new(rng.next_u64());
        let (clock_s, w_ref, difficulty_ref) = (clock.clone(), w.clone(), difficulty.clone());
        let script = Box::new(
            move |request: &splinter_factory::online::seams::RolloutRequest, _| {
                let now = clock_s.now();
                let u = unit(&mut script_rng).max(1e-9);
                // Pareto tail, shape 1.2.
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    clippy::cast_precision_loss
                )]
                let duration_ms = ((w_ref.base_ms as f64) * u.powf(-1.0 / 1.2)).min(1e7) as u64;
                let in_outage = w_ref.outage.is_some_and(|(period, len)| now % period < len);
                if in_outage || unit(&mut script_rng) < w_ref.p_fail {
                    return Plan {
                        duration_ms,
                        result: Err(if in_outage {
                            "verifier unavailable"
                        } else {
                            "worker lost"
                        }
                        .into()),
                    };
                }
                let task: usize = request.task[1..].parse().unwrap();
                let reward = if w_ref.all_equal {
                    Some(1.0)
                } else if unit(&mut script_rng) < w_ref.p_unscored {
                    None
                } else {
                    Some(f64::from(u8::from(
                        unit(&mut script_rng) < difficulty_ref[task],
                    )))
                };
                let tokens = 1 + script_rng.next_u64() % w_ref.max_tokens;
                Plan {
                    duration_ms,
                    result: Ok((reward, tokens, unit(&mut script_rng) >= w_ref.p_inexact)),
                }
            },
        );
        let mut executor = Executor::new(clock.clone(), learner.served.clone(), script);
        // How this leg ends early, if it does.
        let interrupt = if interruptions > 0 { rng.below(3) } else { 3 };
        let cancel_at = Cell::new(u64::MAX);
        match interrupt {
            0 => executor.die_after = Some(rng.below(200) + 1),
            1 => cancel_at.set(clock.now() + rng.between(1, 200_000)),
            2 => learner.die_after_accepts = Some(learner.accepts + rng.below(20) + 1),
            _ => learner.die_after_accepts = None,
        }
        let (clock_c, changes) = (clock.clone(), ledger.changes.clone());
        let last_change = Cell::new((changes.get(), clock.now()));
        let stalled = Cell::new(false);
        let cancel = || {
            let now = clock_c.now();
            if changes.get() != last_change.get().0 {
                last_change.set((changes.get(), now));
            } else if now - last_change.get().1 >= STALL_MS {
                stalled.set(true);
            }
            now >= cancel_at.get() || stalled.get()
        };
        let slots = w.settings.rollout_slots as usize;
        let result = run(
            w.settings.clone(),
            Seams {
                executor: &mut executor,
                learner: &mut learner,
                selector: &mut selector,
                ledger: &mut ledger,
                clock: &clock,
                cancel: &cancel,
            },
        );
        tally.rollouts += executor
            .ends
            .iter()
            .filter(|(_, e)| e.result.is_ok())
            .count();
        assert!(
            executor.peak_running() <= slots,
            "seed {seed}: more rollouts than slots"
        );
        assert_eq!(
            executor.starts.len(),
            executor.ends.len() + executor.cancelled + executor.running(),
            "seed {seed}: a start neither ended, was cancelled nor was running at a kill"
        );
        assert!(
            !stalled.get(),
            "seed {seed}: no progress (deadlock): {:?}",
            ledger
                .groups
                .values()
                .filter(|e| !e.group.state.is_terminal())
                .map(|e| (&e.group.key, e.group.state))
                .collect::<Vec<_>>()
        );
        match result {
            Ok(report) if report.ended == Ended::Finished => {
                let stats = &report.stats;
                check_finished(seed, &w, &ledger, &learner, stats);
                tally.committed += stats.groups_committed;
                tally.filtered += stats.filtered_zero_variance
                    + stats.filtered_not_exact
                    + stats.filtered_unscored;
                tally.stale += stats.dropped_stale + stats.dropped_by_learner;
                tally.failures += stats.rollout_failures;
                tally.abandoned += stats.rollouts_abandoned;
                tally.learner_full += stats.learner_full;
                tally.resends += stats.resends;
                return;
            }
            Ok(_) => assert_eq!(interrupt, 1, "seed {seed}: cancelled without being asked"),
            Err(OnlineError::Executor(e)) => {
                assert_eq!((interrupt, e.0.as_str()), (0, "killed"), "seed {seed}")
            }
            Err(OnlineError::Learner(e)) => assert_eq!(
                (interrupt, e.message.as_str()),
                (2, "connection lost"),
                "seed {seed}"
            ),
            Err(other) => panic!("seed {seed}: {other}"),
        }
        interruptions = interruptions.saturating_sub(1);
        tally.interruptions += 1;
        if rng.coin() {
            learner.restart();
        }
    }
}

fn check_finished(
    seed: u64,
    w: &World,
    ledger: &MemoryLedger,
    learner: &ScriptedLearner,
    stats: &splinter_factory::online::stats::Stats,
) {
    assert_eq!(ledger.groups.len() as u64, w.groups, "seed {seed}");
    let mut committed = 0;
    for entry in ledger.groups.values() {
        let group = &entry.group;
        assert!(
            group.state.is_terminal(),
            "seed {seed}: {} left {}",
            group.key,
            group.state
        );
        assert!(
            entry.rollouts.iter().all(|r| r.state.is_terminal()),
            "seed {seed}: {} has an unfinished rollout",
            group.key
        );
        for rollout in &entry.rollouts {
            assert_eq!(
                rollout.state == RolloutState::Finished,
                rollout.outcome.is_some()
            );
        }
        if group.state == GroupState::Committed {
            committed += 1;
            assert_eq!(
                learner.commits.get(&group.key),
                Some(&1),
                "seed {seed}: {}",
                group.key
            );
        }
        if w.all_equal {
            assert_ne!(
                group.state,
                GroupState::Committed,
                "seed {seed}: a flat group trained"
            );
        }
    }
    assert!(
        learner.commits.values().all(|n| *n == 1),
        "seed {seed}: a group committed twice"
    );
    assert_eq!(
        learner.commits.len(),
        committed,
        "seed {seed}: the learner committed a group the ledger lost"
    );
    // The last leg's statistics, rebuilt from the ledger on resume, account
    // for every group.
    assert_eq!(
        stats.groups_started,
        stats.groups_committed
            + stats.filtered_zero_variance
            + stats.filtered_not_exact
            + stats.filtered_unscored
            + stats.dropped_stale
            + stats.dropped_by_learner,
        "seed {seed}: {stats:?}"
    );
    assert_eq!(stats.groups_committed, committed as u64, "seed {seed}");
    assert_eq!((stats.in_flight, stats.queue_tokens), (0, 0), "seed {seed}");
    assert!(
        learner.queued_tokens()
            <= w.plan.capacity + u64::from(w.settings.group_size) * w.max_tokens
    );
}

#[test]
fn simulated_runs_finish_bounded_with_every_group_and_rollout_accounted_for() {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap();
    let seed = now.as_secs() ^ u64::from(now.subsec_nanos()).rotate_left(32);
    eprintln!("simulation seed {seed}");
    let mut rng = SplitMix64::new(seed);
    let mut tally = Tally::default();
    while tally.rollouts < EPISODES {
        simulate(rng.next_u64(), &mut tally);
    }
    eprintln!("{tally:?}");
    for (what, n) in [
        ("interruptions", tally.interruptions as u64),
        ("committed groups", tally.committed),
        ("filtered groups", tally.filtered),
        ("stale groups", tally.stale),
        ("failures", tally.failures),
        ("abandoned rollouts", tally.abandoned),
        ("full learner answers", tally.learner_full),
        ("resends", tally.resends),
    ] {
        assert!(n > 0, "the simulation never produced {what}: {tally:?}");
    }
}

/// Runs that once failed, by the seed each printed, kept so they keep
/// passing.
#[test]
fn simulated_runs_that_once_failed_finish() {
    // A restarted learner took a resent applied group back and dropped the
    // connection before answering; the next resend was answered `duplicate`
    // and the group stayed applied here while the learner dropped it as
    // stale: a deadlock.
    simulate(14_490_181_401_853_905_193, &mut Tally::default());
    // Groups recorded sent that the learner had applied, counted as queued
    // after a resume: the allowance held every new group back while the
    // learner waited for a step's worth.
    simulate(13_351_248_293_160_495_685, &mut Tally::default());
}
