// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for spending a reinforcement-learning
// budget only on tasks that teach, for its clients. If your team needs
// expertise in curricula for policy-gradient training you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: the frontier selector, driven by a whole run, spends fewer groups
//! on tasks whose rewards never differ than the tasks in turn do, retires a
//! task the policy always passes, and is rebuilt from the ledger alone when
//! the controller resumes.

use splinter_factory::online::driver::{run, RunLedger, Seams};
use splinter_factory::online::frontier::{Selector, SelectorKind};
use splinter_factory::online::Settings;
use splinter_store::factory::runstate::RunState;

use super::doubles::{Clock, Executor, Plan, ScriptedLearner};
use super::{learner_plan, settings};

/// Rollouts of `always` pass, of `never` fail, of `sometimes` pass on even
/// indices.
fn script() -> super::doubles::Script {
    Box::new(|request, _| Plan {
        duration_ms: 50,
        result: Ok((
            Some(match request.task.as_str() {
                "always" => 1.0,
                "never" => 0.0,
                _ => f64::from(u8::from(request.index % 2 == 0)),
            }),
            40,
            true,
        )),
    })
}

fn tasks() -> Vec<String> {
    ["always", "never", "sometimes"].map(String::from).to_vec()
}

/// The share of scored groups whose rewards did not differ, of a run of 60
/// groups drawn by `kind`, and the state it left.
fn zero_variance_share(kind: SelectorKind) -> (f64, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let state = RunState::open(dir.path()).unwrap();
    let fence = state.acquire("controller", 30, 0).unwrap();
    let clock = Clock::default();
    let mut learner = ScriptedLearner::new(clock.clone(), learner_plan());
    let mut executor = Executor::new(clock.clone(), learner.served.clone(), script());
    // seed: the frontier's draws derive from it; the share is compared, not the draws.
    let mut selector = Selector::new(kind, tasks(), 60, 11).unwrap();
    let mut ledger = RunLedger::new(&state, fence, 30, 0);
    let report = run(
        Settings {
            max_staleness: 100,
            group_size: 4,
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
    (report.stats.frac_zero_std().unwrap(), dir)
}

#[test]
fn the_frontier_spends_fewer_groups_where_rewards_never_differ_than_the_tasks_in_turn() {
    let (turns, _) = zero_variance_share(SelectorKind::RoundRobin);
    let (frontier, dir) = zero_variance_share(SelectorKind::Frontier);
    assert!((turns - 2.0 / 3.0).abs() < 0.02, "{turns}");
    assert!(
        frontier < turns / 2.0,
        "the frontier draws where groups teach: {frontier} against {turns}"
    );
    // Resumed from the ledger alone, the selector knows what the run knew.
    let state = RunState::open(dir.path()).unwrap();
    let entries = state.group_entries().unwrap();
    // seed: as above.
    let mut resumed = Selector::new(SelectorKind::Frontier, tasks(), 60, 11).unwrap();
    let (_, _) = splinter_factory::online::control::Controller::restore(
        Settings {
            max_staleness: 100,
            group_size: 4,
            ..settings()
        },
        entries,
        &splinter_factory::online::seams::LearnerState::default(),
        &mut resumed,
    )
    .unwrap();
    let report = resumed.report();
    assert_eq!(report["tasks"]["always"]["retired"], true, "{report:#}");
    assert_eq!(report["tasks"]["sometimes"]["retired"], false, "{report:#}");
}
