// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for running agent experiments that survive crashes and restarts for its
// clients. If your team needs expertise in resumable evaluation and training pipelines then you can procure our
// services by sending an email to info@swedishembedded.com.

//! A run stopped while a batch of attempts is in flight, and run again.

use splinter_core::factory::product::TrialResult;
use splinter_factory::run::rollout::trials_of;
use splinter_factory::run::RunError;

use crate::factory_run::{resolved, run};
use crate::fakes::{FakeBackend, FakeTrainer, Outcome};
use crate::repo::{Repo, DATASET, RECIPE};

#[test]
fn a_controller_killed_mid_rollout_resumes_without_repeating_a_finished_attempt() {
    let repo = Repo::new();
    let resolved = resolved(&repo, RECIPE.split(DATASET).next().unwrap());
    let backend = FakeBackend::with(Outcome::Pass);
    *backend.crash_after.lock().unwrap() = Some(2);
    let trainer = FakeTrainer::default();
    let dir = tempfile::tempdir().unwrap();
    let crashed = run(&resolved, dir.path(), &backend, &trainer);
    assert!(matches!(crashed, Err(RunError::Backend(_))), "{crashed:?}");
    assert_eq!(backend.executed().len(), 2);

    run(&resolved, dir.path(), &backend, &trainer).unwrap();
    let local: Vec<_> = backend
        .executed()
        .into_iter()
        .filter(|s| s.slot.pass == "local")
        .collect();
    let mut slots: Vec<_> = local
        .iter()
        .map(|s| (s.task.id.clone(), s.slot.slot))
        .collect();
    slots.sort();
    slots.dedup();
    assert_eq!(slots.len(), 4);
    assert_eq!(
        local.len(),
        4,
        "the two attempts that reported before the kill were reconciled, not run again"
    );
    assert_eq!(trials_of(&dir.path().join("local")).unwrap().len(), 4);
}

#[test]
fn attempts_lost_because_the_run_was_interrupted_do_not_spend_a_slots_retries() {
    // A run stopped before its batch reported (a killed controller, a node out of memory, an operator restart)
    // loses every attempt in flight. That says nothing about the tasks: the slots are run again however often it
    // happened, and only losses of a batch the worker ran to its end count against `retries`.
    let repo = Repo::new();
    let resolved = resolved(&repo, RECIPE.split(DATASET).next().unwrap());
    let backend = FakeBackend::with(Outcome::Pass);
    let trainer = FakeTrainer::default();
    let dir = tempfile::tempdir().unwrap();
    for _ in 0..4 {
        *backend.crash_after.lock().unwrap() = Some(0);
        let interrupted = run(&resolved, dir.path(), &backend, &trainer);
        assert!(
            matches!(interrupted, Err(RunError::Backend(_))),
            "{interrupted:?}"
        );
    }
    run(&resolved, dir.path(), &backend, &trainer).unwrap();
    let trials = trials_of(&dir.path().join("local")).unwrap();
    assert_eq!(trials.len(), 4);
    assert!(trials.iter().all(TrialResult::verified_success));
}
