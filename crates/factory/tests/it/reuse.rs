// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! What a changed recipe reuses: another curation of the same trials runs no
//! rollout again, another eval plan trains no candidate again, and a run
//! copied to another place verifies with the original gone.

use crate::factory_run::{decisive, resolved, run};
use crate::fakes::{FakeBackend, FakeTrainer, Outcome};
use crate::repo::{Repo, RECIPE};
use splinter_factory::run::{fork, verify};

#[test]
fn a_changed_dataset_or_eval_plan_reuses_what_is_upstream_and_a_copied_run_verifies_alone() {
    let repo = Repo::new();
    decisive(&repo);
    let backend = FakeBackend {
        candidate: Some(Outcome::Pass),
        ..FakeBackend::with(Outcome::Pass)
    };
    let trainer = FakeTrainer::default();
    let first = tempfile::tempdir().unwrap();
    run(&resolved(&repo, RECIPE), first.path(), &backend, &trainer).unwrap();
    let computed = |stages: Vec<(String, bool)>| -> Vec<String> {
        stages
            .into_iter()
            .filter(|(_, c)| *c)
            .map(|(k, _)| k)
            .collect()
    };

    // Another curation of the same trials: no rollout runs again.
    let executed = backend.executed().len();
    let curated = RECIPE.replace("seed: 13 }", "thinking: true, seed: 13 }");
    let second = tempfile::tempdir().unwrap();
    fork(first.path(), second.path()).unwrap();
    let stages = computed(
        run(
            &resolved(&repo, &curated),
            second.path(),
            &backend,
            &trainer,
        )
        .unwrap(),
    );
    assert!(
        !stages
            .iter()
            .any(|k| k.starts_with("rollouts/") || k == "tasks"),
        "{stages:?}"
    );
    assert!(stages.contains(&"datasets/sft".to_owned()), "{stages:?}");
    assert_eq!(
        backend.executed()[executed..]
            .iter()
            .filter(|s| !s.slot.pass.starts_with("eval-"))
            .count(),
        0
    );

    // Another eval plan: the candidates are reused, the evaluation is new.
    let trained = trainer.trained.lock().unwrap().len();
    // seed: the plan must state its seed; the fake backend draws nothing from it.
    repo.write(
        "eval/other.yaml",
        "schema: splinter.evalplan/v1\ncollection: collections/vim.yaml\nsplit: eval\n\
         environment: environments/vim.yaml\nattempts: 8\nseed: 6\ndecision: { alpha: 0.01 }\n",
    );
    let replanned = RECIPE.replace("plan: eval/dev.yaml", "plan: eval/other.yaml");
    let third = tempfile::tempdir().unwrap();
    fork(first.path(), third.path()).unwrap();
    let stages = computed(
        run(
            &resolved(&repo, &replanned),
            third.path(),
            &backend,
            &trainer,
        )
        .unwrap(),
    );
    assert!(
        !stages.iter().any(|k| k.starts_with("training/")),
        "{stages:?}"
    );
    assert!(stages.contains(&"eval/lora-r8".to_owned()), "{stages:?}");
    assert_eq!(trainer.trained.lock().unwrap().len(), trained);

    // The run copied elsewhere, its original gone, verifies on its own.
    let copy = tempfile::tempdir().unwrap();
    let status = std::process::Command::new("cp")
        .arg("-a")
        .arg(format!("{}/.", third.path().display()))
        .arg(copy.path())
        .status()
        .unwrap();
    assert!(status.success());
    drop(third);
    let problems = verify(copy.path()).unwrap();
    assert!(problems.is_empty(), "{problems:?}");
}
