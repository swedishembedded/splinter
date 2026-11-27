// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! What a changed recipe reuses: another curation of the same trials runs no
//! rollout again, another eval plan trains no candidate again, a fork of
//! several runs reuses what each of them committed, and a run copied to
//! another place verifies with the original gone.

use crate::factory_run::{decisive, resolved, run};
use crate::fakes::{FakeBackend, FakeTrainer, Outcome};
use crate::repo::{Repo, RECIPE};
use splinter_factory::run::{fork, fork_from, verify, ForkOrigin, FORK_FILE};

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
            .any(|k| ["tasks", "local", "remote"].contains(&k.as_str())),
        "{stages:?}"
    );
    assert!(stages.contains(&"sft".to_owned()), "{stages:?}");
    assert_eq!(
        backend.executed()[executed..]
            .iter()
            .filter(
                |s| !s.slot.pass.starts_with("baseline-") && !s.slot.pass.ends_with(".candidate")
            )
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
    assert!(!stages.iter().any(|k| k.starts_with("lora-")), "{stages:?}");
    assert!(stages.contains(&"eval-lora-r8".to_owned()), "{stages:?}");
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

/// A recipe of a task pin and an attempt stage `id` per `(id, seed)`, each
/// on the pinned tasks alone, and a dataset over them when `dataset`.
fn attempts_recipe(stages: &[(&str, u64)], dataset: bool) -> String {
    let head = RECIPE.split("  - id: local\n").next().unwrap();
    let mut out = head.to_owned();
    for (id, seed) in stages {
        out.push_str(&format!(
            "  - id: {id}\n    uses: attempt.collect\n    inputs: {{ tasks: $stages.tasks.tasks }}\n    \
             with: {{ agent: sven, model: student, environment: environments/vim.yaml, attempts: 2, seed: {seed} }}\n"
        ));
    }
    if dataset {
        let attempts: Vec<String> = stages
            .iter()
            .map(|(id, _)| format!("$stages.{id}.attempts"))
            .collect();
        // seed: a recipe states every seed; the fake backend and trainer draw nothing from them.
        out.push_str(&format!(
            "  - id: sft\n    uses: dataset.project\n    inputs: {{ attempts: [{}] }}\n    \
             with: {{ target: student, seed: 13 }}\n",
            attempts.join(", ")
        ));
    }
    out
}

/// The run each stage of `run`'s fork came from, by stage key.
fn sources_of(run: &std::path::Path) -> std::collections::BTreeMap<String, std::path::PathBuf> {
    let origin: ForkOrigin =
        serde_json::from_slice(&std::fs::read(run.join(FORK_FILE)).unwrap()).unwrap();
    origin
        .products
        .into_iter()
        .map(|p| (p.name, p.source))
        .collect()
}

#[test]
fn a_fork_of_several_runs_reuses_what_each_committed_first_match_first_and_computes_the_rest() {
    let repo = Repo::new();
    let backend = FakeBackend::with(Outcome::Pass);
    let trainer = FakeTrainer::default();
    // Two runs that each computed a different stage over the pin they share
    // (both forks of the run that pinned the tasks).
    let base = tempfile::tempdir().unwrap();
    run(
        &resolved(&repo, &attempts_recipe(&[], false)),
        base.path(),
        &backend,
        &trainer,
    )
    .unwrap();
    let lineage = tempfile::tempdir().unwrap();
    fork(base.path(), lineage.path()).unwrap();
    run(
        &resolved(&repo, &attempts_recipe(&[("local", 11)], false)),
        lineage.path(),
        &backend,
        &trainer,
    )
    .unwrap();
    let material = tempfile::tempdir().unwrap();
    fork(base.path(), material.path()).unwrap();
    run(
        &resolved(&repo, &attempts_recipe(&[("material", 21)], false)),
        material.path(),
        &backend,
        &trainer,
    )
    .unwrap();
    let executed = backend.executed().len();

    // The next generation uses both: only the dataset over them is new.
    let next = tempfile::tempdir().unwrap();
    let forked = fork_from(&[lineage.path(), material.path()], next.path()).unwrap();
    assert_eq!(forked.per_source, [2, 1], "the pin is taken from the first");
    assert!(forked.conflicts.is_empty(), "{:?}", forked.conflicts);
    let recipe = attempts_recipe(&[("local", 11), ("material", 21)], true);
    let stages = run(&resolved(&repo, &recipe), next.path(), &backend, &trainer).unwrap();
    let computed: Vec<&str> = stages
        .iter()
        .filter(|(_, c)| *c)
        .map(|(k, _)| k.as_str())
        .collect();
    assert_eq!(computed, ["sft"], "{stages:?}");
    assert_eq!(backend.executed().len(), executed, "no attempt ran again");
    let sources = sources_of(next.path());
    let lineage_dir = std::fs::canonicalize(lineage.path()).unwrap();
    let material_dir = std::fs::canonicalize(material.path()).unwrap();
    assert_eq!(
        sources["tasks"], lineage_dir,
        "first match in the order given"
    );
    assert_eq!(sources["local"], lineage_dir);
    assert_eq!(sources["material"], material_dir);
    assert!(verify(next.path()).unwrap().is_empty());

    // The other order takes the shared pin from the material run.
    let reversed = tempfile::tempdir().unwrap();
    fork_from(&[material.path(), lineage.path()], reversed.path()).unwrap();
    assert_eq!(sources_of(reversed.path())["tasks"], material_dir);

    // A stage whose identity is in neither run is computed.
    let changed = tempfile::tempdir().unwrap();
    fork_from(&[lineage.path(), material.path()], changed.path()).unwrap();
    let recipe = attempts_recipe(&[("local", 11), ("material", 22)], false);
    let stages = run(
        &resolved(&repo, &recipe),
        changed.path(),
        &backend,
        &trainer,
    )
    .unwrap();
    let computed: Vec<&str> = stages
        .iter()
        .filter(|(_, c)| *c)
        .map(|(k, _)| k.as_str())
        .collect();
    assert_eq!(computed, ["material"], "{stages:?}");
    assert!(backend.executed().len() > executed);

    // A run that pinned the tasks again holds another product of the pin,
    // so nothing downstream of it can be reused: the fork says so.
    let alone = tempfile::tempdir().unwrap();
    run(
        &resolved(&repo, &attempts_recipe(&[("material", 21)], false)),
        alone.path(),
        &backend,
        &trainer,
    )
    .unwrap();
    let mixed = tempfile::tempdir().unwrap();
    let forked = fork_from(&[lineage.path(), alone.path()], mixed.path()).unwrap();
    let alone_dir = std::fs::canonicalize(alone.path()).unwrap();
    assert_eq!(forked.conflicts, [("tasks".to_owned(), alone_dir)]);
}
