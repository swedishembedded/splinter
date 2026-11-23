// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training environments a campaign can learn
// from, admitted only once a base model's measured pass rate shows there is
// something to learn, for its clients. If your team needs expertise in
// verifiable RL environments, you can procure our services by sending an
// email to info@swedishembedded.com.

//! `environment.qualify`: environments built from an objective enter a
//! campaign only once a base model's attempts at their dev instances put
//! its pass rate inside the band: one it always solves, or never does,
//! gives a group-relative trainer nothing to learn from.

use serde_json::Value;
use splinter_factory::resolve::resolve;
use splinter_factory::run::tasks::tasks_of;
use splinter_factory::run::{Factory, Services};

use crate::fakes::{FakeBackend, Outcome};
use crate::repo::{connections, Repo};

// seed: the recipe states every seed; the fake backend's outcomes are scripted.
const BUILT: &str = "schema: splinter.recipe/v2
name: qualified
models:
  base: \"brain:Qwen/Qwen3-4B\"
stages:
  - id: built
    uses: environment.build
    with: { objective: objectives/python.md, environments: 1, instances: 2, seed: 7 }
  - { id: dev, uses: tasks.pin, inputs: { collection: $stages.built.environments }, with: { split: dev } }
";

// seed: a recipe states every seed; the fake backend's outcomes are scripted, so none draws.
const MEASURED: &str = "  - id: base
    uses: attempt.collect
    inputs: { tasks: $stages.dev.tasks }
    with: { agent: sven, model: base, environment: environments/vim.yaml, attempts: 8, seed: 11 }
  - id: qualified
    uses: environment.qualify
    inputs: { environments: $stages.built.environments, attempts: $stages.base.attempts }
  - { id: train, uses: tasks.pin, inputs: { collection: $stages.qualified.environments }, with: { split: train } }
";

fn run(
    repo: &Repo,
    recipe: &str,
    backend: &FakeBackend,
    dir: &std::path::Path,
) -> Result<(), String> {
    let resolved = resolve(&repo.root, recipe, connections()).map_err(|e| e.to_string())?;
    let services = Services {
        backend,
        trainer: None,
        models: None,
        variables: None,
    };
    let mut factory = Factory::open(&resolved, dir, services, "test").map_err(|e| e.to_string())?;
    factory.run().map(|_| ()).map_err(|e| e.to_string())
}

/// Builds the environment and has the base model attempt each dev instance
/// eight times, passing the first `passes` of each; what qualification
/// made of it.
fn qualified(passes: usize) -> (tempfile::TempDir, Result<(), String>) {
    let repo = Repo::new();
    repo.write(
        "objectives/python.md",
        "Predict the output of Python snippets\n",
    );
    let dir = tempfile::tempdir().unwrap();
    let backend = FakeBackend::with(Outcome::Fail);
    run(&repo, BUILT, &backend, dir.path()).unwrap();
    let dev = tasks_of(&dir.path().join("dev")).unwrap();
    assert_eq!(dev.tasks.len(), 2, "{:?}", dev.tasks);
    for task in &dev.tasks {
        let outcomes: Vec<Outcome> = (0..8)
            .map(|i| {
                if i < passes {
                    Outcome::Pass
                } else {
                    Outcome::Fail
                }
            })
            .collect();
        backend.script(&task.id, &outcomes);
    }
    // The ids of the dev instances are known once they are built; the
    // measured run is a fork of the built one, so it reuses what was built.
    let measured = tempfile::tempdir().unwrap();
    splinter_factory::run::fork(dir.path(), measured.path()).unwrap();
    let outcome = run(
        &repo,
        &(BUILT.to_owned() + MEASURED),
        &backend,
        measured.path(),
    );
    (measured, outcome)
}

#[test]
fn an_environment_a_base_model_solves_half_the_time_is_admitted_to_training() {
    let (dir, outcome) = qualified(4);
    outcome.unwrap();
    let train = tasks_of(&dir.path().join("train")).unwrap();
    assert_eq!(train.tasks.len(), 2, "{:?}", train.tasks);
    let reports: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("qualified/qualify.json")).unwrap(),
    )
    .unwrap();
    let report = &reports.as_array().unwrap()[0];
    assert_eq!(report["status"], "admitted", "{report:#}");
    assert_eq!(report["band"]["attempts"], 16, "{report:#}");
}

#[test]
fn an_environment_a_base_model_always_solves_is_kept_out_of_training() {
    let (_, outcome) = qualified(8);
    let why = outcome.unwrap_err();
    assert!(why.contains("base-band"), "{why}");
}

/// Scenario B: a document to executable RLVR tasks. The objective document
/// is compiled into environments whose instances are Harbor tasks with
/// their own verifiers, the base model attempts the training instances, and
/// those it solves only sometimes are the collection an online trainer
/// rolls out again.
#[test]
fn scenario_b_a_document_becomes_executable_rlvr_tasks() {
    let repo = Repo::new();
    repo.write(
        "objectives/python.md",
        "Predict the output of Python snippets\n",
    );
    // seed: the recipe states every seed; the fake backend's outcomes are scripted.
    let built = "schema: splinter.recipe/v2
name: rlvr
models:
  base: \"brain:Qwen/Qwen3-4B\"
stages:
  - id: built
    uses: environment.build
    with: { objective: objectives/python.md, environments: 1, instances: 2, seed: 7 }
  - { id: train, uses: tasks.pin, inputs: { collection: $stages.built.environments }, with: { split: train } }
";
    let backend = FakeBackend::with(Outcome::Fail);
    let first = tempfile::tempdir().unwrap();
    run(&repo, built, &backend, first.path()).unwrap();
    let tasks = tasks_of(&first.path().join("train")).unwrap();
    assert_eq!(tasks.tasks.len(), 2);
    // One instance solved sometimes, the other always.
    backend.script(&tasks.tasks[0].id, &[Outcome::Pass, Outcome::Fail]);
    backend.script(&tasks.tasks[1].id, &[Outcome::Pass, Outcome::Pass]);
    let rlvr = built.to_owned()
        + "  - id: base
    uses: attempt.collect
    inputs: { tasks: $stages.train.tasks }
    with: { agent: sven, model: base, environment: environments/vim.yaml, attempts: 2, seed: 11 }
  - id: rlvr
    uses: dataset.rlvr
    inputs: { tasks: $stages.train.tasks, attempts: [$stages.base.attempts] }
  - { id: online, uses: tasks.pin, inputs: { collection: $stages.rlvr.tasks }, with: { split: train } }
";
    let second = tempfile::tempdir().unwrap();
    splinter_factory::run::fork(first.path(), second.path()).unwrap();
    run(&repo, &rlvr, &backend, second.path()).unwrap();
    let online = tasks_of(&second.path().join("online")).unwrap();
    assert_eq!(
        online
            .tasks
            .iter()
            .map(|t| t.id.as_str())
            .collect::<Vec<_>>(),
        [tasks.tasks[0].id.as_str()]
    );
    assert!(second
        .path()
        .join("online")
        .join(&tasks.tasks[0].id)
        .join("task.toml")
        .is_file());
}
