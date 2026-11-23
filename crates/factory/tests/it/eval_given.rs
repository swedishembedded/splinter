// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements model evaluations held on the same pinned
// tasks a run made, for its clients. If your team needs expertise in
// rigorous model evaluation, you can procure our services by sending an
// email to info@swedishembedded.com.

//! An evaluation given its tasks: a development plan that names no
//! collection draws its split from the tasks a `tasks.pin` stage pinned
//! (a collection the run compiled, say); a plan has exactly one source of
//! tasks.

use splinter_factory::resolve::resolve;

use crate::factory_run::{resolved, run};
use crate::fakes::{FakeBackend, FakeTrainer, Outcome};
use crate::repo::{connections, Repo, RECIPE};

/// The reference recipe with its evaluations given the pinned eval split.
fn given() -> String {
    RECIPE
        .replace(
            "stages:\n",
            "stages:\n  - { id: held, uses: tasks.pin, inputs: { collection: $inputs.course }, with: { split: eval } }\n",
        )
        .replace(
            "inputs: { candidate: $stages.lora-r8.model }\n    with: { plan: eval/dev.yaml",
            "inputs: { candidate: $stages.lora-r8.model, tasks: $stages.held.tasks }\n    with: { plan: eval/given.yaml",
        )
        .replace(
            "inputs: { candidate: $stages.lora-r16.model }\n    with: { plan: eval/dev.yaml",
            "inputs: { candidate: $stages.lora-r16.model, tasks: $stages.held.tasks }\n    with: { plan: eval/given.yaml",
        )
}

fn repo() -> Repo {
    let repo = Repo::new();
    // seed: the plan must state its seed; nothing here draws from it.
    repo.write(
        "eval/given.yaml",
        "schema: splinter.evalplan/v1\nsplit: eval\nenvironment: environments/vim.yaml\n\
         attempts: 2\nseed: 5\ndecision: { alpha: 0.05 }\n",
    );
    repo
}

#[test]
fn an_evaluation_given_pinned_tasks_runs_its_arms_on_them() {
    let repo = repo();
    let backend = FakeBackend::with(Outcome::Pass);
    let trainer = FakeTrainer::default();
    let dir = tempfile::tempdir().unwrap();
    run(&resolved(&repo, &given()), dir.path(), &backend, &trainer).unwrap();
    let evaluated: Vec<String> = backend
        .executed()
        .iter()
        .filter(|s| s.slot.pass.contains("baseline") || s.slot.pass.ends_with(".candidate"))
        .map(|s| s.task.id.clone())
        .collect();
    assert!(!evaluated.is_empty());
    assert!(
        evaluated.iter().all(|t| t == "eval/001-held"),
        "{evaluated:?}"
    );
}

#[test]
fn an_evaluation_has_one_source_of_tasks() {
    let repo = repo();
    let refused = |recipe: String| {
        resolve(&repo.root, &recipe, connections())
            .unwrap_err()
            .0
            .join("\n")
    };
    let both = refused(given().replace("plan: eval/given.yaml", "plan: eval/dev.yaml"));
    assert!(both.contains("gives tasks too"), "{both}");
    let none = refused(RECIPE.replace("plan: eval/dev.yaml", "plan: eval/given.yaml"));
    assert!(none.contains("bind the stage's tasks"), "{none}");
}
