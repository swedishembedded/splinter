// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements programmable agent workflows that are
// measured on the same tasks as the plain agent, for its clients. If your
// team needs expertise in agent harnesses and their evaluation, you can
// procure our services by sending an email to info@swedishembedded.com.

//! `harness:` on an attempt stage: every attempt runs the named sven harness
//! in place of the agent's own conversation, given the task's instruction.
//! The harness is pinned like any file the agent is handed, so a changed
//! harness is a changed stage and nothing else; one that cannot serve an
//! attempt is refused before anything runs.

use splinter_factory::resolve::resolve;

use crate::factory_run::{resolved, run};
use crate::fakes::{FakeBackend, FakeTrainer, Outcome};
use crate::repo::{connections, dataset, Repo, DATASET, RECIPE};

const HARNESS: &str = r#"api_version: "2"
name: attempt
inputs:
  task: { type: string, description: "The task's instruction." }
agent:
  profile: coding
execution:
  limits: { model_turns: 20 }
workflow:
  - name: work
    task: "{{ inputs.task }}"
"#;

const LOCAL: &str = "      model: student\n      environment: environments/vim.yaml\n";

/// The recipe up to a portable dataset, its `local` stage run by `harness`.
fn with_harness(harness: &str) -> String {
    RECIPE.split(DATASET).next().unwrap().replace(
        LOCAL,
        &format!("      model: student\n      harness: {harness}\n      environment: environments/vim.yaml\n"),
    ) + &dataset(&["local", "remote"], ", projection: portable-sft")
}

fn repo() -> Repo {
    let repo = Repo::new();
    repo.write("harnesses/attempt.yaml", HARNESS);
    repo
}

#[test]
fn a_harness_stage_hands_every_attempt_the_pinned_harness() {
    let repo = repo();
    let backend = FakeBackend::with(Outcome::Pass);
    let trainer = FakeTrainer::default();
    let dir = tempfile::tempdir().unwrap();
    let recipe = with_harness("harnesses/attempt.yaml");
    run(&resolved(&repo, &recipe), dir.path(), &backend, &trainer).unwrap();

    let executed = backend.executed();
    assert!(!executed.is_empty());
    for spec in &executed {
        assert_eq!(
            spec.agent.harness.as_ref().unwrap().path,
            "harnesses/attempt.yaml"
        );
        assert_eq!(spec.agent.name, "sven");
    }

    // A changed harness is a changed stage: only it and what it feeds run again.
    let before = executed.len();
    repo.write(
        "harnesses/attempt.yaml",
        &HARNESS.replace("model_turns: 20", "model_turns: 30"),
    );
    let manifest = run(&resolved(&repo, &recipe), dir.path(), &backend, &trainer);
    // The run was started on the first harness: continuing it is refused,
    // forking it reuses what the change leaves alone.
    assert!(manifest.is_err());
    let fork = tempfile::tempdir().unwrap();
    splinter_factory::run::fork(dir.path(), fork.path()).unwrap();
    let stages = run(&resolved(&repo, &recipe), fork.path(), &backend, &trainer).unwrap();
    let computed: Vec<&str> = stages
        .iter()
        .filter(|(_, computed)| *computed)
        .map(|(key, _)| key.as_str())
        .collect();
    assert!(
        computed.contains(&"local") && !computed.contains(&"tasks"),
        "{stages:?}"
    );
    assert!(backend.executed().len() > before);
}

#[test]
fn a_harness_that_cannot_serve_an_attempt_is_refused_before_anything_runs() {
    let repo = repo();
    repo.write(
        "harnesses/no-task.yaml",
        &HARNESS
            .replace("  task: {", "  feature: {")
            .replace("inputs.task", "inputs.feature"),
    );
    let refused = |recipe: String| {
        resolve(&repo.root, &recipe, connections())
            .unwrap_err()
            .0
            .join("\n")
    };
    let no_task = refused(with_harness("harnesses/no-task.yaml"));
    assert!(no_task.contains("`task` input"), "{no_task}");
    let missing = refused(with_harness("harnesses/none.yaml"));
    assert!(missing.contains("harnesses/none.yaml"), "{missing}");
    let other_agent = refused(with_harness("harnesses/attempt.yaml").replacen(
        "      agent: sven\n      args",
        "      agent: claude-code\n      args",
        1,
    ));
    assert!(
        other_agent.contains("harness") && other_agent.contains("sven"),
        "{other_agent}"
    );
    let native = refused(
        with_harness("harnesses/attempt.yaml")
            .replace(", projection: portable-sft", ", projection: native-sft"),
    );
    assert!(native.contains("portable-sft"), "{native}");
}
