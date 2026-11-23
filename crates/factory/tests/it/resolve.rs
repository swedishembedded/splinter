// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! A recipe is read with every file it names, every problem is reported at
//! once, and its graph orders each stage after all it consumes.

use splinter_core::factory::identity::StageKind;
use splinter_factory::resolve::resolve;

use crate::repo::{connections, Repo, RECIPE};

#[test]
fn a_recipe_resolves_into_an_ordered_graph_of_its_stages() {
    let repo = Repo::new();
    let resolved = resolve(&repo.root, RECIPE, connections()).expect("the recipe resolves");
    let ids: Vec<&str> = resolved
        .collection
        .collection
        .tasks
        .iter()
        .map(|t| t.id.as_str())
        .collect();
    assert_eq!(
        ids,
        ["train/001-first", "train/002-second"],
        "only the train split"
    );
    assert_eq!(
        resolved.collection.collection.tasks[0].family.as_deref(),
        Some("course")
    );
    let keys: Vec<&str> = resolved
        .plan
        .stages
        .iter()
        .map(|s| s.key.as_str())
        .collect();
    assert_eq!(
        keys,
        [
            "tasks",
            "rollouts/local",
            "rollouts/remote",
            "datasets/sft",
            "training/lora-r8",
            "training/lora-r16",
            "eval/lora-r8",
            "eval/lora-r16",
            "release"
        ]
    );
    assert!(resolved.plan.is_ordered());
    assert_eq!(
        resolved.plan.stage("rollouts/remote").unwrap().depends_on,
        ["tasks", "rollouts/local"]
    );
    assert_eq!(
        resolved.plan.stage("eval/lora-r8").unwrap().kind,
        StageKind::Eval
    );
    // Changing a variant's training touches only its own downstream.
    let downstream = resolved.plan.downstream("training/lora-r8");
    assert_eq!(
        downstream.into_iter().collect::<Vec<_>>(),
        ["eval/lora-r8", "release"]
    );
    let (plan, tasks) = resolved.eval.as_ref().expect("an eval plan");
    assert!(!plan.protected());
    assert_eq!(tasks.as_ref().unwrap().split(Some("eval")).len(), 1);
}

#[test]
fn every_problem_of_a_recipe_is_reported_at_once() {
    let repo = Repo::new();
    let broken = RECIPE
        .replace("unsolved_by: [local]", "unsolved_by: [later]")
        .replace("sources: [local, remote]", "sources: [local, missing]")
        .replace(
            "candidates: [lora-r8, lora-r16]",
            "candidates: [lora-r8, lora-r99]",
        )
        .replace("\"hosted:deepseek", "\"nowhere:deepseek")
        .replace(
            "presets/sven.yaml]\n    model",
            "presets/gone.yaml]\n    model",
        )
        .replace("SVEN_KEY: \"env:SVEN_KEY\"", "SVEN_KEY: sk-literal");
    let problems = resolve(&repo.root, &broken, connections())
        .expect_err("refused")
        .0;
    let text = problems.join("\n");
    for expected in [
        "selects tasks unsolved by later",
        "reads missing, which is not a rollout pass",
        "candidate lora-r99 is not a training variant",
        "names connection nowhere",
        "presets/gone.yaml",
        "SVEN_KEY looks like a credential",
    ] {
        assert!(
            text.contains(expected),
            "{expected:?} missing from:\n{text}"
        );
    }
}

#[test]
fn an_unknown_key_is_refused_rather_than_ignored() {
    let repo = Repo::new();
    let typo = RECIPE.replace(
        "    attempts: 2\n    seed: 11",
        "    attempts: 2\n    seed: 11\n    attempt: 3",
    );
    let problems = resolve(&repo.root, &typo, connections())
        .expect_err("refused")
        .0;
    assert!(
        problems[0].contains("unknown field `attempt`"),
        "{problems:?}"
    );
}

#[test]
fn an_environment_that_demands_strong_isolation_refuses_proot() {
    let repo = Repo::new();
    repo.write(
        "environments/vim.yaml",
        "schema: splinter.environment/v1\nisolation: strong\nsandbox: { provider: proot }\n",
    );
    let problems = resolve(&repo.root, RECIPE, connections())
        .expect_err("refused")
        .0;
    assert!(
        problems
            .iter()
            .any(|p| p.contains("proot is not a security boundary")),
        "{problems:?}"
    );
}

#[test]
fn the_lock_names_exactly_what_changed_since_a_run_started() {
    let repo = Repo::new();
    let before = resolve(&repo.root, RECIPE, connections()).unwrap().lock;
    assert!(before.changes(&before).is_empty());
    repo.write("presets/sven.yaml", "agent: { max_turns: 40 }\n");
    repo.write(
        "tasks/vim/train/002-second/instruction.md",
        "Do it differently.\n",
    );
    let after = resolve(&repo.root, RECIPE, connections()).unwrap().lock;
    assert_eq!(
        after.changes(&before),
        ["the task collection changed", "presets/sven.yaml changed"]
    );
}

#[test]
fn a_literal_credential_in_connections_is_refused() {
    let error = splinter_factory::connections::Connections::parse(
        "connections:\n  hosted: { protocol: openai, token: sk-123 }\n",
    )
    .expect_err("refused");
    assert!(error.to_string().contains("env:NAME"));
}

#[test]
fn a_collection_names_the_command_that_generates_its_tasks_and_a_missing_root_says_so() {
    let repo = Repo::new();
    repo.write(
        "collections/vim.yaml",
        "schema: splinter.collection/v1\nname: vim\nroot: build/vim\n\
         generate: [cp, -r, tasks/vim, build/vim]\n",
    );
    let problems = resolve(&repo.root, RECIPE, connections())
        .expect_err("no tasks yet")
        .0;
    assert!(
        problems
            .iter()
            .any(|p| p.contains("splinter generate collections/vim.yaml")),
        "{problems:?}"
    );
    std::fs::create_dir_all(repo.root.join("build")).unwrap();
    splinter_factory::collection::generate(&repo.root, "collections/vim.yaml").unwrap();
    assert!(resolve(&repo.root, RECIPE, connections()).is_ok());
}

#[test]
fn examination_tasks_never_reach_a_dataset_under_any_name() {
    let repo = Repo::new();
    // No split: the collection's held-out task would be attempted for training.
    let unsplit = RECIPE.replace(", split: train }", " }");
    let problems = resolve(&repo.root, &unsplit, connections())
        .expect_err("refused")
        .0;
    assert!(
        problems
            .iter()
            .any(|p| p.contains("eval/001-held") && p.contains("examination")),
        "{problems:?}"
    );
    // The held-out task copied under a training name is the same task.
    for file in ["task.toml", "instruction.md", "tests/test.sh"] {
        let text =
            std::fs::read_to_string(repo.root.join("tasks/vim/eval/001-held").join(file)).unwrap();
        repo.write(&format!("tasks/vim/train/003-copy/{file}"), &text);
    }
    let problems = resolve(&repo.root, RECIPE, connections())
        .expect_err("refused")
        .0;
    assert!(
        problems
            .iter()
            .any(|p| p.contains("train/003-copy") && p.contains("eval/001-held")),
        "{problems:?}"
    );
    // A recipe with no dataset trains on nothing, so it may attempt them.
    let no_dataset = unsplit.split("dataset:").next().unwrap().to_owned();
    std::fs::remove_dir_all(repo.root.join("tasks/vim/train/003-copy")).unwrap();
    resolve(&repo.root, &no_dataset, connections()).expect("a recipe that trains on nothing");
}

#[test]
fn a_connection_drops_only_headers_the_gateway_forwards() {
    use splinter_factory::connections::Connections;
    let connections = Connections::parse(
        "connections:\n  hosted: { protocol: anthropic, url: \"https://example.invalid/v1\", \
         token: \"env:KEY\", drop_headers: [anthropic-beta] }\n",
    )
    .unwrap();
    let binding = connections.bind(&"hosted:m".parse().unwrap()).unwrap();
    assert_eq!(binding.drop_headers, ["anthropic-beta"]);
    let refused = Connections::parse(
        "connections:\n  hosted: { protocol: anthropic, drop_headers: [authorization] }\n",
    )
    .unwrap_err()
    .to_string();
    assert!(
        refused.contains("authorization") && refused.contains("anthropic-beta"),
        "{refused}"
    );
}
