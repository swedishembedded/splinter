// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for configuration and recipe
// parsers that refuse a damaged file by name instead of crashing, for its
// clients. If your team needs expertise in property-based testing of
// orchestration tools then you can procure our services by sending an email
// to info@swedishembedded.com.

//! Spec, by property: every file the factory reads from a user or a task
//! repository (a recipe, an environment, an evaluation plan, the machine
//! configuration, an operator's grants, a repository's connections, a task
//! collection and its tasks' `task.toml`) is answered, a value or an error,
//! however it is damaged: truncated, a byte changed, a span lost or
//! repeated. None panics. Each starts from a document the parser accepts,
//! so the damage reaches its checks and not only its first byte; a failure
//! is shrunk and saved in `proptest-regressions/`.
#![allow(clippy::unwrap_used)]

use proptest::prelude::*;
use splinter_conformance::fuzz::{config, damaged_text};
use splinter_factory::connections::Connections;
use splinter_factory::environment::Environment;
use splinter_factory::eval_plan::EvalPlan;
use splinter_factory::grants::Grants;
use splinter_factory::machine_config::MachineConfig;
use splinter_factory::{collection, graph};

const RECIPES: [&str; 5] = [
    include_str!("../../templates/agent-sft/recipes/agent-sft.yaml"),
    include_str!("../../templates/generations/recipes/generation.yaml"),
    include_str!("../../templates/harness-search/recipes/harness-search.yaml"),
    include_str!("../../templates/knowledge-learning/recipes/knowledge-learning.yaml"),
    include_str!("../../templates/repository-practice/recipes/repository-practice.yaml"),
];
const ENVIRONMENT: &str = include_str!("../../templates/common/environments/default.yaml");
const EVAL_PLAN: &str = include_str!("../../templates/common/eval/held.yaml");
const COLLECTION: &str = include_str!("../../templates/common/collections/tasks.yaml");
const TASK: &str = include_str!("../../templates/common/tasks/train/001-hello/task.toml");

const MACHINE: &str = "schema: splinter.config/v1
connections:
  brain:
    protocol: openai
    url: http://127.0.0.1:8990/v1
    token: env:SVEN_KEY
operators:
  examiner:
    read: [/srv/exams]
    write: [/var/lib/examiner]
    network: false
    env: [EXAM_KEY]
";
const GRANTS: &str = "schema: splinter.operator-grants/v1
grants:
  examiner:
    read: [/srv/exams]
    write: [/var/lib/examiner]
    env: [EXAM_KEY]
";
const CONNECTIONS: &str = "connections:
  brain:
    protocol: openai
    url: http://127.0.0.1:8990/v1
    token: env:SVEN_KEY
";

#[test]
fn every_seed_is_accepted_as_written() {
    for recipe in RECIPES {
        graph::parse(recipe).unwrap();
    }
    Environment::parse(ENVIRONMENT).unwrap();
    EvalPlan::parse(EVAL_PLAN).unwrap();
    MachineConfig::parse(MACHINE).unwrap();
    Grants::parse(GRANTS).unwrap();
    Connections::parse(CONNECTIONS).unwrap();
    collection_of(COLLECTION, TASK).unwrap();
}

/// The collection `text` names, holding one task whose `task.toml` is `task`.
fn collection_of(text: &str, task: &str) -> Result<(), String> {
    let repository = tempfile::tempdir().unwrap();
    let tasks = repository.path().join("tasks/train/001-hello");
    std::fs::create_dir_all(&tasks).unwrap();
    std::fs::write(tasks.join("task.toml"), task).unwrap();
    std::fs::write(tasks.join("instruction.md"), "Say hello.").unwrap();
    std::fs::write(repository.path().join("tasks.yaml"), text).unwrap();
    collection::load(repository.path(), "tasks.yaml")
        .map(drop)
        .map_err(|e| e.to_string())
}

fn recipe() -> impl Strategy<Value = String> {
    prop::sample::select(RECIPES.to_vec()).prop_flat_map(damaged_text)
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn a_damaged_recipe_is_answered(text in recipe()) {
        let _ = graph::parse(&text);
    }

    #[test]
    fn a_damaged_environment_or_plan_is_answered(
        environment in damaged_text(ENVIRONMENT),
        plan in damaged_text(EVAL_PLAN),
    ) {
        let _ = Environment::parse(&environment);
        let _ = EvalPlan::parse(&plan);
    }

    #[test]
    fn a_damaged_machine_file_is_answered(
        machine in damaged_text(MACHINE),
        grants in damaged_text(GRANTS),
        connections in damaged_text(CONNECTIONS),
    ) {
        if let Ok(config) = MachineConfig::parse(&machine) {
            // What is accepted is what the checks allow: no literal token.
            for connection in config.connections.values() {
                prop_assert!(connection.token.as_deref().is_none_or(|t| t.starts_with("env:")));
            }
        }
        let _ = Grants::parse(&grants);
        let _ = Connections::parse(&connections);
    }
}

proptest! {
    // A collection is read from disk: fewer cases, each a directory.
    #![proptest_config(ProptestConfig { cases: 512, ..config() })]

    #[test]
    fn a_damaged_collection_or_task_definition_is_answered(
        text in damaged_text(COLLECTION),
        task in damaged_text(TASK),
    ) {
        let _ = collection_of(&text, TASK);
        let _ = collection_of(COLLECTION, &task);
    }
}
