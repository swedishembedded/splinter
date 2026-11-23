// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! A scratch task repository: three Harbor tasks in two splits, an
//! environment, an evaluation plan, an agent file and a recipe over them.

use std::path::{Path, PathBuf};

use splinter_factory::connections::Connections;

pub struct Repo {
    _dir: tempfile::TempDir,
    pub root: PathBuf,
}

// seed: a recipe must state every seed; these values reach no random choice in
// these tests, which only resolve and plan the recipe.
pub const RECIPE: &str = r#"
schema: splinter.factory/v1
name: vim-mastery
tasks: { collection: collections/vim.yaml, split: train }
rollouts:
  - name: local
    agent: sven
    args: [--config, agents/sven.yaml]
    files: [agents/sven.yaml]
    model: "brain:Qwen/Qwen3-4B"
    environment: environments/vim.yaml
    attempts: 2
    seed: 11
  - name: remote
    agent: sven
    model: "hosted:deepseek/deepseek-v4.1-flash"
    environment: environments/vim.yaml
    environment_variables: { SVEN_KEY: "env:SVEN_KEY" }
    tasks: { unsolved_by: [local] }
    attempts: 2
    seed: 12
dataset: { name: sft, sources: [local, remote], target: "brain:Qwen/Qwen3-4B", seed: 13 }
training:
  base: "brain:Qwen/Qwen3-4B"
  variants:
    - { name: lora-r8, method: lora, params: { rank: 8 }, seed: 1 }
    - { name: lora-r16, method: lora, params: { rank: 16 }, seed: 1 }
eval: { plan: eval/dev.yaml, agent: sven, baseline: "brain:Qwen/Qwen3-4B", candidates: [lora-r8, lora-r16] }
release: { promote: false }
"#;

pub const CONNECTIONS: &str = r#"
connections:
  brain: { protocol: openai, url: "http://127.0.0.1:8990/v1" }
  hosted: { protocol: openai, url: "https://example.invalid/v1", token: "env:SVEN_KEY" }
"#;

pub fn connections() -> Connections {
    Connections::parse(CONNECTIONS).expect("connections")
}

fn write(root: &Path, path: &str, text: &str) {
    let path = root.join(path);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

fn task(root: &Path, id: &str, family: &str) {
    write(
        root,
        &format!("tasks/vim/{id}/task.toml"),
        &format!("[task]\nname = \"org/{id}\"\n[metadata]\nbench_family = \"{family}\"\n"),
    );
    write(
        root,
        &format!("tasks/vim/{id}/instruction.md"),
        &format!("Do {id}.\n"),
    );
    write(root, &format!("tasks/vim/{id}/tests/test.sh"), "exit 0\n");
}

impl Repo {
    pub fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        task(&root, "train/001-first", "course");
        task(&root, "train/002-second", "course");
        task(&root, "eval/001-held", "course");
        write(
            &root,
            "collections/vim.yaml",
            "schema: splinter.collection/v1\nname: vim\nroot: tasks/vim\n",
        );
        write(
            &root,
            "environments/vim.yaml",
            "schema: splinter.environment/v1\nsandbox: { provider: proot, resources: { timeout: 10m } }\n",
        );
        // seed: the plan must state its seed; nothing here draws from it.
        write(
            &root,
            "eval/dev.yaml",
            "schema: splinter.evalplan/v1\ncollection: collections/vim.yaml\nsplit: eval\n\
             environment: environments/vim.yaml\nattempts: 2\nseed: 5\ndecision: { alpha: 0.05 }\n",
        );
        write(&root, "agents/sven.yaml", "agent: { max_turns: 20 }\n");
        write(&root, "recipes/vim.yaml", RECIPE);
        Self { _dir: dir, root }
    }

    pub fn write(&self, path: &str, text: &str) {
        write(&self.root, path, text);
    }
}
