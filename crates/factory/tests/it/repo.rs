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
schema: splinter.recipe/v2
name: vim-mastery
inputs:
  course: { kind: files, path: collections/vim.yaml }
models:
  student: "brain:Qwen/Qwen3-4B"
  teacher: "hosted:deepseek/deepseek-v4.1-flash"
stages:
  - id: tasks
    uses: tasks.pin
    inputs: { collection: $inputs.course }
    with: { split: train }
  - id: local
    uses: attempt.collect
    inputs: { tasks: $stages.tasks.tasks }
    with:
      agent: sven
      args: [--config, presets/sven.yaml]
      files: [presets/sven.yaml]
      model: student
      environment: environments/vim.yaml
      attempts: 2
      seed: 11
  - id: remote
    uses: attempt.collect
    inputs: { tasks: $stages.tasks.tasks, unsolved_by: [$stages.local.attempts] }
    with:
      agent: sven
      model: teacher
      environment: environments/vim.yaml
      environment_variables: { SVEN_KEY: "env:SVEN_KEY" }
      attempts: 2
      seed: 12
  - id: sft
    uses: dataset.project
    inputs: { attempts: [$stages.local.attempts, $stages.remote.attempts] }
    with: { target: student, seed: 13 }
  - id: lora-r8
    uses: model.train
    inputs: { dataset: $stages.sft.dataset }
    with: { base: student, method: lora, params: { rank: 8 }, seed: 1 }
  - id: lora-r16
    uses: model.train
    inputs: { dataset: $stages.sft.dataset }
    with: { base: student, method: lora, params: { rank: 16 }, seed: 1 }
  - id: eval-lora-r8
    uses: evaluation
    inputs: { candidate: $stages.lora-r8.model }
    with: { plan: eval/dev.yaml, agent: sven, baseline: student, comparisons: 2 }
  - id: eval-lora-r16
    uses: evaluation
    inputs: { candidate: $stages.lora-r16.model }
    with: { plan: eval/dev.yaml, agent: sven, baseline: student, comparisons: 2 }
  - id: release
    uses: promotion
    inputs: { evaluations: [$stages.eval-lora-r8.evaluation, $stages.eval-lora-r16.evaluation] }
    with: { promote: false }
"#;

/// Where the reference recipe's dataset stage begins: the recipe up to it
/// collects attempts only.
pub const DATASET: &str = "  - id: sft\n";
/// Where its training stages begin.
pub const TRAINING: &str = "  - id: lora-r8\n";
/// Where its evaluation stages begin.
pub const EVALUATION: &str = "  - id: eval-lora-r8\n";

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
            "schema: splinter.environment/v1\nsandbox: { provider: proot, resources: { timeout: 10m } }\nnetwork: { mode: open }\n",
        );
        // seed: the plan must state its seed; nothing here draws from it.
        write(
            &root,
            "eval/dev.yaml",
            "schema: splinter.evalplan/v1\ncollection: collections/vim.yaml\nsplit: eval\n\
             environment: environments/vim.yaml\nattempts: 2\nseed: 5\ndecision: { alpha: 0.05 }\n",
        );
        write(&root, "presets/sven.yaml", "agent: { max_turns: 20 }\n");
        write(&root, "recipes/vim.yaml", RECIPE);
        Self { _dir: dir, root }
    }

    pub fn write(&self, path: &str, text: &str) {
        write(&self.root, path, text);
    }
}

/// A `dataset.project` stage `sft` over the attempts of `sources`, with
/// `extra` parameters (`, projection: portable-sft`).
// seed: a recipe states the dataset's seed; the split it draws is not what these tests check.
pub fn dataset(sources: &[&str], extra: &str) -> String {
    let attempts: Vec<String> = sources
        .iter()
        .map(|s| format!("$stages.{s}.attempts"))
        .collect();
    format!(
        "  - id: sft\n    uses: dataset.project\n    inputs: {{ attempts: [{}] }}\n    \
         with: {{ target: student, seed: 13{extra} }}\n",
        attempts.join(", ")
    )
}

/// An `evaluation` stage `eval-<candidate>` of the candidate trained by the
/// stage `candidate`, under `plan`, against the model role `baseline`.
pub fn evaluation(candidate: &str, plan: &str, baseline: &str) -> String {
    format!(
        "  - id: eval-{candidate}\n    uses: evaluation\n    inputs: {{ candidate: $stages.{candidate}.model }}\n    \
         with: {{ plan: {plan}, agent: sven, baseline: {baseline} }}\n"
    )
}
