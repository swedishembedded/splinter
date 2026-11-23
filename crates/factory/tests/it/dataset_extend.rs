// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! A recipe's `dataset.extend`: new records join an existing dataset only as
//! new solutions of their task. The base is carried over byte for byte, a
//! repeated solution is left out and listed, a task's side never changes, and
//! the same inputs give the same dataset.

use serde_json::json;
use splinter_factory::run::dataset::dataset_of;

use crate::factory_run::{resolved, run};
use crate::fakes::{FakeBackend, FakeTrainer, Outcome};
use crate::repo::Repo;

// seed: a recipe states every seed; the fake backend and trainer draw nothing from them.
const RECIPE: &str = r#"
schema: splinter.recipe/v2
name: extended
inputs:
  course: { kind: files, path: collections/vim.yaml }
  base: { kind: files, path: data/base }
  new: { kind: files, path: data/new }
models:
  student: "brain:Qwen/Qwen3-4B"
stages:
  - id: base
    uses: dataset.import
    inputs: { records: $inputs.base }
    with: { target: student }
  - id: new
    uses: dataset.import
    inputs: { records: $inputs.new }
    with: { target: student }
  - id: grown
    uses: dataset.extend
    inputs: { base: $stages.base.dataset, candidates: $stages.new.dataset }
    with: { target: student, per_task: 3, ignore_call_arguments: [intent], holdout_seed: 11 }
"#;

fn record(task: &str, command: &str, intent: &str) -> String {
    json!({
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "do it"},
            {"role": "assistant", "content": "ok", "train": true, "tool_calls": [{
                "id": "c0", "type": "function",
                "function": {"name": "vim", "arguments": json!({"commands": [command], "intent": intent}).to_string()}
            }]},
            {"role": "tool", "tool_call_id": "c0", "content": "done"},
        ],
        "tools": [],
        "metadata": {"task": task},
    })
    .to_string()
}

fn write(repo: &Repo, dir: &str, train: &[String], validation: &[String]) {
    repo.write(
        &format!("{dir}/train.jsonl"),
        &format!("{}\n", train.join("\n")),
    );
    if !validation.is_empty() {
        repo.write(
            &format!("{dir}/validation.jsonl"),
            &format!("{}\n", validation.join("\n")),
        );
    }
}

fn lines(path: &std::path::Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect()
}

#[test]
fn only_new_solutions_join_and_no_task_changes_side() {
    let base_train = [record("a", "x", "one"), record("b", "y", "one")];
    let base_validation = [record("h", "z", "one")];
    let repo = Repo::new();
    write(&repo, "data/base", &base_train, &base_validation);
    write(
        &repo,
        "data/new",
        &[
            // The base's solution with other words in its intent: a repeat.
            record("a", "x", "worded differently"),
            // Three new solutions of a, which has one record: two fit under per_task of 3.
            record("a", "w", "new"),
            record("a", "v", "new"),
            record("a", "u", "new"),
            // A new solution of the held-out task stays held out.
            record("h", "q", "new"),
        ],
        &[],
    );
    let recipe = resolved(&repo, RECIPE);
    let dir = tempfile::tempdir().unwrap();
    let (backend, trainer) = (FakeBackend::with(Outcome::Pass), FakeTrainer::default());

    let first = run(&recipe, dir.path(), &backend, &trainer).unwrap();
    assert!(first.iter().all(|(_, computed)| *computed), "{first:?}");

    let view = dir.path().join("grown");
    let grown = dataset_of(&view).unwrap();
    assert_eq!(grown.projection, "extended");
    let train = lines(&view.join(&grown.train.path));
    let validation = lines(&view.join(&grown.validation.path));
    // The base comes first, byte for byte and in order.
    assert_eq!(train[..2], base_train);
    assert_eq!(validation[0], base_validation[0]);
    // a had one record and per_task is 3: two of its three new solutions fit, one is over the cap.
    let of = |lines: &[String], task: &str| {
        lines
            .iter()
            .filter(|l| l.contains(&format!(r#""task":"{task}""#)))
            .count()
    };
    assert_eq!(of(&train, "a"), 3, "{train:?}");
    assert_eq!(of(&validation, "h"), 2, "a held-out task stays held out");
    assert_eq!(of(&train, "h"), 0);
    let left_out = grown.exclusions.clone();
    assert_eq!(left_out.get("duplicate_solution"), Some(&1), "{left_out:?}");
    assert_eq!(left_out.get("over_task_cap"), Some(&1), "{left_out:?}");

    // The lineage names the parents and what was left out.
    let lineage: serde_json::Value =
        serde_json::from_slice(&std::fs::read(view.join("lineage.json")).unwrap()).unwrap();
    assert_eq!(lineage["admitted"]["train"], 2);
    assert_eq!(lineage["admitted"]["validation"], 1);
    assert_eq!(lineage["per_task"], 3);

    // The same inputs give the same dataset, and a run again reuses it.
    let second = run(&recipe, dir.path(), &backend, &trainer).unwrap();
    assert!(second.iter().all(|(_, computed)| !*computed), "{second:?}");
}
