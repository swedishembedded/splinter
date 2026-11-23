// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! A recipe's `dataset.import`: records made elsewhere are carried over
//! byte for byte as a dataset that trains and is evaluated like any, a record
//! that cannot train is refused and never dropped, and the evaluation arms
//! run under the agent files and system prompt the records were made with.

use serde_json::json;
use splinter_factory::run::dataset::dataset_of;

use crate::factory_run::{resolved, run};
use crate::fakes::{FakeBackend, FakeTrainer, Outcome};
use crate::repo::Repo;

// seed: a recipe states every seed; the fake backend and trainer draw nothing from them.
const RECIPE: &str = r#"
schema: splinter.recipe/v2
name: imported
inputs:
  course: { kind: files, path: collections/vim.yaml }
  records: { kind: files, path: data/sft }
models:
  student: "brain:Qwen/Qwen3-4B"
stages:
  - id: sft
    uses: dataset.import
    inputs: { records: $inputs.records }
    with: { target: student }
  - id: lora
    uses: model.train
    inputs: { dataset: $stages.sft.dataset }
    with: { base: student, method: lora, params: { rank: 16 }, seed: 1 }
  - id: eval-lora
    uses: evaluation
    inputs: { candidate: $stages.lora.model }
    with:
      plan: eval/dev.yaml
      agent: sven
      args: [--config, presets/sven.yaml]
      files: [presets/sven.yaml]
      system_prompt: prompts/narrate.md
      baseline: student
"#;

fn record(group: &str, supervised: bool) -> String {
    json!({
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "do it"},
            {"role": "assistant", "content": "done", "train": supervised},
        ],
        "tools": [],
        "metadata": {"group": group},
    })
    .to_string()
}

fn repo(train: &[String], validation: &[String]) -> Repo {
    let repo = Repo::new();
    repo.write("data/sft/train.jsonl", &format!("{}\n", train.join("\n")));
    if !validation.is_empty() {
        repo.write(
            "data/sft/validation.jsonl",
            &format!("{}\n", validation.join("\n")),
        );
    }
    repo.write("prompts/narrate.md", "Say what you will do.\n");
    repo
}

fn failure(repo: &Repo) -> String {
    let recipe = resolved(repo, RECIPE);
    let dir = tempfile::tempdir().unwrap();
    run(
        &recipe,
        dir.path(),
        &FakeBackend::with(Outcome::Pass),
        &FakeTrainer::default(),
    )
    .unwrap_err()
    .to_string()
}

#[test]
fn supplied_records_train_unchanged_and_the_arms_run_under_the_records_prompt() {
    let train = [record("a", true), record("b", true)];
    let validation = [record("c", true)];
    let repo = repo(&train, &validation);
    let recipe = resolved(&repo, RECIPE);
    let backend = FakeBackend::with(Outcome::Pass);
    let trainer = FakeTrainer::default();
    let dir = tempfile::tempdir().unwrap();

    let first = run(&recipe, dir.path(), &backend, &trainer).unwrap();
    assert!(first.iter().all(|(_, computed)| *computed), "{first:?}");

    let view = dir.path().join("sft");
    let dataset = dataset_of(&view).unwrap();
    assert_eq!(dataset.projection, "imported");
    assert!(!dataset.thinking);
    assert_eq!(
        std::fs::read_to_string(view.join(&dataset.train.path)).unwrap(),
        format!("{}\n", train.join("\n"))
    );
    assert_eq!(
        std::fs::read_to_string(view.join(&dataset.validation.path)).unwrap(),
        format!("{}\n", validation.join("\n"))
    );
    assert_eq!(trainer.trained.lock().unwrap().len(), 1);

    // Every arm of the evaluation is given the same files and system prompt.
    let evaluated = backend.executed();
    assert!(!evaluated.is_empty());
    for spec in &evaluated {
        assert_eq!(spec.agent.args, ["--config", "presets/sven.yaml"]);
        assert_eq!(
            spec.agent
                .files
                .iter()
                .map(|f| f.path.as_str())
                .collect::<Vec<_>>(),
            ["presets/sven.yaml"]
        );
        assert_eq!(
            spec.agent.system_prompt.as_ref().map(|f| f.path.as_str()),
            Some("prompts/narrate.md")
        );
    }

    // A second run reuses everything; a changed record is a changed input.
    let second = run(&recipe, dir.path(), &backend, &trainer).unwrap();
    assert!(second.iter().all(|(_, computed)| !*computed), "{second:?}");
    repo.write(
        "data/sft/train.jsonl",
        &format!("{}\n", [record("a", true), record("d", true)].join("\n")),
    );
    let changed = resolved(&repo, RECIPE);
    let refused = run(&changed, dir.path(), &backend, &trainer).unwrap_err();
    assert!(refused.to_string().contains("data/sft"), "{refused}");
}

#[test]
fn a_record_that_cannot_train_is_refused_with_its_line_never_dropped() {
    let unsupervised = failure(&repo(&[record("a", true), record("b", false)], &[]));
    assert!(
        unsupervised.contains("train.jsonl line 2")
            && unsupervised.contains("supervises no message"),
        "{unsupervised}"
    );
    let not_chat = failure(&repo(&["{\"text\": \"x\"}".to_owned()], &[]));
    assert!(not_chat.contains("no `messages`"), "{not_chat}");
    let overlap = failure(&repo(&[record("a", true)], &[record("a", true)]));
    assert!(overlap.contains("groups in both splits"), "{overlap}");
    let reasoning = json!({
        "messages": [{"role": "assistant", "content": "x", "reasoning_content": "hm", "train": true}],
        "metadata": {"group": "r"},
    })
    .to_string();
    let thinking = failure(&repo(&[reasoning], &[]));
    assert!(thinking.contains("no-think mode"), "{thinking}");
}
