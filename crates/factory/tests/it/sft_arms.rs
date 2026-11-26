// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements controlled fine-tuning experiments whose
// arms differ only in what they supervise, for its clients. If your team needs
// expertise in measuring what an agent learns from its data, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The options of an SFT `dataset.project` stage that make the arms of a
//! controlled experiment from one set of attempts: which actions are
//! supervised, how many trials of a task are kept, a negative control, and
//! the composition every SFT dataset reports.

use serde_json::Value;
use splinter_factory::run::fork;

use crate::factory_run::{resolved, run};
use crate::fakes::{FakeBackend, FakeTrainer, Outcome, FIRST_CALL};
use crate::repo::Repo;

// seed: a recipe states every seed; the fake backend's outcomes are scripted, so none draws.
const RECIPE: &str = "schema: splinter.recipe/v2
name: arms
inputs:
  course: { kind: files, path: collections/vim.yaml }
models:
  student: \"brain:Qwen/Qwen3-4B\"
stages:
  - { id: tasks, uses: tasks.pin, inputs: { collection: $inputs.course }, with: { split: train } }
  - id: local
    uses: attempt.collect
    inputs: { tasks: $stages.tasks.tasks }
    with: { agent: sven, model: student, environment: environments/vim.yaml, attempts: 1, seed: 11 }
  - id: sft
    uses: dataset.project
    inputs: { attempts: [$stages.local.attempts] }
    with: { target: student, seed: 13 }
";

/// `RECIPE` with a second SFT stage `arm` given `with`.
// seed: the arm states the same seed as `sft`, so two arms differ only in `with`.
fn with_arm(with: &str) -> String {
    format!(
        "{RECIPE}  - id: arm\n    uses: dataset.project\n    inputs: {{ attempts: [$stages.local.attempts] }}\n    \
         with: {{ target: student, seed: 13, {with} }}\n"
    )
}

fn lines(path: &std::path::Path) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

/// Each record's task and, in order, whether each assistant message is supervised.
fn supervision(path: &std::path::Path) -> Vec<(String, Vec<bool>)> {
    let mut out: Vec<(String, Vec<bool>)> = lines(path)
        .iter()
        .map(|r| {
            let task = r["metadata"]["group"].as_str().unwrap().to_owned();
            let trained = r["messages"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|m| m["role"] == "assistant")
                .map(|m| m["train"].as_bool().unwrap())
                .collect();
            (task, trained)
        })
        .collect();
    out.sort();
    out
}

#[test]
fn productive_supervision_keeps_an_action_answered_only_by_errors_as_context() {
    let repo = Repo::new();
    let backend = FakeBackend::with(Outcome::Pass);
    backend.script("train/001-first", &[Outcome::PassAfterError]);
    let trainer = FakeTrainer::default();
    let dir = tempfile::tempdir().unwrap();
    run(
        &resolved(&repo, &with_arm("supervise: productive")),
        dir.path(),
        &backend,
        &trainer,
    )
    .unwrap();

    // Every action supervised by default, the failed one included.
    assert_eq!(
        supervision(&dir.path().join("sft/train.jsonl")),
        [
            ("train/001-first".to_owned(), vec![true, true, true]),
            ("train/002-second".to_owned(), vec![true, true, true]),
        ]
    );
    // The action whose call failed is context; the recovery and the final
    // answer are supervised; a trial with no error is unchanged.
    assert_eq!(
        supervision(&dir.path().join("arm/train.jsonl")),
        [
            ("train/001-first".to_owned(), vec![false, true, true]),
            ("train/002-second".to_owned(), vec![true, true, true]),
        ]
    );
    let masked = lines(&dir.path().join("arm/train.jsonl"))
        .into_iter()
        .find(|r| r["metadata"]["group"] == "train/001-first")
        .unwrap();
    let context = masked["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["role"] == "assistant")
        .unwrap();
    assert!(
        context["tool_calls"][0]["id"]
            .as_str()
            .unwrap()
            .ends_with(FIRST_CALL),
        "{context}"
    );
    let unchanged = |stage: &str| {
        lines(&dir.path().join(stage).join("train.jsonl"))
            .into_iter()
            .find(|r| r["metadata"]["group"] == "train/002-second")
            .unwrap()
    };
    assert_eq!(unchanged("sft"), unchanged("arm"));
    let manifest: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("arm/dataset.json")).unwrap(),
    )
    .unwrap();
    assert!(manifest.get("control").is_none());
    let report = std::fs::read_to_string(dir.path().join("arm/report.md")).unwrap();
    assert!(
        report.contains("1 action whose every call failed kept as context"),
        "{report}"
    );
}

#[test]
fn naming_the_default_supervision_changes_no_stage_identity() {
    let repo = Repo::new();
    let backend = FakeBackend::with(Outcome::Pass);
    let trainer = FakeTrainer::default();
    let first = tempfile::tempdir().unwrap();
    run(&resolved(&repo, RECIPE), first.path(), &backend, &trainer).unwrap();
    let second = tempfile::tempdir().unwrap();
    fork(first.path(), second.path()).unwrap();
    let named = RECIPE.replace("seed: 13 }", "seed: 13, supervise: all }");
    let computed: Vec<String> = run(&resolved(&repo, &named), second.path(), &backend, &trainer)
        .unwrap()
        .into_iter()
        .filter(|(_, computed)| *computed)
        .map(|(key, _)| key)
        .collect();
    assert!(computed.is_empty(), "{computed:?}");
}

#[test]
fn productive_supervision_needs_the_requests_the_model_was_sent() {
    let repo = Repo::new();
    let recipe = with_arm("supervise: productive, projection: portable-sft");
    let refused =
        splinter_factory::resolve::resolve(&repo.root, &recipe, crate::repo::connections())
            .err()
            .map(|e| format!("{e:?}"))
            .unwrap_or_default();
    assert!(refused.contains("supervise: productive"), "{refused}");
    let recipe = with_arm("objective: preference, supervise: productive");
    let refused =
        splinter_factory::resolve::resolve(&repo.root, &recipe, crate::repo::connections())
            .err()
            .map(|e| format!("{e:?}"))
            .unwrap_or_default();
    assert!(refused.contains("supervise"), "{refused}");
}
