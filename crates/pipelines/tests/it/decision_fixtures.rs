// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for fitting small decision models
// to labelled choices for its clients. If your team needs expertise in turning
// agent benchmark runs into auditable, leak-free fixtures then you can procure
// our services by sending an email to info@swedishembedded.com.

//! Labelled decision fixtures: auditable labels and a split that cannot leak.

#![allow(clippy::unwrap_used)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde_json::json;
use splinter_pipelines::decision_fixtures::{
    assign_splits, build, to_jsonl, LabelSource, Split, SplitPlan,
};

const SUITE: &str = "tool_selection";

fn task_id(seed: u64) -> String {
    format!("{SUITE}-s{seed}")
}

fn target(seed: u64) -> String {
    format!("catalog-target_{seed}")
}

/// Writes one run of `variant` under `out` as the benchmark does: the record,
/// the task with its oracle label and, when asked, the decisions served.
fn write_run(out: &Path, seed: u64, variant: &str, role: &str, tokens: u64, decisions: bool) {
    let dir = out.join("runs").join(task_id(seed)).join(variant);
    std::fs::create_dir_all(&dir).unwrap();
    let query = format!("find thing {seed}");
    let record = json!({
        "schema": "sven.ablation.run/1",
        "task_id": task_id(seed),
        "suite": SUITE,
        "seed": seed,
        "variant": {"id": variant, "name": variant, "role": role},
        "probe": null,
        "status": {"state": "completed"},
        "tokens": {"total": tokens, "recovery_expansions": 2},
        "score": {"passed": true},
    });
    std::fs::write(dir.join("record.json"), record.to_string()).unwrap();
    let task = json!({
        "schema": "sven.ablation.task/1",
        "oracle_labels": [{
            "decision_type": "tool_routing",
            "state_contains": query,
            "correct_option": target(seed),
        }],
    });
    std::fs::write(dir.join("task.json"), task.to_string()).unwrap();
    if !decisions {
        return;
    }
    // Eight tools in name order; the right one is among them.
    let mut options: Vec<(String, Option<String>)> = (0..7)
        .map(|n| {
            (
                format!("catalog-other_{n}"),
                Some(format!("Does thing {n}.")),
            )
        })
        .collect();
    options.push((target(seed), Some("The right one.".into())));
    options.sort();
    let served = |state: String, options: &[(String, Option<String>)]| {
        json!({
            "inputs_digest": "d1",
            "state": state,
            "questions": [{
                "id": "tool", "kind": "choice",
                "instruction": "Which option fits the task best?",
                "options": options, "top": null, "top_probability": null,
            }],
        })
    };
    let mut reversed = options.clone();
    reversed.reverse();
    let lines = [
        served(format!("Task: {query}"), &options),
        // The same question in the other option order.
        served(format!("Task: {query}"), &reversed),
        // A question no label covers.
        served("Task: something nobody labelled".into(), &options),
    ];
    let text: Vec<String> = lines.iter().map(ToString::to_string).collect();
    std::fs::write(dir.join("decisions.jsonl"), text.join("\n") + "\n").unwrap();
}

fn dir_with_runs(seeds: &[u64]) -> tempfile::TempDir {
    let out = tempfile::tempdir().unwrap();
    for seed in seeds {
        write_run(out.path(), *seed, "A", "baseline", 10_000, false);
        for variant in ["C", "G"] {
            write_run(out.path(), *seed, variant, "heuristic", 4_000, true);
        }
    }
    out
}

#[test]
fn an_example_is_a_state_options_and_the_labelled_option_with_its_provenance() {
    let out = dir_with_runs(&[1]);
    let examples = build(out.path(), &SplitPlan::with_seed(0)).unwrap();
    // One example per variant that served decisions; the second option order
    // and the unlabelled question add none.
    assert_eq!(examples.len(), 2, "{examples:#?}");
    let e = &examples[0];
    assert_eq!(e.schema, "splinter.decision-fixture/1");
    assert_eq!(e.decision_type, "tool_routing");
    assert_eq!(e.label_source, LabelSource::Oracle);
    assert_eq!(e.options.len(), 8);
    assert_eq!(e.options[e.label_index].id, e.label);
    assert_eq!(
        e.options[e.label_index].description.as_deref(),
        Some("The right one.")
    );
    assert!(e.question.state.starts_with("Task: "));
    assert_eq!(e.state_ref.task, "tool_selection-s1");
    assert_eq!(e.state_ref.inputs_digest, "d1");
    assert_eq!(e.outcome.llm_tokens_saved, Some(6_000));
    assert_eq!(e.outcome.recovery_calls, 2);
    assert!(e.outcome.task_passed);
}

#[test]
fn the_output_is_json_lines_that_read_back() {
    let out = dir_with_runs(&[1]);
    let text = to_jsonl(&build(out.path(), &SplitPlan::with_seed(0)).unwrap()).unwrap();
    assert!(!text.is_empty());
    for line in text.lines() {
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        for key in [
            "decision_type",
            "state_ref",
            "question",
            "options",
            "label",
            "label_index",
            "label_source",
            "outcome",
            "split",
        ] {
            assert!(v.get(key).is_some(), "{key}");
        }
        assert_eq!(v["label_source"], "oracle");
    }
}

#[test]
fn every_example_of_a_task_is_in_one_split_whatever_the_variant() {
    let out = dir_with_runs(&[1, 2, 3, 4, 5]);
    let examples = build(out.path(), &SplitPlan::with_seed(0)).unwrap();
    let mut by_task: BTreeMap<&str, BTreeSet<Split>> = BTreeMap::new();
    for e in &examples {
        by_task
            .entry(&e.state_ref.task)
            .or_default()
            .insert(e.split);
    }
    assert_eq!(by_task.len(), 5);
    assert!(by_task.values().all(|s| s.len() == 1), "{by_task:?}");
    let splits: BTreeSet<Split> = by_task.values().flatten().copied().collect();
    assert_eq!(
        splits.len(),
        3,
        "five seeds fill train, validation and test"
    );
}

#[test]
fn the_same_seed_gives_the_same_split_and_another_seed_may_not() {
    let groups: BTreeSet<(String, u64)> = (1..=10).map(|s| (SUITE.to_string(), s)).collect();
    let plan = SplitPlan::with_seed;
    let a = assign_splits(&groups, &plan(1));
    assert_eq!(a, assign_splits(&groups, &plan(1)));
    assert_ne!(a, assign_splits(&groups, &plan(2)));
    let count = |s: Split| a.values().filter(|x| **x == s).count();
    assert_eq!(
        (
            count(Split::Test),
            count(Split::Validation),
            count(Split::Train)
        ),
        (2, 2, 6)
    );
}

#[test]
fn a_family_with_fewer_than_three_seeds_is_all_train_and_a_holdout_family_is_all_test() {
    let groups: BTreeSet<(String, u64)> = [
        (SUITE, 1),
        (SUITE, 2),
        ("relevant_file", 1),
        ("relevant_file", 2),
        ("relevant_file", 3),
    ]
    .into_iter()
    .map(|(s, n)| (s.to_string(), n))
    .collect();
    let plan = SplitPlan {
        holdout_suites: vec!["relevant_file".into()],
        ..SplitPlan::with_seed(0)
    };
    let s = assign_splits(&groups, &plan);
    assert!(s
        .iter()
        .filter(|((suite, _), _)| suite == SUITE)
        .all(|(_, v)| *v == Split::Train));
    assert!(s
        .iter()
        .filter(|((suite, _), _)| suite == "relevant_file")
        .all(|(_, v)| *v == Split::Test));
}

#[test]
fn a_run_without_served_decisions_contributes_nothing() {
    let out = tempfile::tempdir().unwrap();
    write_run(out.path(), 1, "A", "baseline", 10_000, false);
    write_run(out.path(), 1, "B", "heuristic", 4_000, false);
    assert!(build(out.path(), &SplitPlan::with_seed(0))
        .unwrap()
        .is_empty());
}

#[test]
fn a_holdout_family_no_run_belongs_to_is_refused_and_a_bad_record_is_never_skipped() {
    let out = dir_with_runs(&[1]);
    let plan = SplitPlan {
        holdout_suites: vec!["tool_selecton".into()],
        ..SplitPlan::with_seed(0)
    };
    assert!(build(out.path(), &plan)
        .unwrap_err()
        .contains("tool_selecton"));

    let record = out
        .path()
        .join("runs")
        .join(task_id(1))
        .join("C")
        .join("record.json");
    std::fs::write(&record, r#"{"schema": "other/1"}"#).unwrap();
    assert!(build(out.path(), &SplitPlan::with_seed(0)).is_err());
}
