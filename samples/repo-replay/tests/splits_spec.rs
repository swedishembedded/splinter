// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec: train, eval and test are cut by time over the whole range, every
//! attempt has a readable trace, and the datasets hold only their own commits.

#![allow(clippy::unwrap_used)]

use splinter_repo_replay::audit::{
    default_config, history_diffs, open_context, run_blocking, scripted_models,
};
use splinter_repo_replay::dataset::DatasetSettings;
use splinter_repo_replay::fixture;
use splinter_repo_replay::scripted::{Behaviour, Specificity};
use splinter_repo_replay::splits::{
    band_edges, target_features, write_splits, Overlaps, SplitOptions,
};
use splinter_sdk::vocabulary::build::BuildInfo;
use splinter_sdk::vocabulary::prompt::CODING_AGENT_PROMPT;

/// The split of each of `total` commits, oldest first, by the percentages.
fn cut(total: usize, train: usize, eval: usize) -> Vec<&'static str> {
    let (train_until, eval_until) = band_edges(total, train, eval);
    (0..total)
        .map(|at| match f64::from(u32::try_from(at).unwrap()) {
            at if at <= train_until => "train",
            at if at <= eval_until => "eval",
            _ => "test",
        })
        .collect()
}

#[test]
fn the_cut_is_by_position_oldest_first() {
    assert_eq!(
        cut(10, 80, 10),
        ["train", "train", "train", "train", "train", "train", "train", "train", "eval", "test"]
    );
}

#[test]
fn the_cut_does_not_move_with_the_pass_rate_and_is_the_percent_of_the_commits_of_the_range() {
    // 452 commits at 80/10: the first 361 train, the next 45 eval, the rest test.
    let cuts = cut(452, 80, 10);
    assert_eq!(cuts.iter().filter(|c| **c == "train").count(), 361);
    assert_eq!(cuts.iter().filter(|c| **c == "eval").count(), 45);
    assert_eq!(cuts.iter().filter(|c| **c == "test").count(), 46);
    assert_eq!(cuts[360], "train");
    assert_eq!(cuts[361], "eval");
}

#[test]
fn a_change_and_its_revert_have_the_same_target_features_and_unrelated_changes_do_not() {
    let change = "diff --git a/calc.py b/calc.py\n--- a/calc.py\n+++ b/calc.py\n@@ -1,3 +1,4 @@\n def add(a, b):\n-    return a + b\n+    subtotal = compute_the_sum(a, b)\n+    return subtotal\n";
    let revert = "diff --git a/calc.py b/calc.py\n--- a/calc.py\n+++ b/calc.py\n@@ -1,4 +1,3 @@\n def add(a, b):\n-    subtotal = compute_the_sum(a, b)\n-    return subtotal\n+    return a + b\n";
    let other = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-print(\"hello world\")\n+print(\"goodbye world\")\n";
    assert_eq!(target_features(change), target_features(revert));
    assert!(target_features(change).is_disjoint(&target_features(other)));
}

#[test]
fn each_split_holds_only_its_commits_and_every_attempt_has_a_trace() {
    let work = tempfile::tempdir().unwrap();
    let source = work.path().join("source");
    std::fs::create_dir(&source).unwrap();
    let ids = fixture::build(&source).unwrap();
    let truths = history_diffs(&source).unwrap();
    let state = work.path().join("state");
    let ctx = open_context(&state).unwrap();
    let mut config = default_config(&source, &state);
    config.styles = vec![splinter_repo_replay::style::PlanStyle::Terse];
    let report = run_blocking(
        &ctx,
        &scripted_models(Behaviour::Honest, Specificity::NamesFiles, &truths),
        &config,
    )
    .unwrap();
    let attempts = report
        .entries
        .iter()
        .filter(|e| e.experience.is_some())
        .count();

    let out = work.path().join("out");
    let settings = DatasetSettings {
        serving_prompt: CODING_AGENT_PROMPT.to_string(),
        max_block: None,
    };
    let summary = write_splits(
        &ctx,
        &state,
        &out,
        &settings,
        &SplitOptions::default(),
        &BuildInfo::default(),
    )
    .unwrap();

    assert_eq!(summary["total_commits"], ids.len());
    let splits = &summary["splits"];
    let counted: u64 = ["train", "eval", "test"]
        .iter()
        .map(|s| splits[s]["commits"].as_u64().unwrap())
        .sum();
    assert_eq!(
        counted as usize,
        ids.len(),
        "every commit is in exactly one split"
    );
    assert_eq!(splits["train"]["first"], ids[0].as_str());
    assert_eq!(splits["test"]["last"], ids[ids.len() - 1].as_str());
    let records: u64 = ["train", "eval", "test"]
        .iter()
        .map(|s| splits[s]["records"].as_u64().unwrap())
        .sum();
    assert_eq!(
        records as usize, attempts,
        "every passed attempt is in one dataset"
    );
    let traces = ["train", "eval", "test"]
        .iter()
        .map(|s| std::fs::read_dir(out.join("traces").join(s)).map_or(0, Iterator::count))
        .sum::<usize>();
    assert_eq!(traces, attempts, "one trace per stored attempt");
}

#[test]
fn a_split_with_no_passed_attempt_has_no_records_but_still_has_its_traces() {
    let work = tempfile::tempdir().unwrap();
    let source = work.path().join("source");
    std::fs::create_dir(&source).unwrap();
    fixture::build(&source).unwrap();
    let truths = history_diffs(&source).unwrap();
    let state = work.path().join("state");
    let ctx = open_context(&state).unwrap();
    let mut config = default_config(&source, &state);
    config.styles = vec![splinter_repo_replay::style::PlanStyle::Terse];
    run_blocking(
        &ctx,
        &scripted_models(Behaviour::Junk, Specificity::NamesFiles, &truths),
        &config,
    )
    .unwrap();
    let out = work.path().join("out");
    let settings = DatasetSettings {
        serving_prompt: CODING_AGENT_PROMPT.to_string(),
        max_block: None,
    };
    let summary = write_splits(
        &ctx,
        &state,
        &out,
        &settings,
        &SplitOptions::default(),
        &BuildInfo::default(),
    )
    .unwrap();
    assert_eq!(summary["splits"]["train"]["records"], 0);
    assert!(std::fs::read_dir(out.join("traces/train")).unwrap().count() > 0);
}

/// A state of honest attempts over the fixture followed by a commit that
/// reverts the documentation and helper change made early in its history, and the source it came from.
fn state_with_a_revert(work: &std::path::Path) -> (splinter_sdk::Context, std::path::PathBuf) {
    let source = work.join("source");
    std::fs::create_dir(&source).unwrap();
    let ids = fixture::build(&source).unwrap();
    fixture::revert(&source, &ids[5]).unwrap();
    let truths = history_diffs(&source).unwrap();
    let state = work.join("state");
    let ctx = open_context(&state).unwrap();
    let mut config = default_config(&source, &state);
    config.styles = vec![splinter_repo_replay::style::PlanStyle::Terse];
    run_blocking(
        &ctx,
        &scripted_models(Behaviour::Honest, Specificity::NamesFiles, &truths),
        &config,
    )
    .unwrap();
    (ctx, state)
}

fn settings() -> DatasetSettings {
    DatasetSettings {
        serving_prompt: CODING_AGENT_PROMPT.to_string(),
        max_block: None,
    }
}

#[test]
fn a_test_commit_that_reverts_a_training_commit_is_refused_until_the_overlap_is_dropped() {
    let work = tempfile::tempdir().unwrap();
    let (ctx, state) = state_with_a_revert(work.path());
    let out = work.path().join("out");

    let refused = write_splits(
        &ctx,
        &state,
        &out,
        &settings(),
        &SplitOptions::default(),
        &BuildInfo::default(),
    );
    let message = format!("{:#}", refused.unwrap_err());
    assert!(message.contains("duplicate"), "{message}");

    let options = SplitOptions {
        overlaps: Overlaps::Drop,
        ..SplitOptions::default()
    };
    let summary = write_splits(
        &ctx,
        &state,
        &out,
        &settings(),
        &options,
        &BuildInfo::default(),
    )
    .unwrap();
    assert_eq!(summary["excluded"]["content_overlap"], 1);
    assert_eq!(summary["overlaps"].as_array().unwrap().len(), 1);
    // The held-out side of the pair stays: the revert is in the test split.
    assert_eq!(summary["splits"]["test"]["records"], 1);
}

#[test]
fn the_split_has_a_content_address_that_does_not_change_when_it_is_written_again() {
    let work = tempfile::tempdir().unwrap();
    let source = work.path().join("source");
    std::fs::create_dir(&source).unwrap();
    fixture::build(&source).unwrap();
    let truths = history_diffs(&source).unwrap();
    let state = work.path().join("state");
    let ctx = open_context(&state).unwrap();
    let mut config = default_config(&source, &state);
    config.styles = vec![splinter_repo_replay::style::PlanStyle::Terse];
    run_blocking(
        &ctx,
        &scripted_models(Behaviour::Honest, Specificity::NamesFiles, &truths),
        &config,
    )
    .unwrap();
    let write = |dir: &str| {
        write_splits(
            &ctx,
            &state,
            &work.path().join(dir),
            &settings(),
            &SplitOptions::default(),
            &BuildInfo::default(),
        )
        .unwrap()
    };
    let (first, second) = (write("a"), write("b"));
    assert!(first["split_digest"]
        .as_str()
        .unwrap()
        .starts_with("blake3:"));
    assert_eq!(first["split_digest"], second["split_digest"]);
}
