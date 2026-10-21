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
use splinter_repo_replay::serving_prompt::SERVING_PROMPT;
use splinter_repo_replay::splits::{split_of, write_splits};

#[test]
fn the_cut_is_by_position_oldest_first() {
    let cuts: Vec<&str> = (0..10).map(|i| split_of(i, 10, 80, 10)).collect();
    assert_eq!(
        cuts,
        ["train", "train", "train", "train", "train", "train", "train", "train", "eval", "test"]
    );
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
        serving_prompt: SERVING_PROMPT.to_string(),
        max_block: None,
    };
    let summary = write_splits(&ctx, &state, &out, &settings, 80, 10, Default::default()).unwrap();

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
        serving_prompt: SERVING_PROMPT.to_string(),
        max_block: None,
    };
    let summary = write_splits(&ctx, &state, &out, &settings, 80, 10, Default::default()).unwrap();
    assert_eq!(summary["splits"]["train"]["records"], 0);
    assert!(std::fs::read_dir(out.join("traces/train")).unwrap().count() > 0);
}
