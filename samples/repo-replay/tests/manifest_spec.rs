// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec: a built dataset comes with a manifest beside `splits.json` that says
//! where it comes from, what is in each split, how long its records are, what
//! was left out and why, what was scrubbed (counts only), and whether the
//! trainer takes it, all without a model or a device.

#![allow(clippy::unwrap_used)]

use splinter_repo_replay::audit::{
    default_config, history_diffs, open_context, run_blocking, scripted_models,
};
use splinter_repo_replay::dataset::DatasetSettings;
use splinter_repo_replay::fixture;
use splinter_repo_replay::manifest::FILE;
use splinter_repo_replay::scripted::{Behaviour, Specificity};
use splinter_repo_replay::splits::{write_splits, SplitOptions};
use splinter_repo_replay::style::PlanStyle;
use splinter_sdk::vocabulary::build::{BuildInfo, Revisions};
use splinter_sdk::vocabulary::prompt::CODING_AGENT_PROMPT;

fn manifest_of_an_honest_run(max_block: Option<usize>) -> (serde_json::Value, serde_json::Value) {
    let work = tempfile::tempdir().unwrap();
    let source = work.path().join("source");
    std::fs::create_dir(&source).unwrap();
    fixture::build(&source).unwrap();
    let truths = history_diffs(&source).unwrap();
    let state = work.path().join("state");
    let ctx = open_context(&state).unwrap();
    let mut config = default_config(&source, &state);
    config.styles = vec![PlanStyle::Terse, PlanStyle::Detailed];
    run_blocking(
        &ctx,
        &scripted_models(Behaviour::Honest, Specificity::NamesFiles, &truths),
        &config,
    )
    .unwrap();
    let out = work.path().join("out");
    let build = BuildInfo {
        revisions: Revisions {
            splinter: Some("1111111".into()),
            sven: Some("2222222".into()),
            brain: Some("3333333".into()),
        },
        ..BuildInfo::default()
    };
    let settings = DatasetSettings {
        serving_prompt: CODING_AGENT_PROMPT.to_string(),
        max_block,
    };
    let summary = write_splits(
        &ctx,
        &state,
        &out,
        &settings,
        &SplitOptions::default(),
        &build,
    )
    .unwrap();
    let manifest = serde_json::from_slice(&std::fs::read(out.join(FILE)).unwrap()).unwrap();
    (summary, manifest)
}

#[test]
fn the_manifest_names_where_the_dataset_comes_from() {
    let (summary, manifest) = manifest_of_an_honest_run(None);
    let from = &manifest["provenance"];
    assert_eq!(from["revisions"]["splinter"], "1111111");
    assert_eq!(from["revisions"]["sven"], "2222222");
    assert_eq!(from["revisions"]["brain"], "3333333");
    assert_eq!(from["source"], "source");
    assert!(from["git_version"]
        .as_str()
        .unwrap()
        .starts_with("git version"));
    assert!(!from["harness_documents"].as_array().unwrap().is_empty());
    assert!(from["serving_prompt_digest"].is_string());
    assert_eq!(manifest["split_digest"], summary["split_digest"]);
}

#[test]
fn the_manifest_counts_each_split_by_record_type_and_style() {
    let (summary, manifest) = manifest_of_an_honest_run(None);
    let mut records = 0;
    for split in ["train", "eval", "test"] {
        let section = &manifest["splits"][split];
        let n = summary["splits"][split]["records"].as_u64().unwrap();
        assert_eq!(section["records"].as_u64().unwrap(), n, "{split}");
        let styles: u64 = section["by_style"]
            .as_object()
            .unwrap()
            .values()
            .map(|v| v.as_u64().unwrap())
            .sum();
        assert_eq!(styles, n, "{split}");
        assert_eq!(
            section["first_try"].as_u64().unwrap() + section["corrected"].as_u64().unwrap(),
            n
        );
        records += n;
    }
    assert!(records > 0);
    assert!(manifest["skipped_commits"].as_object().unwrap().len() > 1);
    assert!(manifest["scrubbed"]["attempts"].as_u64().unwrap() > 0);
    for kind in ["trailers", "people", "secrets"] {
        assert!(manifest["scrubbed"][kind].is_u64(), "{kind}");
    }
}

#[test]
fn the_manifest_measures_the_records_and_says_how() {
    let (_, manifest) = manifest_of_an_honest_run(Some(100_000));
    let lengths = &manifest["splits"]["train"]["token_lengths"];
    let (min, p50, p90, max) = (
        lengths["min"].as_u64().unwrap(),
        lengths["p50"].as_u64().unwrap(),
        lengths["p90"].as_u64().unwrap(),
        lengths["max"].as_u64().unwrap(),
    );
    assert!(0 < min && min <= p50 && p50 <= p90 && p90 <= max);
    assert_eq!(
        lengths["records"], manifest["splits"]["train"]["records"],
        "one length per record"
    );
    assert!(lengths["counter"].as_str().unwrap().len() > 5);
    assert_eq!(lengths["over_max_block"], 0);
}

#[test]
fn the_trainer_would_take_each_dataset_and_every_record_opens_with_the_serving_prompt() {
    let (_, manifest) = manifest_of_an_honest_run(None);
    let trained = &manifest["splits"]["train"]["trainer"];
    assert_eq!(trained["accepts"], true, "{trained}");
    assert_eq!(trained["regime"], "Sft");
    assert_eq!(trained["records_not_under_the_serving_prompt"], 0);
    assert!(trained["examples"].as_u64().unwrap() >= 1);
}
