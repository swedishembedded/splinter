// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec (RED until the agent's shell is confined, consolidation commit 8): what
//! the agent must not reach is not just flagged afterwards, it cannot be read.
//! An agent whose shell prints a file of the source repository, of the mirror
//! of the history or of the hidden case directory gets nothing back: the
//! sandbox refuses, so the content is in no tool result and so in no record.
//!
//! Today the shell is sven's unconfined coding shell and only the gate's
//! `forbidden_shell` / `outside_tree` rules notice (the class stays as a second
//! line of defence, `audit`).

#![allow(clippy::unwrap_used)]

use std::path::Path;

use splinter_repo_replay::audit::{
    default_config, history_diffs, open_context, run_blocking, scripted_models,
};
use splinter_repo_replay::fixture;
use splinter_repo_replay::scripted::{Behaviour, Specificity};
use splinter_repo_replay::style::PlanStyle;
use splinter_repo_replay::trace::Trace;
use splinter_sdk::vocabulary::digest::Digest;
use splinter_sdk::vocabulary::experience::ExperienceId;

/// What every shell result of the attempt at the fixture's `edit` commit holds
/// when the agent printed `expression`.
fn shell_results(expression: &dyn Fn(&Path, &Path) -> String) -> Vec<String> {
    let work = tempfile::tempdir().unwrap();
    let source = work.path().join("source");
    std::fs::create_dir(&source).unwrap();
    let ids = fixture::build(&source).unwrap();
    let truths = history_diffs(&source).unwrap();
    let state = work.path().join("state");
    std::fs::create_dir(&state).unwrap();
    let ctx = open_context(&state).unwrap();
    let mut config = default_config(&source, &state);
    config.from = Some(ids[1].clone());
    config.to = Some(ids[2].clone());
    config.styles = vec![PlanStyle::Terse];
    let state = state.canonicalize().unwrap();
    let source = source.canonicalize().unwrap();
    let report = run_blocking(
        &ctx,
        &scripted_models(
            Behaviour::ReadsPath(expression(&source, &state)),
            Specificity::NamesFiles,
            &truths,
        ),
        &config,
    )
    .unwrap();
    let hex = report
        .entries
        .iter()
        .find_map(|e| e.experience.clone())
        .ok_or("the attempt was stored")
        .unwrap();
    let experience = ctx
        .experiences()
        .get(&ExperienceId(Digest::from_content_hex(&hex).unwrap()))
        .unwrap();
    Trace::of(&experience.trajectory)
        .calls
        .iter()
        .filter(|c| c.tool == "shell")
        .map(|c| c.result.clone())
        .collect()
}

#[test]
fn a_file_of_the_source_repository_cannot_be_read_from_the_agents_shell() {
    let results = shell_results(&|source, _| format!("{}/README.md", source.display()));
    assert!(
        results
            .iter()
            .all(|r| !r.contains("A tiny arithmetic library")),
        "the source repository's README was printed: {results:?}"
    );
}

#[test]
fn the_mirror_of_the_history_cannot_be_read_from_the_agents_shell() {
    let results = shell_results(&|_, state| format!("{}/mirror.git/HEAD", state.display()));
    assert!(
        results.iter().all(|r| !r.contains("ref:")),
        "the mirror's HEAD was printed: {results:?}"
    );
}

#[test]
fn the_hidden_case_directory_cannot_be_read_from_the_agents_shell() {
    let results =
        shell_results(&|_, state| format!("{}/scratch/target-*/case.json", state.display()));
    assert!(
        results
            .iter()
            .all(|r| !r.contains("\"plan\"") && !r.contains("\"diff\"")),
        "the case file was printed: {results:?}"
    );
}
