// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec: what the sample reads from git does not depend on the reader's git
//! configuration. A hostile `.gitconfig` (forced colour, another diff
//! algorithm, an external diff driver, prefix and quoting changes) changes
//! neither the commit list, nor the diffs the agent is shown, nor a verdict.

#![allow(clippy::unwrap_used)]

use splinter_repo_replay::audit::{
    default_config, history_diffs, open_context, run_blocking, scripted_models, verdicts,
};
use splinter_repo_replay::fixture;
use splinter_repo_replay::git::mirror;
use splinter_repo_replay::grounding_prompt::mask_diff;
use splinter_repo_replay::scripted::{Behaviour, Specificity};
use splinter_repo_replay::scrub::Scrubber;
use splinter_repo_replay::select::{prepare, Selected};

const HOSTILE: &str = "[color]\n\tui = always\n[diff]\n\talgorithm = histogram\n\texternal = false\n\tnoprefix = true\n\tmnemonicPrefix = true\n[core]\n\tquotepath = true\n\tpager = false\n\tautocrlf = true\n";

/// Commit list, masked diffs and honest verdicts of the fixture.
fn observe(
    work: &std::path::Path,
    source: &std::path::Path,
) -> (Vec<String>, Vec<String>, Vec<String>) {
    let git = mirror(source, &work.join("mirror.git")).unwrap();
    let chain = git.first_parent_chain(None, None).unwrap();
    let shas = chain.iter().map(|c| c.sha.clone()).collect();
    let scrubber = Scrubber::new([fixture::AUTHOR, fixture::AUTHOR_EMAIL]);
    let diffs = chain
        .iter()
        .filter_map(
            |m| match prepare(&git, m, &scrubber, &Default::default()).unwrap() {
                Selected::Replay(plan) => Some(mask_diff(&plan.diff)),
                Selected::Skipped(..) => None,
            },
        )
        .collect();
    let truths = history_diffs(source).unwrap();
    let state = work.join("state");
    let ctx = open_context(&state).unwrap();
    let report = run_blocking(
        &ctx,
        &scripted_models(Behaviour::Honest, Specificity::NamesFiles, &truths),
        &default_config(source, &state),
    )
    .unwrap();
    let outcome = verdicts(&report.entries)
        .iter()
        .map(|v| format!("{v:?}"))
        .collect();
    (shas, diffs, outcome)
}

#[test]
fn a_hostile_git_configuration_changes_nothing() {
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("source");
    std::fs::create_dir(&source).unwrap();
    fixture::build(&source).unwrap();

    let clean = observe(&root.path().join("clean"), &source);

    let home = root.path().join("home");
    std::fs::create_dir_all(home.join(".config/git")).unwrap();
    std::fs::write(home.join(".gitconfig"), HOSTILE).unwrap();
    std::fs::write(home.join(".config/git/config"), HOSTILE).unwrap();
    // The hostile configuration is real: plain git, with this home, colours its output.
    let plain = std::process::Command::new("git")
        .args(["-C"])
        .arg(&source)
        .args(["log", "-1"])
        .env("HOME", &home)
        .env_remove("GIT_CONFIG_GLOBAL")
        .output()
        .unwrap();
    assert!(
        String::from_utf8_lossy(&plain.stdout).contains('\u{1b}'),
        "the hostile config must bite plain git"
    );

    std::env::set_var("HOME", &home);
    std::env::set_var("XDG_CONFIG_HOME", home.join(".config"));
    let hostile = observe(&root.path().join("hostile"), &source);

    assert_eq!(clean.0, hostile.0, "commit list");
    assert_eq!(clean.1, hostile.1, "masked diffs");
    assert_eq!(clean.2, hostile.2, "verdicts");
    assert!(!clean.1.is_empty() && !clean.2.is_empty());
}
