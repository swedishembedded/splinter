// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec: a killed run must not leave attempt trees behind. A run removes the
//! trees of attempts that no process holds any more when it starts, never the
//! tree of a live attempt, and every attempt removes its own trees when it
//! ends.

#![allow(clippy::unwrap_used)]

use std::path::Path;
use std::time::{Duration, SystemTime};

use splinter_repo_replay::audit::{
    default_config, history_diffs, open_context, run_blocking, scripted_models,
};
use splinter_repo_replay::fixture;
use splinter_repo_replay::scratch::{Claim, LIVE_FILE};
use splinter_repo_replay::scripted::{Behaviour, Specificity};

fn leftovers(scratch: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(scratch)
        .map(|rd| {
            rd.map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    names.sort();
    names
}

fn leave_tree(dir: &Path) {
    std::fs::create_dir_all(dir.join("tree/src")).unwrap();
    std::fs::write(dir.join("tree/src/lib.rs"), "fn f() {}\n").unwrap();
}

fn age(dir: &Path) {
    let old = SystemTime::now() - Duration::from_secs(24 * 3600);
    std::fs::File::open(dir).unwrap().set_modified(old).unwrap();
}

#[test]
fn a_run_sweeps_trees_no_attempt_holds_and_keeps_live_ones() {
    let work = tempfile::tempdir().unwrap();
    let source = work.path().join("source");
    std::fs::create_dir(&source).unwrap();
    let ids = fixture::build(&source).unwrap();
    let truths = history_diffs(&source).unwrap();
    let state = work.path().join("state");
    let scratch = state.join("scratch");

    // Left by a killed run of the current binary: marker present, nobody holds it.
    let dead = scratch.join("attempt-dead");
    leave_tree(&dead);
    std::fs::write(dead.join(LIVE_FILE), "").unwrap();
    // Left by a binary from before the marker existed: no marker, old.
    let ancient = scratch.join("attempt-ancient");
    leave_tree(&ancient);
    age(&ancient);
    // Another process is working in this one right now.
    let live = Claim::create(&scratch, "attempt-").unwrap();
    leave_tree(live.path());
    // A directory that is not ours.
    std::fs::create_dir_all(scratch.join("keep-me")).unwrap();

    let ctx = open_context(&state).unwrap();
    let mut config = default_config(&source, &state);
    config.from = Some(ids[1].clone());
    config.to = Some(ids[2].clone());
    run_blocking(
        &ctx,
        &scripted_models(Behaviour::Honest, Specificity::NamesFiles, &truths),
        &config,
    )
    .unwrap();

    let names = leftovers(&scratch);
    let live_name = live
        .path()
        .file_name()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    assert_eq!(names, vec![live_name, "keep-me".to_string()], "{names:?}");
    assert!(live.path().join("tree/src/lib.rs").exists());
}

#[test]
fn a_finished_run_leaves_no_attempt_tree() {
    let work = tempfile::tempdir().unwrap();
    let source = work.path().join("source");
    std::fs::create_dir(&source).unwrap();
    let ids = fixture::build(&source).unwrap();
    let truths = history_diffs(&source).unwrap();
    let state = work.path().join("state");
    let ctx = open_context(&state).unwrap();
    let mut config = default_config(&source, &state);
    config.from = Some(ids[1].clone());
    config.to = Some(ids[2].clone());
    for behaviour in [Behaviour::Honest, Behaviour::Junk] {
        run_blocking(
            &ctx,
            &scripted_models(behaviour, Specificity::NamesFiles, &truths),
            &config,
        )
        .unwrap();
        assert_eq!(leftovers(&state.join("scratch")), Vec::<String>::new());
    }
}

#[test]
fn a_claim_removes_its_tree_when_dropped() {
    let work = tempfile::tempdir().unwrap();
    let claim = Claim::create(work.path(), "attempt-").unwrap();
    leave_tree(claim.path());
    let path = claim.path().to_path_buf();
    drop(claim);
    assert!(!path.exists());
}
