// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec: when a gate rule turns out wrong and is fixed, the attempts it failed
//! on that ground alone are done again; nothing else is.

#![allow(clippy::unwrap_used)]

use splinter_repo_replay::audit::{
    default_config, history_diffs, open_context, run_blocking, scripted_models,
};
use splinter_repo_replay::fixture;
use splinter_repo_replay::ledger::{self, Outcome};
use splinter_repo_replay::plan_prompt::PlanStyle;
use splinter_repo_replay::scripted::{Behaviour, Specificity};

#[test]
fn only_attempts_that_failed_for_the_named_classes_alone_are_redone() {
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
    config.styles = vec![PlanStyle::Terse];

    // A stray file fails the gate with `junk` (and `unmotivated_location`).
    run_blocking(
        &ctx,
        &scripted_models(Behaviour::Junk, Specificity::NamesFiles, &truths),
        &config,
    )
    .unwrap();
    let honest = scripted_models(Behaviour::Honest, Specificity::NamesFiles, &truths);

    // Not named: left as it was.
    config.redo_classes = vec!["tree_mismatch".into()];
    let report = run_blocking(&ctx, &honest, &config).unwrap();
    assert!(
        report.entries.iter().all(|e| e.outcome == Outcome::Skipped),
        "{:?}",
        report.entries
    );

    // Named, but the failure has another reason too: left as it was.
    config.redo_classes = vec!["junk".into(), "unmotivated_location".into()];
    assert!(run_blocking(&ctx, &honest, &config)
        .unwrap()
        .entries
        .iter()
        .all(|e| e.outcome == Outcome::Skipped));

    // Every reason named: done again, and the new verdict is the last.
    config.redo_classes = ["junk", "unread_edit", "unmotivated_location"]
        .map(String::from)
        .to_vec();
    let report = run_blocking(&ctx, &honest, &config).unwrap();
    assert!(
        report.entries.iter().any(|e| e.outcome == Outcome::Passed),
        "{:?}",
        report.entries
    );
    let last = ledger::read(&state)
        .unwrap()
        .into_iter()
        .rfind(|e| e.experience.is_some())
        .unwrap();
    assert_eq!(last.outcome, Outcome::Passed);
    config.redo_classes = vec!["junk".into(), "unmotivated_location".into()];
    assert!(
        run_blocking(&ctx, &honest, &config)
            .unwrap()
            .entries
            .iter()
            .all(|e| e.outcome == Outcome::Skipped),
        "a passed attempt is not redone"
    );
}
