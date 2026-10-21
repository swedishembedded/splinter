// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec: a model that ends its turn announcing a call it never made (a real
//! 27B does this) is told to go on, a bounded number of times, instead of the
//! attempt being lost.

#![allow(clippy::unwrap_used)]

use splinter_repo_replay::audit::{
    default_config, history_diffs, open_context, run_blocking, scripted_models,
};
use splinter_repo_replay::fixture;
use splinter_repo_replay::ledger::Outcome;
use splinter_repo_replay::scripted::{Behaviour, Specificity};
use splinter_repo_replay::style::PlanStyle;

fn outcome_with(follow_ups: usize) -> (Outcome, Vec<String>) {
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
    config.follow_ups = follow_ups;
    let report = run_blocking(
        &ctx,
        &scripted_models(
            Behaviour::AnnouncesThenGoesOn,
            Specificity::NamesFiles,
            &truths,
        ),
        &config,
    )
    .unwrap();
    let entry = report
        .entries
        .iter()
        .find(|e| e.outcome != Outcome::Skipped)
        .unwrap();
    (entry.outcome, entry.classes.clone())
}

#[test]
fn an_announced_but_unmade_call_is_followed_up_a_bounded_number_of_times() {
    let (without, classes) = outcome_with(0);
    assert_eq!(without, Outcome::Failed);
    assert!(classes.iter().any(|c| c == "run_incomplete"), "{classes:?}");
    let (with, classes) = outcome_with(2);
    assert_eq!(with, Outcome::Passed, "{classes:?}");
}
