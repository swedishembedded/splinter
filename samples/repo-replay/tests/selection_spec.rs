// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec: which commits are replayed, and what of a commit the agent and the
//! plan are shown.

#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;

use splinter_repo_replay::fixture;
use splinter_repo_replay::git::mirror;
use splinter_repo_replay::scrub::Scrubber;
use splinter_repo_replay::select::{prepare, SelectConfig, Selected, Skip};

fn classify(config: &SelectConfig) -> BTreeMap<&'static str, Vec<Selected>> {
    let work = tempfile::tempdir().unwrap();
    let source = work.path().join("source");
    std::fs::create_dir(&source).unwrap();
    fixture::build(&source).unwrap();
    let git = mirror(&source, &work.path().join("mirror.git")).unwrap();
    let chain = git.first_parent_chain(None, None).unwrap();
    let scrubber = Scrubber::new([fixture::AUTHOR, fixture::AUTHOR_EMAIL]);
    let mut out: BTreeMap<&'static str, Vec<Selected>> = BTreeMap::new();
    for meta in &chain {
        let selected = prepare(&git, meta, &scrubber, config).unwrap();
        let class = match &selected {
            Selected::Replay(_) => "replay",
            Selected::Skipped(_, why) => why.class(),
        };
        out.entry(class).or_default().push(selected);
    }
    out
}

#[test]
fn every_kind_of_commit_is_replayed_or_skipped_under_its_own_class() {
    let by_class = classify(&SelectConfig::default());
    let counts: Vec<(&str, usize)> = by_class.iter().map(|(k, v)| (*k, v.len())).collect();
    assert_eq!(
        counts,
        [
            ("binary", 1),
            ("merge", 1),
            ("mostly_generated", 1),
            ("replay", 8),
            ("root", 1),
            ("secret_in_diff", 1)
        ]
    );
}

#[test]
fn generated_paths_are_neither_asked_for_nor_shown() {
    let by_class = classify(&SelectConfig::default());
    let lock = by_class["replay"]
        .iter()
        .find_map(|s| match s {
            Selected::Replay(plan) if plan.message.starts_with("Raise the retry limit") => {
                Some(plan)
            }
            _ => None,
        })
        .unwrap();
    assert!(
        lock.changes.iter().all(|c| c.path == "config.py"),
        "{:?}",
        lock.changes
    );
    assert!(!lock.diff.contains("package-lock.json"));
    assert!(!lock.message.contains(fixture::AUTHOR) && !lock.message.contains("Signed-off-by"));
}

#[test]
fn bounds_skip_what_is_too_large_or_does_not_fit_the_prompt() {
    let small = SelectConfig {
        max_changed_lines: 3,
        ..SelectConfig::default()
    };
    let skipped = classify(&small);
    assert!(skipped.get("oversize").is_some_and(|v| !v.is_empty()));
    assert!(!skipped["replay"].is_empty() && skipped["replay"].len() < 8);

    let tight = SelectConfig {
        max_diff_tokens: 1,
        ..SelectConfig::default()
    };
    let budget = classify(&tight);
    assert_eq!(
        budget["diff_over_context_share"].len(),
        9,
        "the budget skips the seven replayable commits and the one with a secret, whose budget check comes first"
    );
    assert!(!budget.contains_key("replay"));
    assert!(matches!(
        budget["diff_over_context_share"][0],
        Selected::Skipped(_, Skip::DiffOverContextShare)
    ));
}

#[test]
fn the_diff_may_take_a_quarter_of_the_context_unless_told_otherwise() {
    assert_eq!(SelectConfig::default().max_diff_tokens, 32768 / 4);
    let wide = SelectConfig::default().with_context_share(100_000, 10);
    assert_eq!(wide.max_diff_tokens, 10_000);
    // The estimate is conservative: three characters a token.
    assert_eq!(splinter_repo_replay::select::estimate_tokens("abcdefg"), 3);
}
