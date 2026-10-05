// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements release gates that hold a learned model to
// what it knew before, for its clients. If your team needs expertise in
// measuring that a model has not forgotten while it learned, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec: the anchor suites the samples ship freeze as they stand: every line
//! is a closed-book task graded against its reference by code, so the release
//! gate's retention check has general knowledge, arithmetic and
//! format-following to hold a candidate to. Several files freeze as one
//! version, and freezing the same files again changes nothing.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::path::{Path, PathBuf};

use common::{scratch_context, Scripted};
use splinter_pipelines::release::anchor;

fn sample(path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../samples")
        .join(path)
}

#[test]
fn the_samples_anchor_suite_freezes_and_freezing_it_again_changes_nothing() {
    let file = sample("jefferson/anchor.jsonl");
    let (_scratch, ctx) = scratch_context("sample-anchor", Scripted::new(|_| String::new()), false);
    let first = anchor::freeze(&ctx, std::slice::from_ref(&file)).unwrap();
    assert_eq!(first.suite.tasks.len(), 70);
    assert_eq!(first.suite.version, 1);
    let again = anchor::freeze(&ctx, &[file]).unwrap();
    assert_eq!((again.suite.version, again.digest), (1, first.digest));
}

/// The skills file holds arithmetic and format tasks, which no judge grades:
/// it freezes on its own, and with the general-knowledge file as one version
/// of all their tasks, in the order the files are given.
#[test]
fn the_skills_suite_freezes_with_the_general_one_as_a_single_version() {
    let general = sample("jefferson/anchor.jsonl");
    let skills = sample("adams/anchor-skills.jsonl");
    let (_scratch, ctx) = scratch_context(
        "sample-anchor-skills",
        Scripted::new(|_| String::new()),
        false,
    );
    let alone = anchor::freeze(&ctx, std::slice::from_ref(&skills)).unwrap();
    assert_eq!(alone.suite.tasks.len(), 40);
    let kinds: std::collections::BTreeSet<&str> = alone
        .suite
        .tasks
        .iter()
        .map(|t| t.task.kind.as_str())
        .collect();
    assert_eq!(kinds, ["arithmetic", "format"].into());
    let both = anchor::freeze(&ctx, &[general, skills]).unwrap();
    assert_eq!((both.suite.version, both.suite.tasks.len()), (2, 110));
    assert_eq!(both.suite.tasks[0].task.kind, "recall");
    assert_eq!(both.suite.tasks[109].task.kind, "format");
    assert!(
        anchor::freeze(&ctx, &[]).is_err(),
        "no file freezes nothing"
    );
}
