// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements release gates that hold a learned model to
// what it knew before, for its clients. If your team needs expertise in
// measuring that a model has not forgotten while it learned, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec: the anchor suite the Jefferson sample ships freezes as it stands: every
//! line is a closed-book task a formal verifier grades, so the release gate's
//! retention check has general knowledge to hold a candidate to, and freezing
//! the same file again changes nothing.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::path::Path;

use common::{scratch_context, Scripted};
use splinter_pipelines::release::anchor;

#[test]
fn the_samples_anchor_suite_freezes_and_freezing_it_again_changes_nothing() {
    let file = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../samples/jefferson/anchor.jsonl");
    let (_scratch, ctx) = scratch_context("sample-anchor", Scripted::new(|_| String::new()), false);
    let first = anchor::freeze(&ctx, &file).unwrap();
    assert_eq!(first.suite.tasks.len(), 40);
    assert_eq!(first.suite.version, 1);
    let again = anchor::freeze(&ctx, &file).unwrap();
    assert_eq!((again.suite.version, again.digest), (1, first.digest));
}
