// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: a commit never replaces a committed product. Two publishers of one
//! stage (a controller whose lease lapsed mid-stage and the one that took
//! over) can both find the stage uncommitted when they prepare; whichever
//! commits second finds the first's product and is refused when its content
//! differs, or returns the committed product when it is the same, and the
//! committed product stays as the first committed it.

#![allow(clippy::unwrap_used)]

use std::fs;

use splinter_core::clock::FixedClock;

use super::*;

fn store(root: &Path) -> ProductStore {
    ProductStore::new(root, Box::new(FixedClock::new("2026-01-01T00:00:00.000Z")))
}

fn staged(store: &ProductStore, body: &str) -> Staging {
    let staging = store.begin_staging().unwrap();
    fs::write(staging.path().join("out.txt"), body).unwrap();
    staging
}

fn prepared(store: &ProductStore, stage: &Digest, body: &str, view: &Path) -> ProductManifest {
    match store
        .prepare(stage, "p", &staged(store, body), view)
        .unwrap()
    {
        Prepared::Viewed(manifest) => manifest,
        Prepared::Existing(_) => panic!("nothing was committed yet"),
    }
}

#[test]
fn a_commit_overtaken_by_another_refuses_to_replace_it_and_the_first_stands() {
    let dir = tempfile::tempdir().unwrap();
    let stage = Digest::sha256_of(b"stage");
    let (stale, current) = (store(dir.path()), store(dir.path()));
    let late = prepared(&stale, &stage, "stale", &dir.path().join("view-a"));
    let committed = current
        .publish(
            &stage,
            "p",
            staged(&current, "current"),
            &dir.path().join("view-b"),
        )
        .unwrap();
    let refused = stale.commit(&stage, late).unwrap_err();
    assert!(
        matches!(refused, ProductError::Immutable { .. }),
        "{refused}"
    );
    assert_eq!(
        current.manifest(&stage).unwrap(),
        committed,
        "the first commit stands"
    );
}

#[test]
fn a_commit_overtaken_by_the_same_content_returns_the_committed_product() {
    let dir = tempfile::tempdir().unwrap();
    let stage = Digest::sha256_of(b"stage");
    let (late_store, first_store) = (store(dir.path()), store(dir.path()));
    let late = prepared(&late_store, &stage, "same", &dir.path().join("view-a"));
    let first = first_store
        .publish(
            &stage,
            "p",
            staged(&first_store, "same"),
            &dir.path().join("view-b"),
        )
        .unwrap();
    assert_eq!(late_store.commit(&stage, late).unwrap(), first);
}
