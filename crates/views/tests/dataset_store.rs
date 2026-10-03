// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: a dataset written under the state root is named by its manifest's
//! digest, so the same projection written twice is one dataset, a later
//! stage names it by that id, and a read refuses a dataset whose files no
//! longer match what the id promises.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use common::*;
use splinter_core::annotation::{Outcome, Strength};
use splinter_store::StateRoot;
use splinter_views::{
    Corpus, DatasetId, DatasetStore, Format, SftFinal, View, ViewError, WriteOptions,
};

fn passed_corpus() -> Corpus {
    let sum = task("arithmetic", "What is two plus two?", Vec::new());
    let pass = answered(&sum, "4", "2026-09-30T01:00:00.000Z");
    let mut corpus = Corpus::new();
    corpus
        .insert(
            pass.clone(),
            vec![verdict(&pass, Outcome::Pass, Strength::Formal)],
        )
        .unwrap();
    corpus
}

#[test]
fn a_stored_dataset_is_named_by_its_manifest_and_verified_on_read() {
    let scratch = Scratch::new("dataset-store");
    let root = StateRoot::new(&scratch.0);
    let store = DatasetStore::new(&splinter_store::workspace::Workspace::at(&root), &root);
    let projection = SftFinal::new(Strength::Formal)
        .project(&passed_corpus())
        .unwrap();

    let stored = store.put(&projection, WriteOptions::default()).unwrap();
    assert_eq!(stored.manifest.format, Format::GenericMessagesV2);
    assert_eq!(stored.manifest.counts.records, 1);
    assert!(stored.path.is_file());
    let again = store.put(&projection, WriteOptions::default()).unwrap();
    assert_eq!(again.id, stored.id, "the same dataset is stored once");
    assert_eq!(store.list().unwrap(), vec![stored.id.clone()]);
    assert_eq!(store.get(&stored.id).unwrap(), stored);

    {
        // The file is read-only; someone with the means changes it anyway.
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&stored.path, std::fs::Permissions::from_mode(0o644)).unwrap();
    }
    std::fs::write(&stored.path, "{}\n").unwrap();
    assert!(
        store.get(&stored.id).is_err(),
        "a dataset whose bytes changed is refused"
    );
    let unknown = splinter_core::digest::Digest::of(b"no such dataset");
    assert!(matches!(
        store.get(&DatasetId(unknown)),
        Err(ViewError::UnknownDataset(_))
    ));
}
