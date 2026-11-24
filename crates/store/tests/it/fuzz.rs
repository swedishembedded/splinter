// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements crash-safe, content-addressed result stores
// that never hand back a damaged product as a sound one, for its clients. If
// your team needs expertise in durable artifact storage, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec, by property: a product's manifest damaged on disk (truncated, a
//! byte changed, a span lost or repeated, a field removed or retyped) is
//! answered by every reader of the store, never a panic; and a damaged
//! manifest is never taken for the product it was: when the store still
//! reads a manifest other than the one it published, verifying the product
//! through its view refuses it.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fs;
use std::path::{Path, PathBuf};

use proptest::prelude::*;
use splinter_conformance::fuzz::{config, damaged_bytes, damaged_json};
use splinter_core::clock::FixedClock;
use splinter_core::digest::Digest;
use splinter_store::factory::products::ProductStore;

fn store(root: &Path) -> ProductStore {
    ProductStore::new(root, Box::new(FixedClock::new("2026-01-01T00:00:00.000Z")))
}

/// A store holding one published product, its id, its view and the path of
/// its manifest in the store.
struct Published {
    dir: tempfile::TempDir,
    id: Digest,
    view: PathBuf,
    manifest: PathBuf,
}

fn published() -> Published {
    let dir = tempfile::tempdir().unwrap();
    let id = Digest::sha256_of(b"rollout");
    let view = dir.path().join("rollouts/local");
    let store = store(dir.path());
    let staging = store.begin_staging().unwrap();
    for (path, body) in [("b/two.txt", "2"), ("one.txt", "1")] {
        let target = staging.path().join(path);
        fs::create_dir_all(target.parent().unwrap()).unwrap();
        fs::write(target, body).unwrap();
    }
    store.publish(&id, "rollouts", staging, &view).unwrap();
    // The store's record of the product: `<stage>.json`, outside the view.
    let record = format!("{}.json", id.hex());
    let manifest = walk(dir.path())
        .into_iter()
        .find(|p| p.ends_with(&record) && !p.starts_with(&view))
        .expect("the store keeps the product's manifest");
    Published {
        dir,
        id,
        view,
        manifest,
    }
}

fn walk(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for entry in fs::read_dir(dir).unwrap().flatten() {
        let path = entry.path();
        if path.is_dir() {
            out.extend(walk(&path));
        } else {
            out.push(path);
        }
    }
    out
}

/// Writes `bytes` as the store's manifest and reads the product back every
/// way the store offers.
fn read_back(bytes: &[u8]) -> Result<(), TestCaseError> {
    let p = published();
    let original = store(p.dir.path()).manifest(&p.id).unwrap();
    // The store may keep its manifest read-only: a damage is a write.
    let mut permissions = fs::metadata(&p.manifest).unwrap().permissions();
    #[allow(clippy::permissions_set_readonly_false)]
    permissions.set_readonly(false);
    fs::set_permissions(&p.manifest, permissions).unwrap();
    fs::write(&p.manifest, bytes).unwrap();
    let reopened = store(p.dir.path());
    let _ = reopened.list();
    if let Ok(read) = reopened.manifest(&p.id) {
        if read != original {
            prop_assert!(
                reopened.verify_product(&p.id, &p.view).is_err(),
                "a manifest other than the published one verified: {read:?}"
            );
        }
    }
    Ok(())
}

proptest! {
    // Each case is a store on disk.
    #![proptest_config(ProptestConfig { cases: 512, ..config() })]

    #[test]
    fn a_manifest_damaged_as_bytes_is_answered_and_never_verified(
        bytes in damaged_bytes(fs::read(published().manifest).unwrap())
    ) {
        read_back(&bytes)?;
    }

    #[test]
    fn a_manifest_damaged_as_a_document_is_answered_and_never_verified(
        doc in damaged_json(serde_json::from_slice(&fs::read(published().manifest).unwrap()).unwrap())
    ) {
        read_back(&serde_json::to_vec(&doc).unwrap())?;
    }
}
