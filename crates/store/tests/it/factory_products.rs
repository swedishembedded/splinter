// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements crash-safe, content-addressed result stores
// for long-running automation, for its clients. If your team needs expertise
// in durable artifact storage, you can procure our services by sending an
// email to info@swedishembedded.com.

//! The factory's immutable product store: commit point, immutability,
//! crash recovery and corruption detection.

#![allow(clippy::unwrap_used)]

use std::fs;
use std::path::Path;

use splinter_core::clock::FixedClock;
use splinter_core::digest::Digest;
use splinter_store::factory::products::{ProductError, ProductStore, Staging};

fn store(root: &Path) -> ProductStore {
    ProductStore::new(root, Box::new(FixedClock::new("2026-01-01T00:00:00.000Z")))
}

fn stage(label: &str) -> Digest {
    Digest::sha256_of(label.as_bytes())
}

fn stage_output(store: &ProductStore, files: &[(&str, &str)]) -> Staging {
    let staging = store.begin_staging().unwrap();
    for (path, body) in files {
        let target = staging.path().join(path);
        fs::create_dir_all(target.parent().unwrap()).unwrap();
        fs::write(target, body).unwrap();
    }
    staging
}

#[test]
fn a_committed_product_survives_reopen_and_is_readable_through_its_view() {
    let dir = tempfile::tempdir().unwrap();
    let view = dir.path().join("rollouts/local");
    let id = stage("rollout");
    {
        let store = store(dir.path());
        let staging = stage_output(&store, &[("b/two.txt", "2"), ("one.txt", "1")]);
        let manifest = store.publish(&id, "rollouts", staging, &view).unwrap();
        assert_eq!(manifest.schema, "splinter.product/1");
        let paths: Vec<_> = manifest.files.iter().map(|f| f.path.as_str()).collect();
        assert_eq!(paths, ["b/two.txt", "one.txt"], "sorted by path");
    }
    let reopened = store(dir.path());
    let manifest = reopened.manifest(&id).unwrap();
    assert_eq!(manifest.name, "rollouts");
    assert_eq!(manifest.created_at, "2026-01-01T00:00:00.000Z");
    assert_eq!(fs::read_to_string(view.join("b/two.txt")).unwrap(), "2");
    assert!(view.join("manifest.json").is_file());
    reopened.verify_product(&id, &view).unwrap();
    let blob = reopened.open(&manifest.files[1].digest).unwrap();
    assert_eq!(blob.metadata().unwrap().len(), 1);
}

#[test]
fn a_crash_before_publish_leaves_no_product_and_recover_cleans_up() {
    let dir = tempfile::tempdir().unwrap();
    let store = store(dir.path());
    let staging = stage_output(&store, &[("out.txt", "partial")]);
    let leftover = staging.path().to_path_buf();
    drop(staging); // the process died before it published
    assert!(leftover.exists());
    assert!(matches!(
        store.manifest(&stage("never")),
        Err(ProductError::UnknownProduct(_))
    ));
    store.recover().unwrap();
    assert!(!leftover.exists());
}

#[test]
fn recover_removes_a_view_whose_product_was_never_committed() {
    let dir = tempfile::tempdir().unwrap();
    let store = store(dir.path());
    let view = dir.path().join("rollouts/local");
    let id = stage("crashy");
    let staging = stage_output(&store, &[("x", "1")]);
    // The process died after the view was in place and before the commit.
    store
        .publish_without_commit(&id, "n", staging, &view)
        .unwrap();
    assert!(view.exists());
    assert!(store.manifest(&id).is_err());
    store.recover().unwrap();
    assert!(!view.exists());
}

#[test]
fn republishing_identical_content_is_a_no_op_and_different_content_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let store = store(dir.path());
    let view = dir.path().join("v");
    let id = stage("same");
    let first = store
        .publish(&id, "n", stage_output(&store, &[("a", "1")]), &view)
        .unwrap();
    let again = store
        .publish(&id, "n", stage_output(&store, &[("a", "1")]), &view)
        .unwrap();
    assert_eq!(first, again);
    let refused = store.publish(&id, "n", stage_output(&store, &[("a", "2")]), &view);
    assert!(matches!(refused, Err(ProductError::Immutable { .. })));
    assert_eq!(store.manifest(&id).unwrap(), first);
    assert_eq!(fs::read_to_string(view.join("a")).unwrap(), "1");
}

#[test]
fn a_corrupted_blob_fails_verification() {
    let dir = tempfile::tempdir().unwrap();
    let store = store(dir.path());
    let view = dir.path().join("v");
    let id = stage("rot");
    let manifest = store
        .publish(&id, "n", stage_output(&store, &[("a", "payload")]), &view)
        .unwrap();
    let digest = &manifest.files[0].digest;
    store.verify(digest).unwrap();
    let blob = dir.path().join("blobs/sha256").join(digest.hex());
    let mut perms = fs::metadata(&blob).unwrap().permissions();
    #[allow(clippy::permissions_set_readonly_false)]
    perms.set_readonly(false);
    fs::set_permissions(&blob, perms).unwrap();
    fs::write(&blob, "tampered").unwrap();
    assert!(matches!(
        store.verify(digest),
        Err(ProductError::BlobCorrupt { .. })
    ));
    assert!(store.verify_product(&id, &view).is_err());
}

#[test]
fn blobs_are_stored_once_by_sha256_address() {
    let dir = tempfile::tempdir().unwrap();
    let store = store(dir.path());
    let first = store.put_bytes(b"hello").unwrap();
    assert_eq!(first, Digest::sha256_of(b"hello"));
    assert_eq!(store.put_bytes(b"hello").unwrap(), first);
    let file = dir.path().join("f");
    fs::write(&file, "hello").unwrap();
    assert_eq!(store.put_file(&file).unwrap(), first);
    assert!(dir.path().join("blobs/sha256").join(first.hex()).is_file());
}

#[test]
fn a_product_imported_into_another_run_is_the_same_product_and_is_listed() {
    let source_dir = tempfile::tempdir().unwrap();
    let target_dir = tempfile::tempdir().unwrap();
    let source = store(source_dir.path());
    let id = stage("rollout");
    let staging = stage_output(
        &source,
        &[("trials.jsonl", "a\n"), ("batches/0/x.txt", "x")],
    );
    let published = source
        .publish(
            &id,
            "rollouts/local",
            staging,
            &source_dir.path().join("rollouts/local"),
        )
        .unwrap();

    let target = store(target_dir.path());
    let view = target_dir.path().join("rollouts/local");
    let imported = target.import(&source, &id, &view).unwrap();
    assert_eq!(imported, published);
    assert_eq!(
        fs::read_to_string(view.join("batches/0/x.txt")).unwrap(),
        "x"
    );
    target.verify_product(&id, &view).unwrap();
    assert_eq!(target.list().unwrap(), vec![published.clone()]);
    // Importing again changes nothing.
    assert_eq!(target.import(&source, &id, &view).unwrap(), published);
}
