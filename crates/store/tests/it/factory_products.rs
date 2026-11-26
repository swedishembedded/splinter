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
use splinter_store::factory::products::{ProductError, ProductManifest, ProductStore, Staging};

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
    let imported = target.import(&source, &id, &view).unwrap().manifest;
    assert_eq!(imported, published);
    assert_eq!(
        fs::read_to_string(view.join("batches/0/x.txt")).unwrap(),
        "x"
    );
    target.verify_product(&id, &view).unwrap();
    assert_eq!(target.list().unwrap(), vec![published.clone()]);
    // Importing again changes nothing.
    assert_eq!(
        target.import(&source, &id, &view).unwrap().manifest,
        published
    );
}

fn publish_two_files(source: &ProductStore, root: &Path, id: &Digest) -> ProductManifest {
    let staging = stage_output(source, &[("a.txt", "alpha"), ("b/c.txt", "gamma")]);
    source.publish(id, "p", staging, &root.join("p")).unwrap()
}

/// A fork shares the immutable bytes of a product with its parent: the
/// imported blobs are the parent's files, not copies, and are reported so.
#[test]
fn an_imported_product_shares_its_blobs_with_the_source_run() {
    use std::os::unix::fs::MetadataExt;
    let (source_dir, target_dir) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let source = store(source_dir.path());
    let id = stage("p");
    let published = publish_two_files(&source, source_dir.path(), &id);
    let target = store(target_dir.path());
    let view = target_dir.path().join("p");
    let imported = target.import(&source, &id, &view).unwrap();
    assert_eq!((imported.blobs_linked, imported.blobs_copied), (2, 0));
    for file in &published.files {
        let blob = |root: &Path| root.join("blobs/sha256").join(file.digest.hex());
        let (theirs, ours) = (
            fs::metadata(blob(source_dir.path())).unwrap(),
            fs::metadata(blob(target_dir.path())).unwrap(),
        );
        assert_eq!(theirs.ino(), ours.ino(), "{} is linked", file.path);
        assert!(ours.nlink() >= 3, "source blob, source view, fork blob");
    }
    target.verify_product(&id, &view).unwrap();
}

/// Where linking is impossible (another device, a file system without
/// links) the bytes are copied and verified, and the copies are counted.
#[test]
fn an_import_that_cannot_link_copies_verified_blobs_and_counts_them() {
    use std::os::unix::fs::MetadataExt;
    let (source_dir, target_dir) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let source = store(source_dir.path());
    let id = stage("p");
    let published = publish_two_files(&source, source_dir.path(), &id);
    let target = store(target_dir.path());
    let view = target_dir.path().join("p");
    let cross_device = |_: &Path, _: &Path| Err(std::io::Error::from_raw_os_error(libc::EXDEV));
    let imported = target
        .import_linking_with(&source, &id, &view, &cross_device)
        .unwrap();
    assert_eq!((imported.blobs_linked, imported.blobs_copied), (0, 2));
    assert_eq!(imported.manifest, published);
    let blob = |root: &Path| {
        root.join("blobs/sha256")
            .join(published.files[0].digest.hex())
    };
    assert_ne!(
        fs::metadata(blob(source_dir.path())).unwrap().ino(),
        fs::metadata(blob(target_dir.path())).unwrap().ino()
    );
    target.verify_product(&id, &view).unwrap();
}

/// A copy that does not hash to its address is refused, never imported.
#[test]
fn an_import_refuses_a_rotted_source_blob_it_had_to_copy() {
    let (source_dir, target_dir) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let source = store(source_dir.path());
    let id = stage("p");
    let published = publish_two_files(&source, source_dir.path(), &id);
    let blob = source_dir
        .path()
        .join("blobs/sha256")
        .join(published.files[0].digest.hex());
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(&blob, fs::Permissions::from_mode(0o644)).unwrap();
    fs::write(&blob, "rotted").unwrap();
    let target = store(target_dir.path());
    let view = target_dir.path().join("p");
    let cross_device = |_: &Path, _: &Path| Err(std::io::Error::from_raw_os_error(libc::EXDEV));
    let refused = target.import_linking_with(&source, &id, &view, &cross_device);
    assert!(matches!(refused, Err(ProductError::BlobCorrupt { .. })));
}

#[test]
fn a_view_another_product_holds_is_made_this_products_again() {
    let dir = tempfile::tempdir().unwrap();
    let store = store(dir.path());
    let view = dir.path().join("datasets/sft");
    let (old, new) = (stage("old"), stage("new"));
    let staging = stage_output(&store, &[("train.jsonl", "old\n")]);
    store.publish(&old, "datasets/sft", staging, &view).unwrap();
    let staging = stage_output(&store, &[("train.jsonl", "new\n")]);
    store.publish(&new, "datasets/sft", staging, &view).unwrap();
    assert_eq!(
        fs::read_to_string(view.join("train.jsonl")).unwrap(),
        "new\n"
    );

    // The old product is reused: its view must show it, not the newer one.
    store.ensure_view(&old, &view).unwrap();
    assert_eq!(
        fs::read_to_string(view.join("train.jsonl")).unwrap(),
        "old\n"
    );
    store.verify_product(&old, &view).unwrap();
    // A view that already shows the product is left as it is.
    store.ensure_view(&old, &view).unwrap();
}

/// A product keeps which of its files are programs: an executable file is
/// executable in every view of it (an import's too), sharing its content
/// with a plain file of the same bytes.
#[test]
fn an_executable_file_stays_executable_in_every_view() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let view = dir.path().join("compiled");
    let id = stage("compiled");
    let store = store(dir.path());
    let staging = stage_output(
        &store,
        &[("tests/test.sh", "exit 0\n"), ("same.txt", "exit 0\n")],
    );
    let script = staging.path().join("tests/test.sh");
    fs::set_permissions(&script, fs::Permissions::from_mode(0o755)).unwrap();
    let manifest = store.publish(&id, "compiled", staging, &view).unwrap();
    let executable: Vec<(&str, bool)> = manifest
        .files
        .iter()
        .map(|f| (f.path.as_str(), f.executable))
        .collect();
    assert_eq!(executable, [("same.txt", false), ("tests/test.sh", true)]);
    let mode = |path: &Path| fs::metadata(path).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode(&view.join("tests/test.sh")), 0o555);
    assert_eq!(mode(&view.join("same.txt")), 0o444);
    store.verify_product(&id, &view).unwrap();

    let other = tempfile::tempdir().unwrap();
    let imported = other.path().join("compiled");
    self::store(other.path())
        .import(&store, &id, &imported)
        .unwrap();
    assert_eq!(mode(&imported.join("tests/test.sh")), 0o555);
}

/// A product is reused only as committed: a view whose file was replaced is
/// made the product's again from its blobs, and a product whose blob changed
/// (rotted, or edited in place through a view's link to it) is refused, never
/// handed to the next stage.
#[test]
fn a_reused_product_is_its_committed_bytes_or_refused() {
    let dir = tempfile::tempdir().unwrap();
    let store = store(dir.path());
    let view = dir.path().join("datasets/sft");
    let id = stage("sft");
    let manifest = store
        .publish(
            &id,
            "datasets/sft",
            stage_output(&store, &[("train.jsonl", "kept\n")]),
            &view,
        )
        .unwrap();

    // An editor's save replaces the file (a new inode): the blob is intact
    // and the view is made the product's again.
    fs::remove_file(view.join("train.jsonl")).unwrap();
    fs::write(view.join("train.jsonl"), "edited by hand\n").unwrap();
    store.ensure_view(&id, &view).unwrap();
    assert_eq!(
        fs::read_to_string(view.join("train.jsonl")).unwrap(),
        "kept\n"
    );
    store.verify_product(&id, &view).unwrap();

    // A view's file is a link to its blob: an edit in place (which only an
    // owner overriding the read-only mode can make) reaches the blob, and the
    // product is refused rather than handed on.
    let blob = dir
        .path()
        .join("blobs/sha256")
        .join(manifest.files[0].digest.hex());
    let mut perms = fs::metadata(&blob).unwrap().permissions();
    #[allow(clippy::permissions_set_readonly_false)]
    perms.set_readonly(false);
    fs::set_permissions(&blob, perms).unwrap();
    fs::write(view.join("train.jsonl"), "rotted\n").unwrap();
    assert!(matches!(
        store.ensure_view(&id, &view),
        Err(ProductError::BlobCorrupt { .. })
    ));
}
