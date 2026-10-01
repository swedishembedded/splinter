// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: content is stored once however often it is put, a small edit to a
//! large object stores only the chunks it touched, a Merkle tree shares every
//! untouched subtree, and corruption is an error rather than wrong bytes.
#![allow(clippy::unwrap_used)]

mod common;

use std::collections::BTreeMap;

use common::Scratch;
use splinter_expdb::blob::{BlobStore, Tree};
use splinter_expdb::ContentId;

/// Deterministic incompressible-looking bytes.
fn noise(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 24) as u8
        })
        .collect()
}

#[test]
fn a_put_object_reads_back_and_its_ref_names_its_bytes() {
    let scratch = Scratch::new();
    let mut blobs = BlobStore::open(&scratch.open()).unwrap();
    let data = noise(300_000, 1);
    let blob = blobs.put(&data).unwrap();
    assert_eq!(blob.id, ContentId::of(&data));
    assert_eq!(blob.len, 300_000);
    assert_eq!(blobs.get(&blob).unwrap(), data);
}

#[test]
fn putting_the_same_bytes_twice_stores_them_once() {
    let scratch = Scratch::new();
    let mut blobs = BlobStore::open(&scratch.open()).unwrap();
    let data = noise(500_000, 2);
    blobs.put(&data).unwrap();
    let after_first = blobs.stats();
    blobs.put(&data).unwrap();
    assert_eq!(blobs.stats().chunks_written, after_first.chunks_written);
    assert_eq!(blobs.stats().bytes_written, after_first.bytes_written);
}

#[test]
fn an_edit_inside_a_large_object_stores_only_the_chunks_it_touched() {
    let scratch = Scratch::new();
    let mut blobs = BlobStore::open(&scratch.open()).unwrap();
    let original = noise(4_000_000, 3);
    blobs.put(&original).unwrap();
    let before = blobs.stats();

    let mut edited = original.clone();
    for byte in &mut edited[2_000_000..2_004_096] {
        *byte ^= 0xff;
    }
    let blob = blobs.put(&edited).unwrap();
    let new_chunks = blobs.stats().chunks_written - before.chunks_written;

    assert!(
        new_chunks <= 4,
        "an edit of one page stored {new_chunks} chunks"
    );
    assert!(
        new_chunks * 4 < before.chunks_written,
        "most chunks must be shared"
    );
    assert_eq!(blobs.get(&blob).unwrap(), edited);
}

#[test]
fn flushed_objects_are_readable_by_a_store_opened_later() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let data = noise(900_000, 4);
    let blob = {
        let mut blobs = BlobStore::open(&db).unwrap();
        let blob = blobs.put(&data).unwrap();
        assert!(!blobs.flush().unwrap().is_empty());
        blob
    };
    let reopened = BlobStore::open(&db).unwrap();
    assert!(reopened.contains(&blob));
    assert_eq!(reopened.get(&blob).unwrap(), data);
}

#[test]
fn an_empty_object_is_a_valid_object() {
    let scratch = Scratch::new();
    let mut blobs = BlobStore::open(&scratch.open()).unwrap();
    let blob = blobs.put(b"").unwrap();
    assert_eq!(blob.len, 0);
    assert_eq!(blobs.get(&blob).unwrap(), b"");
}

#[test]
fn a_flipped_byte_in_a_pack_is_an_error_not_silently_wrong_data() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let data = noise(200_000, 5);
    let blob = {
        let mut blobs = BlobStore::open(&db).unwrap();
        let blob = blobs.put(&data).unwrap();
        blobs.flush().unwrap();
        blob
    };
    let pack = std::fs::read_dir(scratch.dir.path().join("blobs"))
        .unwrap()
        .flat_map(|d| std::fs::read_dir(d.unwrap().path()).unwrap())
        .map(|f| f.unwrap().path())
        .next()
        .unwrap();
    let mut bytes = std::fs::read(&pack).unwrap();
    bytes[1000] ^= 0x01;
    std::fs::write(&pack, bytes).unwrap();

    let reopened = BlobStore::open(&db).unwrap();
    assert!(reopened.get(&blob).is_err());
}

#[test]
fn getting_an_unknown_blob_is_not_found() {
    let scratch = Scratch::new();
    let blobs = BlobStore::open(&scratch.open()).unwrap();
    let missing = splinter_expdb::blob::BlobRef {
        id: ContentId::of(b"never stored"),
        len: 12,
    };
    assert!(blobs.get(&missing).is_err());
}

fn repository() -> BTreeMap<String, Vec<u8>> {
    let mut files = BTreeMap::new();
    for dir in 0..5u64 {
        for file in 0..10u64 {
            files.insert(
                format!("src/mod{dir}/file{file}.rs"),
                noise(3_000, dir * 100 + file),
            );
        }
    }
    files.insert(
        "Cargo.toml".into(),
        b"[package]\nname = \"demo\"\n".to_vec(),
    );
    files
}

#[test]
fn a_tree_reads_back_every_file_by_path() {
    let scratch = Scratch::new();
    let mut blobs = BlobStore::open(&scratch.open()).unwrap();
    let files = repository();
    let root = Tree::write(&mut blobs, &files).unwrap();
    for (path, content) in &files {
        assert_eq!(
            &Tree::read_file(&blobs, root, path).unwrap(),
            content,
            "{path}"
        );
    }
    let listing = Tree::files(&blobs, root).unwrap();
    assert_eq!(
        listing.keys().collect::<Vec<_>>(),
        files.keys().collect::<Vec<_>>()
    );
}

#[test]
fn identical_trees_have_the_same_root() {
    let scratch = Scratch::new();
    let mut blobs = BlobStore::open(&scratch.open()).unwrap();
    let a = Tree::write(&mut blobs, &repository()).unwrap();
    let b = Tree::write(&mut blobs, &repository()).unwrap();
    assert_eq!(a, b);
}

#[test]
fn changing_one_file_stores_that_file_and_its_ancestor_directories_only() {
    let scratch = Scratch::new();
    let mut blobs = BlobStore::open(&scratch.open()).unwrap();
    let root = Tree::write(&mut blobs, &repository()).unwrap();
    let before = blobs.stats();

    let new_root = Tree::update(&mut blobs, root, "src/mod2/file7.rs", b"fn changed() {}").unwrap();

    assert_ne!(root, new_root);
    // The new file, plus the directory objects for src/mod2, src and the root.
    let stored = blobs.stats().chunks_written - before.chunks_written;
    assert_eq!(stored, 4);
    assert_eq!(
        Tree::read_file(&blobs, new_root, "src/mod2/file7.rs").unwrap(),
        b"fn changed() {}"
    );
    // The old snapshot is untouched.
    assert_eq!(
        Tree::read_file(&blobs, root, "src/mod2/file7.rs").unwrap(),
        repository()["src/mod2/file7.rs"]
    );
}

#[test]
fn a_missing_path_in_a_tree_is_not_found() {
    let scratch = Scratch::new();
    let mut blobs = BlobStore::open(&scratch.open()).unwrap();
    let root = Tree::write(&mut blobs, &repository()).unwrap();
    assert!(Tree::read_file(&blobs, root, "src/none.rs").is_err());
}
