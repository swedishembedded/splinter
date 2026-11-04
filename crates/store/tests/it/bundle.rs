// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reproducible packaging of trained models
// for regulated deployments, for its clients. If your team needs expertise in
// shipping model checkpoints that can be verified byte for byte, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: a flat directory packs into one file whose bytes depend only on the
//! files' names and contents, and unpacks to exactly those files; a bundle
//! that is not one this tool wrote is refused and leaves nothing behind.
#![allow(clippy::unwrap_used)]

use std::fs;
use std::path::Path;

use splinter_store::bundle::{pack_directory, unpack_directory};

fn directory(root: &Path, name: &str, files: &[(&str, &[u8])]) -> std::path::PathBuf {
    let dir = root.join(name);
    fs::create_dir_all(&dir).unwrap();
    for (file, bytes) in files {
        fs::write(dir.join(file), bytes).unwrap();
    }
    dir
}

#[test]
fn the_same_files_make_the_same_bytes_and_unpack_to_themselves() {
    let root = tempfile::tempdir().unwrap();
    let files: [(&str, &[u8]); 2] = [("weights.safetensors", &[7u8; 1500]), ("vocab.json", b"{}")];
    let a = directory(root.path(), "a", &files);
    let b = directory(root.path(), "b", &[files[1], files[0]]);
    let first = pack_directory(&a, &root.path().join("a.bundle")).unwrap();
    let second = pack_directory(&b, &root.path().join("b.bundle")).unwrap();
    assert_eq!(first, second, "creation order and location do not matter");
    assert_eq!(first.files, vec!["vocab.json", "weights.safetensors"]);

    let names = unpack_directory(&root.path().join("a.bundle"), &root.path().join("out")).unwrap();
    assert_eq!(names, first.files);
    for (file, bytes) in files {
        assert_eq!(fs::read(root.path().join("out").join(file)).unwrap(), bytes);
    }
    assert!(
        unpack_directory(&root.path().join("a.bundle"), &root.path().join("out")).is_err(),
        "an existing destination is never written into"
    );
}

#[test]
fn what_is_not_a_flat_directory_or_a_bundle_is_refused() {
    let root = tempfile::tempdir().unwrap();
    let nested = directory(root.path(), "nested", &[("a", b"x")]);
    fs::create_dir(nested.join("sub")).unwrap();
    let empty = directory(root.path(), "empty", &[]);
    for dir in [&nested, &empty] {
        assert!(pack_directory(dir, &root.path().join("x.bundle")).is_err());
        assert!(!root.path().join("x.bundle").exists(), "no half file");
    }

    let junk = root.path().join("junk.bundle");
    fs::write(&junk, vec![1u8; 2048]).unwrap();
    let dest = root.path().join("junk-out");
    assert!(unpack_directory(&junk, &dest).is_err());
    assert!(!dest.exists(), "a refused bundle leaves nothing");

    let good = directory(root.path(), "good", &[("a", b"hello")]);
    let bundle = root.path().join("good.bundle");
    pack_directory(&good, &bundle).unwrap();
    let mut bytes = fs::read(&bundle).unwrap();
    bytes.truncate(600);
    fs::write(&bundle, bytes).unwrap();
    assert!(unpack_directory(&bundle, &root.path().join("cut")).is_err());
    assert!(!root.path().join("cut").exists());
}
