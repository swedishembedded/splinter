// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements release steps that publish a fine-tuned
// adapter to a running inference server without a restart, for its clients.
// If your team needs expertise in rolling fine-tuned models out to serving,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: the manifest a release writes is the one `brain serve
//! --adapter-manifest` reads.
//!
//! The judge is the file brain documents: one JSON object with exactly
//! `adapter` and `digest` (the server refuses any other field). A manifest
//! written by [`write_adapter_manifest`] reads back as exactly the adapter and
//! SHA-256 it was given; a later manifest replaces an earlier one
//! whole; a withdrawn manifest is a missing one and nothing is left beside it.
//! Splinter refuses, before writing, what the server would take for another
//! file: a digest that is not a SHA-256, and an adapter path that is not
//! absolute. Whether the named file hashes to the digest and binds to the
//! served base is the server's check, proven by brain's own tests.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;

use splinter_core::digest::Digest;
use splinter_model::manifest::{withdraw_adapter_manifest, write_adapter_manifest};

/// The manifest's two fields, refusing any other, as the server does.
#[derive(Debug, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct AdapterManifest {
    adapter: std::path::PathBuf,
    digest: String,
}

fn read(manifest: &Path) -> AdapterManifest {
    serde_json::from_slice(&std::fs::read(manifest).unwrap())
        .expect("the documented manifest parses what a release wrote")
}

#[test]
fn a_written_manifest_is_what_the_server_reads_and_is_replaced_whole() {
    let dir = tempfile::tempdir().unwrap();
    let (first, second) = (
        dir.path().join("one.safetensors"),
        dir.path().join("two.safetensors"),
    );
    let (first_sha, second_sha) = (Digest::sha256_of(b"one"), Digest::sha256_of(b"two"));
    let manifest = dir.path().join("serving").join("default.json");

    write_adapter_manifest(&manifest, &first, &first_sha).unwrap();
    assert_eq!(
        read(&manifest),
        AdapterManifest {
            adapter: first.clone(),
            digest: first_sha.to_string(),
        }
    );

    write_adapter_manifest(&manifest, &second, &second_sha).unwrap();
    assert_eq!(
        read(&manifest),
        AdapterManifest {
            adapter: second,
            digest: second_sha.to_string(),
        },
        "a later release replaces the manifest whole"
    );

    withdraw_adapter_manifest(&manifest).unwrap();
    assert!(!manifest.exists());
    withdraw_adapter_manifest(&manifest).unwrap();
    assert_eq!(
        std::fs::read_dir(manifest.parent().unwrap())
            .unwrap()
            .count(),
        0,
        "nothing is left beside a manifest"
    );
}

#[test]
fn what_the_server_would_misread_is_refused_before_it_is_written() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = dir.path().join("default.json");
    let adapter = dir.path().join("adapter.safetensors");
    let sha = Digest::sha256_of(b"adapter");
    assert!(write_adapter_manifest(&manifest, &adapter, &Digest::of(b"adapter")).is_err());
    assert!(write_adapter_manifest(&manifest, Path::new("adapter.safetensors"), &sha).is_err());
    assert!(!manifest.exists(), "a refused manifest writes nothing");
}
