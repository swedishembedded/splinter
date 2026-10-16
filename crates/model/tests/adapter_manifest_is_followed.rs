// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements release steps that publish a fine-tuned
// adapter to a running inference server without a restart, for its clients.
// If your team needs expertise in rolling fine-tuned models out to serving,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: the manifest a release writes is the one `brain serve
//! --adapter-manifest` follows.
//!
//! The server's own follower (`ManifestFollower`) is the judge: a manifest
//! written by [`write_adapter_manifest`] yields the release it names once
//! the file hashes to the digest it states and binds to the base; a manifest
//! that names another digest is rejected and yields nothing, so the server
//! keeps what it serves; a later manifest replaces an earlier one whole; a
//! withdrawn manifest is a missing one. Splinter refuses, before writing,
//! what the follower would take for another file: a digest that is not a
//! SHA-256, and an adapter path that is not absolute.
#![allow(clippy::unwrap_used)]

use brain_testutil::adapters::{file_digest, scratch_dir, write_adapter, write_base};
use catalog::adapter_release::{ManifestFollower, ManifestPoll, ServedBase};
use splinter_core::digest::Digest;
use splinter_model::manifest::{withdraw_adapter_manifest, write_adapter_manifest};

#[test]
fn a_written_manifest_is_followed_and_a_wrong_digest_is_not() {
    let dir = scratch_dir("splinter-manifest");
    let base = dir.join("base.safetensors");
    write_base(&base, 1.0);
    let adapter = |name: &str, fill: f32| {
        let path = dir.join(name);
        write_adapter(
            &path,
            "local/base:local:chat:latest",
            "local/base",
            Some(file_digest(&base)),
            fill,
        );
        path
    };
    let (first, second) = (
        adapter("one.safetensors", 0.5),
        adapter("two.safetensors", 0.7),
    );
    let digest_of = |path: &std::path::Path| Digest::parse(&file_digest(path)).unwrap();
    let manifest = dir.join("serving").join("default.json");
    let mut follower = ManifestFollower::new(&manifest, ServedBase::new(&base, None));
    assert!(
        matches!(follower.poll(), ManifestPoll::Missing),
        "nothing is published yet"
    );

    write_adapter_manifest(&manifest, &first, &digest_of(&first)).unwrap();
    let ManifestPoll::Release(release) = follower.poll() else {
        panic!("a manifest written for an adapter bound to the base is a release");
    };
    assert_eq!(
        (
            release.release.path.as_path(),
            release.release.digest.as_str()
        ),
        (first.as_path(), digest_of(&first).as_str())
    );

    // A later release replaces the manifest whole.
    write_adapter_manifest(&manifest, &second, &digest_of(&second)).unwrap();
    let ManifestPoll::Release(release) = follower.poll() else {
        panic!("the next release is followed");
    };
    assert_eq!(release.release.path, second);

    // A manifest that states another file's digest is rejected, and says
    // which; the server keeps what it serves.
    write_adapter_manifest(&manifest, &first, &digest_of(&second)).unwrap();
    match follower.poll() {
        ManifestPoll::Rejected(why) => {
            assert!(why.contains(digest_of(&second).as_str()), "{why}");
        }
        _ => panic!("a digest mismatch is rejected"),
    }

    // Withdrawing it leaves nothing to follow.
    withdraw_adapter_manifest(&manifest).unwrap();
    assert!(matches!(follower.poll(), ManifestPoll::Missing));
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
fn what_the_follower_would_misread_is_refused_before_it_is_written() {
    let dir = scratch_dir("splinter-manifest-refused");
    let manifest = dir.join("default.json");
    let adapter = dir.join("adapter.safetensors");
    let sha = Digest::sha256_of(b"adapter");
    assert!(write_adapter_manifest(&manifest, &adapter, &Digest::of(b"adapter")).is_err());
    assert!(
        write_adapter_manifest(&manifest, std::path::Path::new("adapter.safetensors"), &sha)
            .is_err()
    );
    assert!(!manifest.exists(), "a refused manifest writes nothing");
}
