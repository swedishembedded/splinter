// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements crash-safe, content-addressed result stores
// proven by killing them at every durability boundary, for its clients. If
// your team needs expertise in durable artifact storage, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec, by enumeration (plan 10B): the product store's writes killed at
//! every durability boundary. Over one scenario (a product already
//! committed; a new one published with a shared blob, a new blob and a
//! program; another published under a view of its own; the new one's view
//! then made where the other's was), a child process is ended at each
//! boundary the scenario passes, in turn, as a crash would end it. After
//! each, the store is reopened and recovered, and:
//!
//! * the product committed before is untouched and verifies through its
//!   view;
//! * every product is committed whole or not at all: its manifest reads
//!   and every blob of it hashes to its address, or it is unknown;
//! * recovery leaves no staging directory, temporary file or intent;
//! * a view that exists is the complete view of a committed product;
//! * running the scenario again completes, and ends where a run never
//!   interrupted ends.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use splinter_core::clock::FixedClock;
use splinter_core::digest::Digest;
use splinter_store::factory::products::{ProductError, ProductManifest, ProductStore};
use splinter_store::faults::injection;

const PROBE: &str = "crash::probe";
const DIR_ARG: &str = "crash-probe-dir=";
const AT_ARG: &str = "crash-probe-at=";

fn store(root: &Path) -> ProductStore {
    ProductStore::new(root, Box::new(FixedClock::new("2026-01-01T00:00:00.000Z")))
}

fn stage(label: &str) -> Digest {
    Digest::sha256_of(label.as_bytes())
}

fn publish(store: &ProductStore, id: &Digest, view: &Path, files: &[(&str, &str, bool)]) {
    let staging = store.begin_staging().unwrap();
    for (path, body, executable) in files {
        let target = staging.path().join(path);
        fs::create_dir_all(target.parent().unwrap()).unwrap();
        fs::write(&target, body).unwrap();
        if *executable {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&target, fs::Permissions::from_mode(0o755)).unwrap();
        }
    }
    store.publish(id, "rollouts", staging, view).unwrap();
}

/// The product committed before the scenario.
fn before(root: &Path) {
    publish(
        &store(root),
        &stage("a"),
        &root.join("rollouts/a"),
        &[("two.txt", "2", false), ("one.txt", "1", false)],
    );
}

/// The writes the crash interrupts.
fn scenario(root: &Path) {
    let store = store(root);
    publish(
        &store,
        &stage("b"),
        &root.join("rollouts/b"),
        &[
            ("shared/two.txt", "2", false),
            ("new.txt", "fresh", false),
            ("run.sh", "#!/bin/sh\n", true),
        ],
    );
    publish(
        &store,
        &stage("c"),
        &root.join("rollouts/shared"),
        &[("c.txt", "c", false)],
    );
    // A fork of the run that holds both: b's view where c's was.
    store
        .ensure_view(&stage("b"), &root.join("rollouts/shared"))
        .unwrap();
}

/// Run as a child: crash at the boundary the arguments name.
#[test]
#[ignore = "a child process of the crash enumeration, run by it"]
fn probe() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |prefix: &str| {
        args.iter()
            .find_map(|a| a.strip_prefix(prefix))
            .unwrap_or_else(|| panic!("the crash enumeration passes {prefix}"))
            .to_owned()
    };
    let root = PathBuf::from(arg(DIR_ARG));
    injection::arm(arg(AT_ARG).parse().unwrap());
    scenario(&root);
}

fn committed(store: &ProductStore, label: &str) -> Option<ProductManifest> {
    match store.manifest(&stage(label)) {
        Ok(manifest) => {
            for file in &manifest.files {
                store
                    .verify(&file.digest)
                    .unwrap_or_else(|e| panic!("product {label} committed with a bad blob: {e}"));
            }
            Some(manifest)
        }
        Err(ProductError::UnknownProduct(_)) => None,
        Err(other) => panic!("product {label} is neither committed nor unknown: {other}"),
    }
}

/// Everything under `root` whose name says it is temporary or pending.
fn leftovers(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir).unwrap().flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().into_owned();
            let in_staging = path
                .strip_prefix(root)
                .is_ok_and(|p| p.components().count() == 2 && p.starts_with("staging"));
            if name.contains(".tmp-") || name.ends_with(".partial") || in_staging {
                out.push(path.clone());
            }
            if path.is_dir() && !path.is_symlink() {
                stack.push(path);
            }
        }
    }
    out.extend(
        fs::read_dir(root.join("pending"))
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path()),
    );
    out
}

/// The state of `root` that must hold after any crash and recovery.
fn check_recovered(root: &Path, crash: &str) {
    let store = store(root);
    store.recover().unwrap();
    let a =
        committed(&store, "a").unwrap_or_else(|| panic!("{crash}: the earlier product is gone"));
    assert_eq!(a.files.len(), 2, "{crash}");
    store
        .verify_product(&stage("a"), &root.join("rollouts/a"))
        .unwrap_or_else(|e| panic!("{crash}: the earlier product's view: {e}"));
    let products: Vec<&str> = ["b", "c"]
        .into_iter()
        .filter(|l| committed(&store, l).is_some())
        .collect();
    assert_eq!(
        leftovers(root),
        Vec::<PathBuf>::new(),
        "{crash}: recovery left these"
    );
    for view in ["rollouts/b", "rollouts/shared"] {
        let path = root.join(view);
        if path.exists() {
            assert!(
                products
                    .iter()
                    .any(|l| store.verify_product(&stage(l), &path).is_ok()),
                "{crash}: {view} exists but is no committed product's complete view"
            );
        }
    }
}

/// The manifests and views a run ends with.
fn outcome(root: &Path) -> Vec<String> {
    let store = store(root);
    let mut out = Vec::new();
    for label in ["a", "b", "c"] {
        out.push(format!("{label}: {:?}", committed(&store, label)));
    }
    for (label, view) in [
        ("a", "rollouts/a"),
        ("b", "rollouts/b"),
        ("b", "rollouts/shared"),
    ] {
        out.push(format!(
            "{view} is {label}'s: {:?}",
            store
                .verify_product(&stage(label), &root.join(view))
                .map_err(|e| e.to_string())
        ));
    }
    out
}

#[test]
fn a_crash_at_any_durability_boundary_loses_nothing_committed_and_shows_nothing_partial() {
    // The boundaries the scenario passes, and where it ends uninterrupted.
    let reference = tempfile::tempdir().unwrap();
    before(reference.path());
    injection::reset();
    scenario(reference.path());
    let boundaries = injection::seen();
    let expected = outcome(reference.path());
    assert!(
        boundaries.len() > 20,
        "the scenario passes {} boundaries",
        boundaries.len()
    );
    for name in [
        "blob: synced",
        "view: placed",
        "atomic write: renamed",
        "publish: intent removed",
        "view: old view moved aside",
    ] {
        assert!(
            boundaries.contains(&name),
            "the scenario never reaches {name:?}"
        );
    }

    for (at, name) in boundaries.iter().enumerate() {
        let crash = format!("crashed at boundary {at} ({name})");
        let dir = tempfile::tempdir().unwrap();
        before(dir.path());
        let status = Command::new(std::env::current_exe().unwrap())
            .args(["--ignored", "--exact", PROBE, "--test-threads=1", "--quiet"])
            .arg(format!("{DIR_ARG}{}", dir.path().display()))
            .arg(format!("{AT_ARG}{at}"))
            .output()
            .unwrap();
        assert_eq!(
            status.status.code(),
            Some(injection::CRASHED),
            "{crash}: the child did not end there: {}",
            String::from_utf8_lossy(&status.stdout)
        );
        check_recovered(dir.path(), &crash);
        // Running it again completes and ends where an uninterrupted run does.
        injection::reset();
        scenario(dir.path());
        assert_eq!(outcome(dir.path()), expected, "{crash}: resumed elsewhere");
    }
}
