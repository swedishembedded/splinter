// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements sandboxed execution of untrusted code for
// agents, for its clients. If your team needs expertise in running agent
// workloads without leaving state behind on shared machines, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec: a scratch directory belongs to the process that claimed it. It is
//! removed when the claim ends; one a killed process left behind is removed by
//! the next sweep, never one a live process holds, never one that is not of
//! the prefixes asked for.

#![allow(clippy::unwrap_used)]

use std::time::{Duration, SystemTime};

use splinter_sandbox::scratch::{sweep, Claim, LIVE_FILE};

fn names(root: &std::path::Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(root)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

#[test]
fn a_claim_removes_its_directory_when_dropped() {
    let root = tempfile::tempdir().unwrap();
    let claim = Claim::create(root.path(), "attempt-").unwrap();
    std::fs::create_dir_all(claim.path().join("tree/src")).unwrap();
    let path = claim.path().to_path_buf();
    assert!(path.join(LIVE_FILE).exists());
    drop(claim);
    assert!(!path.exists());
}

#[test]
fn a_sweep_removes_what_nobody_holds_and_only_what_is_ours() {
    let root = tempfile::tempdir().unwrap();
    // Left by a killed process: marker present, nobody holds its lock.
    let dead = root.path().join("attempt-dead");
    std::fs::create_dir_all(dead.join("tree")).unwrap();
    std::fs::write(dead.join(LIVE_FILE), "").unwrap();
    // Left before markers existed: none, and old.
    let ancient = root.path().join("attempt-ancient");
    std::fs::create_dir_all(&ancient).unwrap();
    std::fs::File::open(&ancient)
        .unwrap()
        .set_modified(SystemTime::now() - Duration::from_secs(24 * 3600))
        .unwrap();
    // Being created right now: no marker yet, young.
    std::fs::create_dir_all(root.path().join("attempt-young")).unwrap();
    // Held by a live claim of another prefix we sweep too.
    let live = Claim::create(root.path(), "target-").unwrap();
    // Not ours.
    std::fs::create_dir_all(root.path().join("keep-me")).unwrap();

    let removed = sweep(root.path(), &["attempt-", "target-"]).unwrap();

    assert_eq!(removed, 2);
    let live_name = live
        .path()
        .file_name()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    assert_eq!(
        names(root.path()),
        ["attempt-young", "keep-me", live_name.as_str()]
    );
}

#[test]
fn a_sweep_of_a_root_that_does_not_exist_removes_nothing() {
    let root = tempfile::tempdir().unwrap();
    assert_eq!(
        sweep(&root.path().join("absent"), &["attempt-"]).unwrap(),
        0
    );
}
