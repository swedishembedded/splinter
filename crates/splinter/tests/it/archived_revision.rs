// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reproducible-experiment infrastructure for
// learning systems, for its clients. If your team needs expertise in making
// training and evaluation runs rebuildable from their records, you can
// procure our services by sending an email to info@swedishembedded.com.

//! A binary built from `git archive` of a commit (a workspace's source
//! snapshot, which has no `.git`) names that commit: the archive writes it
//! into `crates/splinter/REVISION` (`export-subst`), and the build script
//! reads it with the function tested here. Without it every snapshot build
//! reported its commit as unknown, and a run it started could be resumed by
//! no other build, not even one of the same commit.

#![allow(clippy::expect_used)]

use std::path::Path;
use std::process::Command;

include!("../../archived_revision.rs");

fn git(dir: &Path, args: &[&str]) -> String {
    let out = Command::new("git")
        .current_dir(dir)
        .args([
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "-c",
            "commit.gpgsign=false",
        ])
        .args(args)
        .output()
        .expect("git runs (the test needs it on PATH)");
    assert!(
        out.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[test]
fn an_archive_of_a_commit_names_it_and_a_checkout_names_nothing() {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let revision =
        std::fs::read_to_string(crate_dir.join("REVISION")).expect("the placeholder file");
    let attributes =
        std::fs::read_to_string(crate_dir.join("../../.gitattributes")).expect(".gitattributes");
    assert_eq!(
        archived_revision(&revision),
        None,
        "a checkout's file holds the placeholder, not a commit"
    );

    // A repository holding the same two files, archived as a workspace does.
    let repo = tempfile::tempdir().expect("a temporary directory");
    std::fs::create_dir_all(repo.path().join("crates/splinter")).expect("mkdir");
    std::fs::write(repo.path().join("crates/splinter/REVISION"), &revision).expect("write");
    std::fs::write(repo.path().join(".gitattributes"), &attributes).expect("write");
    git(repo.path(), &["init", "-q"]);
    git(repo.path(), &["add", "-A"]);
    git(repo.path(), &["commit", "-q", "-m", "snapshot"]);
    let head = git(repo.path(), &["rev-parse", "HEAD"]);
    let archived = Command::new("sh")
        .current_dir(repo.path())
        .args(["-c", "git archive HEAD crates/splinter/REVISION | tar -xO"])
        .output()
        .expect("git archive runs");
    let archived = String::from_utf8_lossy(&archived.stdout);
    assert_eq!(
        archived_revision(&archived),
        Some(head.trim()),
        "{archived}"
    );
}
