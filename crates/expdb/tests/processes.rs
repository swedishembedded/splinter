// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: separate operating-system processes share one database with no
//! coordination. Concurrent writers lose nothing, a process that dies before
//! flushing leaves nothing half-visible, and a signal one process raises is
//! seen by another.
#![allow(clippy::unwrap_used)]

mod common;

use std::path::PathBuf;
use std::process::{Child, Command};

use common::Scratch;

/// The `probe` example, built next to the test binary.
fn probe() -> PathBuf {
    let mut dir = std::env::current_exe().unwrap();
    dir.pop();
    if dir.ends_with("deps") {
        dir.pop();
    }
    let path = dir.join("examples").join("probe");
    assert!(
        path.exists(),
        "missing {path:?}: run `cargo test -p splinter-expdb`, which builds the probe example"
    );
    path
}

fn spawn(args: &[&str]) -> Child {
    Command::new(probe()).args(args).spawn().unwrap()
}

#[test]
fn many_processes_write_one_database_and_lose_nothing() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let root = scratch.dir.path().to_str().unwrap().to_owned();
    const PROCESSES: u32 = 6;
    const EACH: u64 = 25;

    let children: Vec<Child> = (0..PROCESSES)
        .map(|rank| spawn(&["write", &root, &rank.to_string(), &EACH.to_string()]))
        .collect();
    for mut child in children {
        assert!(child.wait().unwrap().success());
    }

    let found = db.snapshot().unwrap().entities("probe").unwrap();
    assert_eq!(found.len() as u64, u64::from(PROCESSES) * EACH);
}

#[test]
fn a_process_that_dies_before_flushing_leaves_nothing_visible_and_nothing_broken() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let root = scratch.dir.path().to_str().unwrap().to_owned();

    assert!(spawn(&["write", &root, "0", "5"]).wait().unwrap().success());
    let status = spawn(&["write-then-die", &root, "1", "50"]).wait().unwrap();
    assert!(!status.success(), "the probe aborts on purpose");

    let snapshot = db.snapshot().unwrap();
    assert_eq!(snapshot.entities("probe").unwrap().len(), 5);
    let report = db.gc().unwrap();
    assert_eq!(report.removed, 0, "nothing is old enough to collect");
    assert_eq!(db.snapshot().unwrap().entities("probe").unwrap().len(), 5);
}

#[test]
fn a_signal_raised_by_one_process_is_seen_by_another() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let root = scratch.dir.path().to_str().unwrap().to_owned();
    assert!(!db.signalled("cancel/run-7").unwrap());
    assert!(spawn(&["signal", &root, "cancel/run-7", "stop"])
        .wait()
        .unwrap()
        .success());
    assert!(db.signalled("cancel/run-7").unwrap());
    assert_eq!(
        db.signal_note("cancel/run-7").unwrap().as_deref(),
        Some("stop")
    );
    assert!(
        !db.signal("cancel/run-7", "later").unwrap(),
        "the first note is kept"
    );
    assert_eq!(
        db.signal_note("cancel/run-7").unwrap().as_deref(),
        Some("stop")
    );
    assert_eq!(db.signal_note("cancel/other").unwrap(), None);
    assert!(
        db.signal("../escape", "x").is_err(),
        "names cannot leave the root"
    );
}
