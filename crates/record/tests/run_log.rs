// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: every command that does pipeline work records a run - what was
//! asked, each stage as it finishes, how it ended and what it produced -
//! readable from any process; a cancel is requested through the run's
//! directory and only while the run is in progress.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use serde_json::json;
use splinter_record::clock::FixedClock;
use splinter_record::experiences::StoreError;
use splinter_record::runs::{
    cancel_requested, list_runs, read_run, request_cancel, RunLog, RunStatus,
};
use splinter_record::StateRoot;

fn scratch(name: &str) -> StateRoot {
    let path = std::env::temp_dir().join(format!("splinter-runs-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    StateRoot::new(path)
}

#[test]
fn a_run_records_its_stages_and_how_it_ended() {
    let root = scratch("record");
    let clock = FixedClock::new("2026-09-30T08:00:00.000Z");
    let mut run =
        RunLog::start(&root, "solve", json!({ "task_set": "sha256:ab" }), &clock).unwrap();
    let id = run.id().to_string();
    assert_eq!(read_run(&root, &id).unwrap().status, RunStatus::Running);

    run.stage("solve", json!({ "solved": 2 }), &clock).unwrap();
    run.finish(
        RunStatus::Completed,
        json!({ "experience_set": "sha256:cd" }),
        None,
        &clock,
    )
    .unwrap();

    let recorded = read_run(&root, &id).unwrap();
    assert_eq!(recorded.command, "solve");
    assert_eq!(recorded.status, RunStatus::Completed);
    assert_eq!(recorded.stages.len(), 1);
    assert_eq!(recorded.stages[0].summary["solved"], 2);
    assert_eq!(recorded.outputs["experience_set"], "sha256:cd");
    assert_eq!(list_runs(&root).unwrap().len(), 1);
    let _ = std::fs::remove_dir_all(root.path());
}

#[test]
fn a_cancel_reaches_only_a_run_in_progress() {
    let root = scratch("cancel");
    let clock = FixedClock::new("2026-09-30T08:00:00.000Z");
    let run = RunLog::start(&root, "learn", json!({}), &clock).unwrap();
    let id = run.id().to_string();
    let dir = root.run_dir(&id);
    assert!(!cancel_requested(&dir));
    request_cancel(&root, &id).unwrap();
    assert!(cancel_requested(&dir));
    run.finish(RunStatus::Cancelled, json!({}), None, &clock)
        .unwrap();

    assert!(matches!(
        request_cancel(&root, &id),
        Err(StoreError::RunNotInProgress { .. })
    ));
    assert!(matches!(
        request_cancel(&root, "run-missing"),
        Err(StoreError::UnknownRun(_))
    ));
    let _ = std::fs::remove_dir_all(root.path());
}
