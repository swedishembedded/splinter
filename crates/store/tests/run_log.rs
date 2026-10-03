// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: every command that does pipeline work records a run - what was
//! asked, each stage as it finishes, how it ended and what it produced -
//! readable from any process; a cancel is requested through the run's
//! directory and only while the run is in progress.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use serde_json::json;
use splinter_core::clock::FixedClock;
use splinter_store::experiences::StoreError;
use splinter_store::runs::{
    cancel_requested, list_runs, read_run, request_cancel, RunLog, RunStatus,
};
use splinter_store::workspace::Workspace;
use splinter_store::StateRoot;

fn scratch(name: &str) -> StateRoot {
    let path = std::env::temp_dir().join(format!("splinter-runs-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    StateRoot::new(path)
}

#[test]
fn a_run_records_its_stages_and_how_it_ended() {
    let root = scratch("record");
    let workspace = Workspace::at(&root);
    let clock = FixedClock::new("2026-09-30T08:00:00.000Z");
    let mut run = RunLog::start(
        &workspace,
        "solve",
        json!({ "task_set": "sha256:ab" }),
        &clock,
    )
    .unwrap();
    let id = run.id().to_string();
    assert_eq!(
        read_run(&workspace, &id).unwrap().status,
        RunStatus::Running
    );

    run.stage("solve", json!({ "solved": 2 }), &clock).unwrap();
    run.finish(
        RunStatus::Completed,
        json!({ "experience_set": "sha256:cd" }),
        None,
        &clock,
    )
    .unwrap();

    // Another process reads the same record.
    let other = Workspace::at(&root);
    let recorded = read_run(&other, &id).unwrap();
    assert_eq!(recorded.command, "solve");
    assert_eq!(recorded.status, RunStatus::Completed);
    assert_eq!(recorded.stages.len(), 1);
    assert_eq!(recorded.stages[0].summary["solved"], 2);
    assert_eq!(recorded.outputs["experience_set"], "sha256:cd");
    assert_eq!(list_runs(&other).unwrap().len(), 1);
    assert!(matches!(
        read_run(&other, "run-missing"),
        Err(StoreError::UnknownRun(_))
    ));
    let _ = std::fs::remove_dir_all(root.path());
}

#[test]
fn runs_started_together_keep_their_own_records() {
    let root = scratch("together");
    let handles: Vec<_> = (0..4)
        .map(|n| {
            let root = root.clone();
            let clock = FixedClock::new("2026-09-30T08:00:00.000Z");
            std::thread::spawn(move || {
                let workspace = Workspace::at(&root);
                let run =
                    RunLog::start(&workspace, &format!("cmd-{n}"), json!({}), &clock).unwrap();
                run.finish(RunStatus::Completed, json!({ "n": n }), None, &clock)
                    .unwrap()
            })
        })
        .collect();
    let finished: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let listed = list_runs(&Workspace::at(&root)).unwrap();
    assert_eq!(listed.len(), 4);
    for run in finished {
        let found = listed.iter().find(|r| r.id == run.id).unwrap();
        assert_eq!(found.command, run.command);
        assert_eq!(found.outputs, run.outputs);
    }
    let _ = std::fs::remove_dir_all(root.path());
}

#[test]
fn a_cancel_reaches_only_a_run_in_progress() {
    let root = scratch("cancel");
    let workspace = Workspace::at(&root);
    let clock = FixedClock::new("2026-09-30T08:00:00.000Z");
    let run = RunLog::start(&workspace, "learn", json!({}), &clock).unwrap();
    let id = run.id().to_string();
    assert!(!cancel_requested(&workspace, &id));
    // The request comes from another process.
    request_cancel(&Workspace::at(&root), &id).unwrap();
    assert!(cancel_requested(&workspace, &id));
    run.finish(RunStatus::Cancelled, json!({}), None, &clock)
        .unwrap();

    assert!(matches!(
        request_cancel(&workspace, &id),
        Err(StoreError::RunNotInProgress { .. })
    ));
    assert!(matches!(
        request_cancel(&workspace, "run-missing"),
        Err(StoreError::UnknownRun(_))
    ));
    let _ = std::fs::remove_dir_all(root.path());
}
