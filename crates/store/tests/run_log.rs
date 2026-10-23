// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: every command that does pipeline work records a run - what was
//! asked, each stage as it finishes, how it ended and what it produced -
//! readable from any process; a cancel is requested through the run's
//! directory and only while the run is in progress.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use serde_json::json;
use splinter_core::build::{BuildInfo, Revisions};
use splinter_core::clock::FixedClock;
use splinter_core::digest::Digest;
use splinter_core::rng::Seed;
use splinter_store::experiences::StoreError;
use splinter_store::runs::{
    cancel_requested, heartbeat, list_runs, read_run, request_cancel, RunLog, RunStatus,
};
use splinter_store::workspace::Workspace;
use splinter_store::StateRoot;

fn scratch(name: &str) -> StateRoot {
    let path = std::env::temp_dir().join(format!("splinter-runs-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    StateRoot::new(path)
}

#[test]
fn a_run_that_stops_heartbeating_is_lost_not_running() {
    let root = scratch("heartbeat");
    let workspace = Workspace::at(&root);
    let clock = FixedClock::new("2026-09-30T08:00:00.000Z");
    let run = RunLog::start(&workspace, "learn", json!({}), &clock).unwrap();
    let id = run.id().to_string();

    // A run recorded before heartbeats existed keeps the status its events say.
    assert_eq!(read_run(&workspace, &id).unwrap().last_heartbeat_unix, None);
    assert_eq!(
        read_run(&workspace, &id).unwrap().status_at(1_000_000, 120),
        RunStatus::Running
    );

    heartbeat(&workspace, &id, 1_000).unwrap();
    heartbeat(&workspace, &id, 1_030).unwrap();
    let recorded = read_run(&workspace, &id).unwrap();
    assert_eq!(recorded.last_heartbeat_unix, Some(1_030));
    assert_eq!(recorded.status_at(1_060, 120), RunStatus::Running);
    assert_eq!(recorded.status_at(1_151, 120), RunStatus::Lost);

    // A finished run is not judged by its heartbeat.
    run.finish(RunStatus::Completed, json!({}), None, &clock)
        .unwrap();
    let recorded = read_run(&workspace, &id).unwrap();
    assert_eq!(recorded.status_at(9_999_999, 120), RunStatus::Completed);
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

    run.stage("solve", json!({ "solved": 2 }), None, None, &clock)
        .unwrap();
    run.stage(
        "verify",
        json!({}),
        Some(1500),
        Some("it broke".into()),
        &clock,
    )
    .unwrap();
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
    assert_eq!(recorded.stages.len(), 2);
    assert_eq!(recorded.stages[0].summary["solved"], 2);
    // A stage recorded without timing reads back without it, as every stage
    // recorded before stages were timed does.
    assert_eq!(
        (recorded.stages[0].duration_ms, &recorded.stages[0].error),
        (None, &None)
    );
    assert_eq!(recorded.stages[1].duration_ms, Some(1500));
    assert_eq!(recorded.stages[1].error.as_deref(), Some("it broke"));
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

#[test]
fn a_run_records_the_seeds_it_used_and_a_run_without_any_reads_with_none() {
    let root = scratch("seeds");
    let workspace = Workspace::at(&root);
    let clock = FixedClock::new("2026-09-30T08:00:00.000Z");
    let mut seeded = RunLog::start(&workspace, "train", json!({}), &clock).unwrap();
    // The largest seed and the smallest: both must survive the record, and 0 is a
    // value, not an absent seed.
    // seed: the boundaries of the range a record must hold
    seeded
        .seed("train", Seed::new(u64::MAX), true, &clock)
        .unwrap();
    seeded.seed("replay", Seed::new(0), false, &clock).unwrap();
    let seeded_id = seeded.id().to_string();
    // A run recorded before seeds were: only a start and a stage.
    let mut old = RunLog::start(&workspace, "solve", json!({}), &clock).unwrap();
    old.stage("solve", json!({}), None, None, &clock).unwrap();
    let old_id = old.id().to_string();

    let other = Workspace::at(&root);
    let seeds = read_run(&other, &seeded_id).unwrap().seeds;
    assert_eq!(seeds.get("train"), Some(&u64::MAX));
    assert_eq!(seeds.get("replay"), Some(&0));
    assert!(read_run(&other, &old_id).unwrap().seeds.is_empty());
    let _ = std::fs::remove_dir_all(root.path());
}

#[test]
fn a_run_names_the_build_that_made_it_and_an_old_run_reads_back_without_one() {
    let root = scratch("build");
    let workspace = Workspace::at(&root);
    let clock = FixedClock::new("2026-09-30T08:00:00.000Z");
    let binary = Digest::sha256_of(b"the splinter binary");
    let build = BuildInfo {
        revisions: Revisions {
            splinter: Some("a".repeat(40)),
            sven: Some("b".repeat(40)),
            brain: Some("c".repeat(40)),
        },
        dirty: Some(true),
        binary_digest: Some(binary.clone()),
        gpu_contexts: Some(0),
    };
    let named = RunLog::start_built(&workspace, "practice", json!({}), &build, &clock).unwrap();
    let named_id = named.id().to_string();
    // A run recorded before builds were named.
    let old = RunLog::start(&workspace, "solve", json!({}), &clock).unwrap();
    let old_id = old.id().to_string();

    let other = Workspace::at(&root);
    let read = read_run(&other, &named_id).unwrap();
    assert_eq!(read.build, build);
    assert_eq!(read.build.binary_digest, Some(binary));
    // Zero contexts is a measurement; it reads back as zero.
    assert_eq!(read.build.gpu_contexts, Some(0));

    let old = read_run(&other, &old_id).unwrap();
    assert_eq!(old.build, BuildInfo::default());
    // Unmeasured is absent from the record, not 0 or "".
    let shown = serde_json::to_value(&old).unwrap();
    for key in ["revisions", "dirty", "binary_digest", "gpu_contexts"] {
        assert!(shown.get(key).is_none(), "{key} in {shown}");
    }
    let shown = serde_json::to_value(&read).unwrap();
    assert_eq!(shown["revisions"]["sven"], "b".repeat(40));
    assert_eq!(shown["dirty"], true);
    let _ = std::fs::remove_dir_all(root.path());
}
