// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The stage ledger's contract: a committed stage is reused, an interrupted
//! one is resumed under a new attempt, a changed input recomputes, and a
//! corrupt output is caught.

#![allow(clippy::unwrap_used)]

use std::cell::Cell;

use serde_json::json;
use splinter_minimind::ledger::{
    KeyParts, LedgerError, OnCorrupt, StageLedger, StageWork, Standing,
};
use splinter_sdk::vocabulary::digest::Digest;

fn key(seed: u64) -> splinter_minimind::ledger::StageKey {
    KeyParts::new("generate")
        .config(&json!({ "seed": seed, "documents": 4 }))
        .input("world", &Digest::of(b"world"))
        .revision("splinter", "r1")
        .key()
}

fn write(dir: &std::path::Path, rel: &str, text: &str) {
    let path = dir.join(rel);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

#[test]
fn a_committed_stage_is_reused_without_running() {
    let dir = tempfile::tempdir().unwrap();
    let ledger = StageLedger::open(dir.path()).unwrap();
    let runs = Cell::new(0);
    let stage = |_: &_| {
        runs.set(runs.get() + 1);
        write(dir.path(), "generate/corpus.txt", "the sky is green");
        Ok::<_, String>(
            StageWork::new(json!({ "lines": 1 })).output("corpus", "generate/corpus.txt"),
        )
    };
    let first = ledger
        .run("generate", &key(7), OnCorrupt::Refuse, stage)
        .unwrap();
    assert!(!first.reused);
    drop(ledger);
    let ledger = StageLedger::open(dir.path()).unwrap();
    let second = ledger
        .run("generate", &key(7), OnCorrupt::Refuse, stage)
        .unwrap();
    assert!(second.reused);
    assert_eq!(runs.get(), 1, "a reused stage does not run");
    assert_eq!(first.commit.outputs, second.commit.outputs);
    assert_eq!(second.commit.summary, json!({ "lines": 1 }));
}

#[test]
fn an_interrupted_stage_resumes_under_a_new_attempt() {
    let dir = tempfile::tempdir().unwrap();
    let ledger = StageLedger::open(dir.path()).unwrap();
    let failed = ledger.run("explore/task-1", &key(1), OnCorrupt::Refuse, |attempt| {
        assert_eq!(attempt.number(), 1);
        assert_eq!(attempt.resumes(), None);
        Err::<StageWork, _>("killed".to_string())
    });
    assert!(matches!(failed, Err(LedgerError::Stage { .. })));
    assert!(matches!(
        ledger.standing("explore/task-1", &key(1)).unwrap(),
        Standing::Interrupted { attempt: 1 }
    ));
    let done = ledger
        .run("explore/task-1", &key(1), OnCorrupt::Refuse, |attempt| {
            assert_eq!(attempt.number(), 2, "a rerun never reuses an attempt id");
            assert_eq!(attempt.resumes(), Some(1));
            assert_ne!(attempt.id(), "explore/task-1#1");
            Ok::<_, String>(StageWork::new(json!({ "passed": true })))
        })
        .unwrap();
    assert_eq!(done.commit.attempt, 2);
    let commits = ledger.commits().unwrap();
    assert_eq!(commits.len(), 1, "only the finished attempt counts");
}

#[test]
fn a_changed_input_recomputes_and_the_old_commit_stays_findable() {
    let dir = tempfile::tempdir().unwrap();
    let ledger = StageLedger::open(dir.path()).unwrap();
    let runs = Cell::new(0);
    let stage = |_: &_| {
        runs.set(runs.get() + 1);
        Ok::<_, String>(StageWork::new(json!({ "run": runs.get() })))
    };
    ledger
        .run("generate", &key(1), OnCorrupt::Refuse, stage)
        .unwrap();
    let changed = ledger
        .run("generate", &key(2), OnCorrupt::Refuse, stage)
        .unwrap();
    assert!(!changed.reused);
    assert_eq!(runs.get(), 2);
    assert_ne!(key(1), key(2), "a changed seed is a new identity");
    let back = ledger
        .run("generate", &key(1), OnCorrupt::Refuse, stage)
        .unwrap();
    assert!(
        back.reused,
        "switching back to an earlier input reuses its commit"
    );
    assert_eq!(back.commit.summary, json!({ "run": 1 }));
}

#[test]
fn a_corrupt_output_is_refused_or_recomputed() {
    let dir = tempfile::tempdir().unwrap();
    let ledger = StageLedger::open(dir.path()).unwrap();
    let stage = |_: &_| {
        write(dir.path(), "ckpt/model.bin", "weights");
        write(dir.path(), "ckpt/config.json", "{}");
        Ok::<_, String>(StageWork::new(json!({})).output("checkpoint", "ckpt"))
    };
    ledger
        .run("pretrain", &key(3), OnCorrupt::Refuse, stage)
        .unwrap();
    write(dir.path(), "ckpt/model.bin", "flipped");
    let refused = ledger.run("pretrain", &key(3), OnCorrupt::Refuse, stage);
    match refused {
        Err(LedgerError::Corrupt { output, .. }) => assert_eq!(output, "checkpoint"),
        other => panic!("a corrupt checkpoint must be refused, got {other:?}"),
    }
    let redone = ledger
        .run("pretrain", &key(3), OnCorrupt::Recompute, stage)
        .unwrap();
    assert!(!redone.reused);
    assert_eq!(
        std::fs::read_to_string(dir.path().join("ckpt/model.bin")).unwrap(),
        "weights"
    );
    assert!(
        ledger
            .run("pretrain", &key(3), OnCorrupt::Refuse, stage)
            .unwrap()
            .reused
    );
}

#[test]
fn an_output_outside_the_run_directory_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let ledger = StageLedger::open(dir.path()).unwrap();
    let escaped = ledger.run("generate", &key(4), OnCorrupt::Refuse, |_| {
        Ok::<_, String>(StageWork::new(json!({})).output("corpus", "../elsewhere"))
    });
    assert!(matches!(escaped, Err(LedgerError::Output { .. })));
    let missing = ledger.run("generate", &key(4), OnCorrupt::Refuse, |_| {
        Ok::<_, String>(StageWork::new(json!({})).output("corpus", "never-written"))
    });
    assert!(matches!(missing, Err(LedgerError::Output { .. })));
}

#[test]
fn a_second_process_cannot_hold_the_same_run() {
    let dir = tempfile::tempdir().unwrap();
    let _held = StageLedger::open(dir.path()).unwrap();
    assert!(matches!(
        StageLedger::open(dir.path()),
        Err(LedgerError::Busy { .. })
    ));
}

#[test]
fn a_torn_last_line_is_an_interruption_not_a_corruption() {
    let dir = tempfile::tempdir().unwrap();
    {
        let ledger = StageLedger::open(dir.path()).unwrap();
        ledger
            .run("generate", &key(5), OnCorrupt::Refuse, |_| {
                Ok::<_, String>(StageWork::new(json!({})))
            })
            .unwrap();
    }
    let pack = dir.path().join("ledger.jsonl");
    let mut text = std::fs::read_to_string(&pack).unwrap();
    text.push_str("{\"record\":\"commit\",\"unit\":\"pre");
    std::fs::write(&pack, text).unwrap();
    let ledger = StageLedger::open(dir.path()).unwrap();
    assert!(matches!(
        ledger.standing("generate", &key(5)).unwrap(),
        Standing::Committed(_)
    ));
}
