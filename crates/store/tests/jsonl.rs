// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: a resumable job log is a file of one JSON record per line. A file
//! that is not there reads as empty; a line that does not parse is reported
//! with its file and number; a record written twice by runs that overlapped
//! is read once; and records appended from many threads never share a line.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use serde::{Deserialize, Serialize};
use splinter_store::jsonl::{read, read_unique, Log};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Row {
    id: u32,
    note: String,
}

fn row(id: u32, note: &str) -> Row {
    Row {
        id,
        note: note.to_string(),
    }
}

#[test]
fn a_log_that_is_not_there_reads_as_empty_and_a_bad_line_is_named() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("log.jsonl");
    assert!(read::<Row>(&path).unwrap().is_empty());
    std::fs::write(&path, "{\"id\":1,\"note\":\"a\"}\nnot json\n").unwrap();
    let message = read::<Row>(&path).unwrap_err().to_string();
    assert!(message.contains("line 2"), "{message}");
    assert!(message.contains("log.jsonl"), "{message}");
}

#[test]
fn a_record_written_twice_is_read_once_and_the_first_stands() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("log.jsonl");
    let log = Log::open(&path).unwrap();
    log.append(&row(1, "first")).unwrap();
    log.append(&row(2, "other")).unwrap();
    log.append(&row(1, "again")).unwrap();
    let rows = read_unique(&path, |r: &Row| r.id).unwrap();
    assert_eq!(rows, vec![row(1, "first"), row(2, "other")]);
    assert_eq!(read::<Row>(&path).unwrap().len(), 3, "the file keeps all");
}

#[test]
fn records_appended_from_many_threads_each_keep_their_own_line() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("log.jsonl");
    let log = Log::open(&path).unwrap();
    std::thread::scope(|scope| {
        for t in 0..8_u32 {
            let log = &log;
            scope.spawn(move || {
                for n in 0..50_u32 {
                    log.append(&row(t * 100 + n, &"x".repeat(2000))).unwrap();
                }
            });
        }
    });
    let rows = read::<Row>(&path).unwrap();
    assert_eq!(rows.len(), 400);
}

#[test]
fn a_log_reopened_appends_after_what_it_holds() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("log.jsonl");
    Log::open(&path).unwrap().append(&row(1, "a")).unwrap();
    Log::open(&path).unwrap().append(&row(2, "b")).unwrap();
    assert_eq!(read::<Row>(&path).unwrap(), vec![row(1, "a"), row(2, "b")]);
}
