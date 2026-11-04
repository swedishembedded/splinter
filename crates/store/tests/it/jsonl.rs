// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: a resumable job log is a file of one JSON record per line. A file
//! that is not there reads as empty; a line that does not parse is reported
//! with its file and number; a record written twice by runs that overlapped
//! is read once; records appended from many threads never share a line; and a
//! last line cut short by a crash or a full disk is reported as torn, left out
//! of the records, and cut off before the next append so it cannot merge with
//! it.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use serde::{Deserialize, Serialize};
use splinter_store::jsonl::{read, read_unique, read_with_tail, Log};

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

#[test]
fn a_last_line_cut_short_is_reported_and_the_next_append_starts_a_new_line() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("log.jsonl");
    std::fs::write(&path, "{\"id\":1,\"note\":\"a\"}\n{\"id\":2,\"no").unwrap();

    let found = read_with_tail::<Row>(&path).unwrap();
    assert_eq!(found.records, vec![row(1, "a")]);
    let torn = found.torn.expect("the cut-short line is reported");
    assert_eq!(torn.line, 2);
    assert_eq!(read::<Row>(&path).unwrap(), vec![row(1, "a")]);

    let log = Log::open(&path).unwrap();
    assert_eq!(log.truncated().map(|t| t.line), Some(2));
    log.append(&row(3, "c")).unwrap();
    assert_eq!(read::<Row>(&path).unwrap(), vec![row(1, "a"), row(3, "c")]);
    assert!(read_with_tail::<Row>(&path).unwrap().torn.is_none());
}

#[test]
fn a_complete_last_record_that_lost_only_its_newline_is_kept() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("log.jsonl");
    std::fs::write(
        &path,
        "{\"id\":1,\"note\":\"a\"}\n{\"id\":2,\"note\":\"b\"}",
    )
    .unwrap();
    let log = Log::open(&path).unwrap();
    assert!(log.truncated().is_none());
    log.append(&row(3, "c")).unwrap();
    assert_eq!(
        read::<Row>(&path).unwrap(),
        vec![row(1, "a"), row(2, "b"), row(3, "c")]
    );
}
