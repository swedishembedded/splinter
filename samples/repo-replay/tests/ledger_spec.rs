// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec: the ledger is a resumable log. A run killed in the middle of a write
//! leaves a cut-short last line; that line is not an entry, it does not stop
//! the next run from reading the ledger, and the next entry is written whole.
//! Ledgers written before the later fields existed still read.

use splinter_repo_replay::ledger::{self, Entry, Outcome};
use splinter_repo_replay::style::PlanStyle;

/// A line of a ledger written before `control`, `corrections` and the
/// supervisor fields existed.
const OLD_LINE: &str = r#"{"commit":"c0ffee","style":"terse","outcome":"passed","classes":[],"changed_lines":4,"files":1,"tool_calls":7,"seconds":12.5,"experience":null,"checks":[],"plan_words":30,"plan_code_percent":null,"requests":0,"model_seconds":0.0,"max_wait_seconds":0.0,"detail":null}"#;

#[test]
fn a_ledger_of_an_earlier_version_reads_with_the_later_fields_defaulted() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(ledger::path(dir.path()), format!("{OLD_LINE}\n")).unwrap();
    let entries = ledger::read(dir.path()).unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].outcome, Outcome::Passed);
    assert_eq!(entries[0].corrections, 0);
}

#[test]
fn a_line_cut_short_by_a_kill_is_not_an_entry_and_the_next_entry_is_written_whole() {
    let dir = tempfile::tempdir().unwrap();
    let cut = &OLD_LINE[..OLD_LINE.len() / 2];
    std::fs::write(ledger::path(dir.path()), format!("{OLD_LINE}\n{cut}")).unwrap();

    assert_eq!(ledger::read(dir.path()).unwrap().len(), 1);

    ledger::append(
        dir.path(),
        &Entry::new("beef", Some(PlanStyle::Detailed), Outcome::Failed),
    )
    .unwrap();
    let entries = ledger::read(dir.path()).unwrap();
    assert_eq!(
        entries
            .iter()
            .map(|e| e.commit.as_str())
            .collect::<Vec<_>>(),
        ["c0ffee", "beef"]
    );
}

#[test]
fn the_last_range_noted_is_the_one_a_later_command_reads() {
    let dir = tempfile::tempdir().unwrap();
    for to in ["aaa", "bbb"] {
        ledger::note_run(
            dir.path(),
            "git version 2",
            std::path::Path::new("/repo"),
            Some("root"),
            Some(to),
            &serde_json::json!({}),
        )
        .unwrap();
    }
    assert_eq!(
        ledger::last_range(dir.path()).unwrap(),
        (Some("root".into()), Some("bbb".into()))
    );
}
