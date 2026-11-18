// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements curation of agent demonstrations so that a
// student model does not learn its teacher's unsafe exploration, for its
// clients. If your team needs expertise in safe distillation of coding agents,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: an sft-step dataset does not teach reading the repository's history.
//!
//! * By default an action whose tool call reads the history or git's raw
//!   storage is not supervised, counted as `repository_history`, and stays in
//!   the context of what follows.
//! * Truncating ends the demonstration before it: it and every action after it
//!   are not supervised (`repository_history`, `after_repository_history`).
//! * Keeping it learns it as before. The manifest names the policy and the mode.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use crate::common;

use common::*;
use serde_json::json;
use splinter_core::annotation::{Outcome, Strength};
use splinter_core::experience::Experience;
use splinter_data::curation::HistoryCuration;
use splinter_data::{
    manifest_path, write_dataset, Corpus, Exclusion, Manifest, Projection, RecordBody, SftStep,
    Unchecked, View,
};

const INSTRUCTION: &str = "Fix the off-by-one in the loop and report the change.";

/// A verified repair whose second action reads the history, as the teacher's
/// did: look around, read the log, edit, answer.
fn repair() -> Experience {
    let task = task("workspace", INSTRUCTION, Vec::new());
    experience(
        &task,
        trajectory(vec![
            user(1, INSTRUCTION),
            call(
                2,
                "c1",
                "shell",
                json!({"command": "git status --short && git diff --stat"}),
                "",
            ),
            call(
                3,
                "c2",
                "shell",
                json!({"command": "git log --oneline -20; git show --stat HEAD | head -3"}),
                "abc123 seed the fault",
            ),
            call(
                4,
                "c3",
                "edit",
                json!({"path": "loop.c", "text": "i < n"}),
                "ok",
            ),
            say(5, "The loop now stops before n."),
        ]),
        Some("The loop now stops before n."),
        "2026-11-18T01:00:00.000Z",
    )
}

fn project(history: Option<HistoryCuration>, packed: bool) -> Projection {
    let experience = repair();
    let mut corpus = Corpus::new();
    corpus
        .insert(
            experience.clone(),
            vec![verdict(&experience, Outcome::Pass, Strength::Formal)],
        )
        .unwrap();
    let view = SftStep::new(Strength::Formal).packed(packed);
    let view = match history {
        Some(history) => view.history(history),
        None => view,
    };
    view.project(&corpus).unwrap()
}

/// The supervised assistant turns of every record, as the commands they ran
/// (or the answer's text).
fn supervised(projection: &Projection) -> Vec<String> {
    projection
        .records
        .iter()
        .flat_map(|r| match &r.body {
            RecordBody::Chat { messages, .. } => messages
                .iter()
                .filter(|m| m.train)
                .map(|m| match m.tool_calls.first() {
                    Some(call) => call.function.arguments.clone(),
                    None => m.content.clone(),
                })
                .collect::<Vec<_>>(),
            other => panic!("a chat record, got {other:?}"),
        })
        .collect()
}

#[test]
fn by_default_a_history_read_is_not_learned_and_stays_in_the_context() {
    for packed in [false, true] {
        let projection = project(None, packed);
        let taught = supervised(&projection);
        assert_eq!(taught.len(), 3, "{taught:?}");
        assert!(taught.iter().all(|t| !t.contains("git log")), "{taught:?}");
        assert_eq!(projection.count(Exclusion::RepositoryHistory), 1);
        // The edit after it still sees what the log said.
        let RecordBody::Chat { messages, .. } = &projection.records.last().unwrap().body else {
            panic!("chat");
        };
        assert!(messages.iter().any(|m| m.content.contains("abc123")));
    }
}

#[test]
fn truncating_ends_the_demonstration_before_the_history_read() {
    let projection = project(Some(HistoryCuration::Truncate), false);
    let taught = supervised(&projection);
    assert_eq!(taught.len(), 1, "{taught:?}");
    assert!(taught[0].contains("git status"));
    assert_eq!(projection.count(Exclusion::RepositoryHistory), 1);
    assert_eq!(projection.count(Exclusion::AfterRepositoryHistory), 2);
}

#[test]
fn keeping_it_learns_it_and_the_manifest_names_the_policy() {
    let kept = project(Some(HistoryCuration::Keep), false);
    assert_eq!(supervised(&kept).len(), 4);
    assert_eq!(kept.count(Exclusion::RepositoryHistory), 0);

    let scratch = Scratch::new("history-curation");
    for (projection, recorded) in [
        (
            project(None, false),
            Some(json!({"policy": "no-repository-history", "mode": "drop-step"})),
        ),
        (
            project(Some(HistoryCuration::Truncate), false),
            Some(json!({"policy": "no-repository-history", "mode": "truncate"})),
        ),
        (kept, None),
    ] {
        let path = scratch.0.join(format!("{}.jsonl", recorded.is_some()));
        let _ = std::fs::remove_file(&path);
        write_dataset(&path, &projection, &Unchecked).unwrap();
        let manifest: Manifest =
            serde_json::from_slice(&std::fs::read(manifest_path(&path)).unwrap()).unwrap();
        assert_eq!(
            manifest.curation.map(|c| serde_json::to_value(c).unwrap()),
            recorded
        );
    }
}
