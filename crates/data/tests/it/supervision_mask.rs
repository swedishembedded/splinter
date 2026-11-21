// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training-data pipelines that supervise only
// what the model itself decided, for its clients. If your team needs
// expertise in supervision masks for agent fine-tuning, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: a supervised dataset trains only the model's own actions. Over one
//! verified agent trajectory holding every kind of turn (the user's
//! instruction, tool calls, tool replies, a later user turn, the final
//! answer), every record of every sft-step build:
//!
//! * supervises only assistant messages: never a user turn, a tool reply or
//!   a system prompt;
//! * unpacked, supervises exactly one message, the last of its record, and the
//!   records together supervise each admitted action once;
//! * packed, supervises the same actions in one record;
//! * `--only-actions <tool>` supervises only the actions that call that tool
//!   and the final answer (an action that calls no tool), the others staying
//!   context;
//! * `--repository-history drop-step` leaves an action that reads the
//!   repository's history unsupervised (counted), in the context of what
//!   follows; `keep` supervises it; `truncate` supervises nothing from it on.
//!
//! The mask check is also run against masks broken on purpose here (a tool
//! reply or the user's turn marked trained): it must catch both.
#![allow(clippy::unwrap_used)]

use crate::common;

use common::*;
use serde_json::json;
use splinter_core::annotation::{Outcome, Strength};
use splinter_core::chat::WireMessage;
use splinter_data::curation::HistoryCuration;
use splinter_data::{Corpus, Exclusion, Projection, RecordBody, SftStep, View};

/// The trajectory: instruction, a history read, a file read, a write, a later
/// user turn, the final answer.
fn corpus() -> Corpus {
    let task = task("workspace-change", "Fix the parser.", vec![]);
    let exp = experience(
        &task,
        trajectory(vec![
            user(1, "Fix the parser."),
            call(
                2,
                "c1",
                "shell",
                json!({"command": "git log --oneline -5"}),
                "abc fix",
            ),
            call(
                3,
                "c2",
                "read_file",
                json!({"path": "src/parse.rs"}),
                "fn parse() {}",
            ),
            call(
                4,
                "c3",
                "shell",
                json!({"command": "sed -i s/a/b/ src/parse.rs"}),
                "ok",
            ),
            say(5, "The parser is fixed."),
        ]),
        Some("The parser is fixed."),
        "2026-11-21T01:00:00.000Z",
    );
    let pass = verdict(&exp, Outcome::Pass, Strength::Executable);
    let mut corpus = Corpus::new();
    corpus.insert(exp, vec![pass]).unwrap();
    corpus
}

fn messages(body: &RecordBody) -> &[WireMessage] {
    match body {
        RecordBody::Chat { messages, .. } => messages,
        other => panic!("a chat record, got {other:?}"),
    }
}

/// The supervised messages of each record, by `(role, content)`.
type Masks = Vec<Vec<WireMessage>>;

fn masks(projection: &Projection) -> Masks {
    projection
        .records
        .iter()
        .map(|r| messages(&r.body).to_vec())
        .collect()
}

/// The first record and message that breaks the mask rule: a supervised
/// message that is not the assistant's.
fn check_mask(masks: &Masks) -> Result<(), String> {
    for (r, record) in masks.iter().enumerate() {
        if let Some((m, message)) = record
            .iter()
            .enumerate()
            .find(|(_, m)| m.train && m.role != "assistant")
        {
            return Err(format!(
                "record {r} message {m} ({}) is supervised: {:?}",
                message.role, message.content
            ));
        }
    }
    Ok(())
}

/// The supervised assistant messages of all records, in order, by content
/// and tool call names.
fn supervised(masks: &Masks) -> Vec<String> {
    masks
        .iter()
        .flatten()
        .filter(|m| m.train)
        .map(|m| serde_json::to_string(m).unwrap())
        .collect()
}

fn calls(masks: &Masks, tool: &str) -> usize {
    supervised(masks)
        .iter()
        .filter(|m| m.contains(&format!("\"{tool}\"")))
        .count()
}

fn build(view: SftStep) -> Projection {
    view.project(&corpus()).unwrap()
}

#[test]
fn only_assistant_actions_are_supervised_one_per_unpacked_record() {
    let unpacked = masks(&build(SftStep::new(Strength::Executable)));
    check_mask(&unpacked).unwrap();
    for (r, record) in unpacked.iter().enumerate() {
        let trained: Vec<usize> = (0..record.len()).filter(|i| record[*i].train).collect();
        assert_eq!(
            trained,
            [record.len() - 1],
            "record {r} supervises its last message only"
        );
    }
    // The history read is dropped by default; the read, the write and the
    // answer are the actions.
    assert_eq!(unpacked.len(), 3);
    let packed = masks(&build(SftStep::new(Strength::Executable).packed(true)));
    check_mask(&packed).unwrap();
    assert_eq!(packed.len(), 1);
    assert_eq!(
        supervised(&packed),
        supervised(&unpacked),
        "packing supervises the same actions"
    );
    // Every non-assistant turn the trajectory holds is in the packed
    // record's context.
    let roles: Vec<&str> = packed[0].iter().map(|m| m.role.as_str()).collect();
    assert!(
        roles.contains(&"user") && roles.contains(&"tool"),
        "{roles:?}"
    );
}

#[test]
fn only_actions_supervises_the_named_tools_calls_and_keeps_the_rest_as_context() {
    let built = build(
        SftStep::new(Strength::Executable)
            .packed(true)
            .only_actions(vec!["read_file".into()]),
    );
    let masks = masks(&built);
    check_mask(&masks).unwrap();
    assert_eq!(calls(&masks, "read_file"), 1);
    assert_eq!(calls(&masks, "shell"), 0, "a shell call is context only");
    assert_eq!(
        supervised(&masks).len(),
        2,
        "the read and the final answer, which calls no tool"
    );
}

#[test]
fn a_history_read_is_dropped_kept_or_ends_the_demonstration_as_asked() {
    let history = |curation| {
        let built = build(
            SftStep::new(Strength::Executable)
                .packed(true)
                .history(curation),
        );
        let masks = masks(&built);
        check_mask(&masks).unwrap();
        (
            supervised(&masks)
                .iter()
                .filter(|m| m.contains("git log"))
                .count(),
            supervised(&masks).len(),
            built.count(Exclusion::RepositoryHistory),
        )
    };
    assert_eq!(history(HistoryCuration::DropStep), (0, 3, 1));
    assert_eq!(history(HistoryCuration::Keep), (1, 4, 0));
    let (history_reads, supervised_after, flagged) = history(HistoryCuration::Truncate);
    assert_eq!((history_reads, supervised_after, flagged), (0, 0, 1));
}

#[test]
fn the_mask_check_catches_a_supervised_tool_reply_or_user_turn() {
    let good = masks(&build(SftStep::new(Strength::Executable).packed(true)));
    for role in ["tool", "user"] {
        let mut broken = good.clone();
        let message = broken[0].iter_mut().find(|m| m.role == role).unwrap();
        message.train = true;
        assert!(
            check_mask(&broken).is_err(),
            "the mask check did not catch a supervised {role} message"
        );
    }
}
