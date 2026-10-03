// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements model-driven task generation that turns
// any source into verifiable training tasks, for its clients. If your team
// needs expertise in synthetic task generation or verifier design, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: what a question is about is never left to guess.
//!
//! * A task set never teaches two answers to one question. The tasks stage
//!   records what each task's question is about; two tasks from different
//!   sources that ask the same question of the same subject and disagree
//!   on the answer are both left out and reported, while the same question
//!   asked of another subject is a different question and is kept.
//! * A task whose set records no subject is not reworded, since nothing
//!   holds a rewording to what it is about.
//! * A source shown with a question (`ask --open-book`) is shown under
//!   what it is: output captured from a command names the command.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use common::{scratch_context, Scripted};
use serde_json::json;
use splinter_campaign::ask::ask;
use splinter_campaign::model_ref::ModelRef;
use splinter_campaign::sources::{self, SourceTarget};
use splinter_campaign::tasks::{generate, Generation};
use splinter_campaign::variants::{generate_variants, VariantsRequest};
use splinter_core::clock::FixedClock;
use splinter_core::source::{CapturedSource, Origin, PartContent};
use splinter_store::tasks::{TaskEntry, TaskSet};
use sven_sdk::CancelToken;

/// Three manuals: two disagree about one board, the third is another board.
const MANUALS: [(&str, &str); 3] = [
    (
        "frobnicator.md",
        "# Frobnicator manual\n\n## Console\n\nThe console UART of the Frobnicator runs at 115200 baud.\n",
    ),
    (
        "frobnicator-errata.md",
        "# Frobnicator errata\n\n## Console\n\nThe console UART of the Frobnicator runs at 9600 baud.\n",
    ),
    (
        "widget.md",
        "# Widget manual\n\n## Console\n\nThe console UART of the Widget runs at 9600 baud.\n",
    ),
];

/// A generator that asks each manual for its board's console baud rate.
fn policy() -> Scripted {
    Scripted::new(|prompt| {
        if !prompt.contains("You write training tasks") {
            return String::new();
        }
        let (subject, answer) = if prompt.contains("Widget") {
            ("Widget", "9600 baud")
        } else if prompt.contains("errata") {
            ("Frobnicator", "9600 baud")
        } else {
            ("Frobnicator", "115200 baud")
        };
        json!({ "tasks": [{
            "instruction": format!("At what baud rate does the {subject} console UART run?"),
            "subject": subject,
            "reference": answer,
            "evidence": [{ "section": 1 }]
        }]})
        .to_string()
    })
}

#[test]
fn contradictory_answers_to_one_question_of_one_subject_are_left_out_of_the_set() {
    let (scratch, ctx) = scratch_context("task-set-contradiction", policy(), false);
    let mut ids = Vec::new();
    for (name, text) in MANUALS {
        let path = scratch.0.join(name);
        std::fs::write(&path, text).unwrap();
        let target = SourceTarget::from_learn_arg(&path.display().to_string()).unwrap();
        ids.push(sources::add(&ctx, &target).unwrap().source.id);
    }
    let generated = generate(
        &ctx,
        &Generation {
            sources: &ids,
            sections: &[],
            kinds: &["recall".to_string()],
            generator: &ModelRef::policy_default(),
            goal: None,
            deadline: None,
            cancel: CancelToken::new(),
        },
    )
    .unwrap();

    assert_eq!(generated.tasks, 1, "{generated:#?}");
    assert_eq!(generated.rejected.get("contradiction"), Some(&2));
    assert_eq!(generated.per_kind["recall"].admitted, 1);
    let notes: Vec<&str> = generated
        .rejections
        .iter()
        .map(|r| r.detail.as_str())
        .collect();
    assert!(
        notes
            .iter()
            .any(|n| n.contains("115200") && n.contains("9600")),
        "{notes:?}"
    );
    let set = ctx.tasks().get_set(&generated.task_set).unwrap();
    let kept = ctx.tasks().get(&set.members[0].task).unwrap();
    assert_eq!(
        kept.instruction,
        "At what baud rate does the Widget console UART run?"
    );
    assert_eq!(set.members[0].subject.as_deref(), Some("Widget"));

    // The same task recorded without its subject is not reworded.
    let unanchored = ctx
        .tasks()
        .put_set(&TaskSet {
            name: "recorded without subjects".into(),
            members: vec![TaskEntry {
                subject: None,
                ..set.members[0].clone()
            }],
        })
        .unwrap();
    let variants = generate_variants(
        &ctx,
        &VariantsRequest {
            task_set: &unanchored,
            generator: &ModelRef::policy_default(),
            per_task: 3,
            deadline: None,
            cancel: CancelToken::new(),
        },
    )
    .unwrap();
    assert_eq!(variants.tasks, 0);
    assert_eq!(variants.ineligible.get("no_subject"), Some(&1));
}

#[test]
fn an_open_book_question_is_shown_which_command_printed_its_material() {
    let policy = Scripted::new(|_| "It frobs verbosely.".into());
    let prompts = policy.prompts.clone();
    let (_scratch, ctx) = scratch_context("subjects-open-book", policy, false);
    let captured = CapturedSource::new(
        Origin::Command {
            argv: vec!["frob".into(), "--help".into()],
            cwd: "/work".into(),
            exit_code: Some(0),
            timed_out: false,
            stdout_truncated: false,
            stderr_truncated: false,
        },
        vec![PartContent {
            name: "stdout".into(),
            media_type: "text/plain".into(),
            bytes: b"-v  frob verbosely\n".to_vec(),
        }],
        &FixedClock::new("2026-09-30T08:00:00.000Z"),
    )
    .unwrap();
    let source = ctx.sources().put_source(&captured).unwrap();

    ask(
        &ctx,
        "What does -v do in frob?",
        Some(source.as_str()),
        &ModelRef::policy_default(),
    )
    .unwrap();
    let prompt = prompts.lock().unwrap().join("\n");
    assert!(prompt.contains("--- `frob --help` stdout ---"), "{prompt}");
}
