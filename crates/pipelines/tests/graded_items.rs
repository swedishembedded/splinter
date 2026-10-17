// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements auditable evaluation of language models, for
// its clients. If your team needs expertise in keeping every graded answer of
// a run so that a grader can be changed and the run regraded without asking
// the model again, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: a run report keeps every graded item, whatever its verdict.
//!
//! * An item carries the task, the model's answer, the verdict (passed,
//!   failed or undecided) and the grader's reason.
//! * An answer within the bound is kept whole.
//! * An answer over the bound is cut with an explicit marker (`truncated`,
//!   its full size), and its full text is kept in the store under the
//!   address the item names, so nothing is dropped silently.
//! * A reason over the bound is cut with the same marker.

#![allow(clippy::unwrap_used)]

mod common;

use common::{scratch_context, Scripted};
use splinter_core::experience::Task;
use splinter_pipelines::graded::{
    stored_answer, GradedItem, Verdict, ANSWER_INLINE_BYTES, REASON_BYTES,
};
use splinter_pipelines::release::probe::Probe;

fn task() -> Task {
    Task::new(
        "recall",
        Vec::new(),
        splinter_core::experience::Environment::closed_book(),
        "What is the capital of Sweden?",
        Vec::new(),
    )
    .unwrap()
}

fn context(test: &str) -> splinter_orchestrator::Context {
    let (scratch, ctx) = scratch_context(test, Scripted::new(|_| String::new()), false);
    std::mem::forget(scratch);
    ctx
}

#[test]
fn passed_failed_and_undecided_items_are_all_kept_with_answer_and_reason() {
    let ctx = context("graded-verdicts");
    let task = task();
    for (verdict, expected) in [
        (Some(true), Verdict::Passed),
        (Some(false), Verdict::Failed),
        (None, Verdict::Undecided),
    ] {
        let probe =
            Probe::graded(Some("Stockholm.".into()), verdict).with_reason("stated reference");
        let item = GradedItem::keep(&ctx, &task, &probe).unwrap();
        assert_eq!(item.task, task.task.id);
        assert_eq!(item.verdict, expected);
        let answer = item.answer.unwrap();
        assert_eq!(
            (answer.text.as_str(), answer.truncated),
            ("Stockholm.", false)
        );
        assert!(answer.stored.is_none());
        assert_eq!(item.reason.as_deref(), Some("stated reference"));
    }
    let none = GradedItem::keep(&ctx, &task, &Probe::graded(None, None)).unwrap();
    assert!(none.answer.is_none() && none.verdict == Verdict::Undecided);
}

#[test]
fn a_large_answer_is_cut_with_a_marker_and_kept_whole_in_the_store() {
    let ctx = context("graded-large");
    let long = "é".repeat(ANSWER_INLINE_BYTES);
    let probe = Probe::graded(Some(long.clone()), None);
    let item = GradedItem::keep(&ctx, &task(), &probe).unwrap();
    let answer = item.answer.unwrap();
    assert!(answer.truncated);
    assert_eq!(answer.bytes, long.len());
    assert!(answer.text.len() <= ANSWER_INLINE_BYTES);
    assert!(
        long.starts_with(&answer.text),
        "a cut at a character boundary"
    );
    let at = answer.stored.expect("the full answer is kept by address");
    assert_eq!(stored_answer(&ctx, &at).unwrap(), long);
}

#[test]
fn a_long_reason_is_cut_with_a_marker() {
    let ctx = context("graded-reason");
    let probe =
        Probe::graded(Some("x".into()), Some(false)).with_reason("r".repeat(REASON_BYTES * 2));
    let item = GradedItem::keep(&ctx, &task(), &probe).unwrap();
    let reason = item.reason.unwrap();
    assert!(reason.len() <= REASON_BYTES + 64);
    assert!(reason.ends_with("[truncated]"), "{reason}");
}
