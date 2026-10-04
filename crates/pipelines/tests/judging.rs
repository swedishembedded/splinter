// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements judges that are measured before their
// verdicts count, for its clients. If your team needs expertise in
// calibrating an LLM judge, you can procure our services by sending an email
// to info@swedishembedded.com.

//! Spec: a judge is calibrated on controls made from tasks' own references,
//! by no model. Each task's reference is the right answer to it. The wrong
//! answer to it is the reference of a task from another family of sources:
//! one about the same letter may say the same thing and is not wrong. A task
//! with no other family to borrow a wrong answer from gets no wrong control,
//! so a judge is never measured on a "wrong" answer that might be right.

// Helpers outside a test function unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use common::{scratch_context, Scripted};
use splinter_core::annotation::Outcome;
use splinter_core::experience::{Environment, Privileged, PrivilegedKind, Span, Task};
use splinter_pipelines::judging::controls;
use splinter_pipelines::sources::{self, SourceTarget};

/// `i` spelled in letters, so a word carries no digit and the text is prose.
fn letters(mut i: usize) -> String {
    let mut out = String::new();
    loop {
        out.push(char::from(b'a' + u8::try_from(i % 26).unwrap()));
        i /= 26;
        if i == 0 {
            return out;
        }
    }
}

fn words(seed: &str, n: usize) -> String {
    (0..n)
        .map(|i| format!("{seed}x{}", letters(i)))
        .collect::<Vec<_>>()
        .join(" ")
}

fn span_of(
    ctx: &splinter_orchestrator::Context,
    dir: &std::path::Path,
    name: &str,
    text: &str,
) -> Span {
    let path = dir.join(name);
    std::fs::write(&path, text).unwrap();
    let id = sources::add(
        ctx,
        &SourceTarget::from_learn_arg(&path.display().to_string()).unwrap(),
    )
    .unwrap()
    .source
    .id;
    let content = ctx.sources().get_source(&id).unwrap().parts[0]
        .content
        .clone();
    Span::new(content, 0, text.len() as u64).unwrap()
}

fn task(span: &Span, question: &str, reference: &str) -> Task {
    Task::new(
        "converse",
        vec![span.clone()],
        Environment::closed_book(),
        question,
        vec![Privileged {
            kind: PrivilegedKind::Reference,
            content: reference.into(),
            span: Some(span.clone()),
        }],
    )
    .unwrap()
}

#[test]
fn the_wrong_answer_to_a_task_is_never_another_task_about_the_same_family_of_sources() {
    let (scratch, ctx) =
        scratch_context("judging-controls", Scripted::new(|_| String::new()), false);
    let letter = words("alpha", 120);
    let reprint = format!("{letter} {}", words("coda", 6));
    let other = words("omega", 120);
    let third = words("sigma", 120);
    let a1 = span_of(&ctx, &scratch.0, "a1.txt", &letter);
    let a2 = span_of(&ctx, &scratch.0, "a2.txt", &reprint);
    let b = span_of(&ctx, &scratch.0, "b.txt", &other);
    let c = span_of(&ctx, &scratch.0, "c.txt", &third);
    let tasks = [
        task(&a1, "First question?", "Reference of the first print."),
        task(&a2, "Second question?", "Reference of the second print."),
        task(&b, "Third question?", "Reference of the other letter."),
        task(&c, "Fourth question?", "Reference of the last letter."),
    ];
    let labelled = controls(&ctx, &tasks).unwrap();
    let answer = |task: &Task, label: Outcome| -> Vec<String> {
        labelled
            .iter()
            .filter(|(t, _, l)| t.task == task.task && *l == label)
            .map(|(_, exp, _)| exp.final_output.clone().unwrap())
            .collect()
    };
    for task in &tasks {
        let reference = task.privileged[0].content.clone();
        assert_eq!(
            answer(task, Outcome::Pass),
            [reference],
            "{}",
            task.instruction
        );
    }
    // The two prints are one family: neither's reference is wrong for the other.
    for wrong in answer(&tasks[0], Outcome::Fail) {
        assert_ne!(wrong, "Reference of the second print.");
        assert_ne!(wrong, "Reference of the first print.");
    }
    for wrong in answer(&tasks[1], Outcome::Fail) {
        assert_ne!(wrong, "Reference of the first print.");
    }
    assert_eq!(answer(&tasks[0], Outcome::Fail).len(), 1);
    // The controls name no model as their solver.
    assert!(labelled
        .iter()
        .all(|(_, e, _)| e.provenance.solver == "splinter/exam-controls"));
}

#[test]
fn one_family_of_sources_has_no_wrong_answer_to_borrow() {
    let (scratch, ctx) = scratch_context(
        "judging-one-family",
        Scripted::new(|_| String::new()),
        false,
    );
    let letter = words("alpha", 120);
    let span = span_of(&ctx, &scratch.0, "a.txt", &letter);
    let tasks = [
        task(&span, "First question?", "Reference one."),
        task(&span, "Second question?", "Reference two."),
    ];
    let labelled = controls(&ctx, &tasks).unwrap();
    assert!(
        labelled.iter().all(|(_, _, l)| *l == Outcome::Pass),
        "{}",
        labelled.len()
    );
    assert_eq!(labelled.len(), 2);
}
