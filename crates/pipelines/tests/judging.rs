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
use splinter_pipelines::judging::{controls, hard_controls};
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
fn the_hard_wrong_answer_is_the_reference_of_another_family_most_like_its_own() {
    let (scratch, ctx) = scratch_context("judging-hard", Scripted::new(|_| String::new()), false);
    let a = span_of(&ctx, &scratch.0, "a.txt", &words("alpha", 120));
    let a_again = span_of(&ctx, &scratch.0, "a2.txt", &words("alpha", 120));
    let b = span_of(&ctx, &scratch.0, "b.txt", &words("omega", 120));
    let c = span_of(&ctx, &scratch.0, "c.txt", &words("sigma", 120));
    let tasks = [
        task(&a, "Q1?", "Taxes on imported wine should be low."),
        task(&b, "Q2?", "Imported wine should carry modest taxes."),
        task(&c, "Q3?", "The militia is the nation's proper defence."),
        task(
            &a_again,
            "Q4?",
            "Taxes on wine from abroad ought to be low.",
        ),
    ];
    let hard = hard_controls(&ctx, &tasks).unwrap();
    let wrong_for = |q: &str| -> Vec<String> {
        hard.iter()
            .filter(|(t, _)| t.instruction == q)
            .map(|(_, e)| e.final_output.clone().unwrap())
            .collect()
    };
    // The nearest reference of another family, not the militia one and not
    // the one from the same family's other print.
    assert_eq!(
        wrong_for("Q1?"),
        ["Imported wine should carry modest taxes."]
    );
    assert_eq!(wrong_for("Q3?").len(), 1);
    assert!(hard
        .iter()
        .all(|(t, e)| { e.final_output.as_deref() != Some(t.privileged[0].content.as_str()) }));
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

/// A task set of `per_family` tasks about each of two families of sources,
/// each task's reference naming its family by a key word.
fn two_families(
    ctx: &splinter_orchestrator::Context,
    dir: &std::path::Path,
    per_family: usize,
) -> splinter_store::tasks::TaskSetId {
    let a = span_of(ctx, dir, "a.txt", &words("alpha", 100));
    let b = span_of(ctx, dir, "b.txt", &words("omega", 100));
    let mut members = Vec::new();
    let store = ctx.tasks();
    for i in 0..per_family {
        for (span, key) in [(&a, "study"), (&b, "expense")] {
            let t = task(
                span,
                &format!("How do I keep a habit of {key} (case {i})?"),
                &format!("Keep up your {key} {i} every day."),
            );
            store.put(&t).unwrap();
            members.push(splinter_store::tasks::TaskEntry {
                task: t.task.id.clone(),
                generator: None,
                prompt: None,
                variant_of: None,
                subject: None,
            });
        }
    }
    store
        .put_set(&splinter_store::tasks::TaskSet {
            name: "two families".into(),
            members,
        })
        .unwrap()
}

/// A judge that says PASS when the answer and the reference name one key
/// word, or - when it is asked about a message and a passage - when the
/// message and the passage do.
fn keyed_judge() -> Scripted {
    Scripted::new(|prompt| {
        let (left, right) = if prompt.contains("MESSAGE:") {
            (
                prompt
                    .split("MESSAGE:\\n")
                    .nth(1)
                    .and_then(|t| t.split("\\n\\nPASSAGE:").next())
                    .unwrap_or_default(),
                prompt.split("PASSAGE:\\n").nth(1).unwrap_or_default(),
            )
        } else {
            (
                prompt
                    .split("REFERENCE:\\n")
                    .nth(1)
                    .and_then(|t| t.split("\\n\\nANSWER:").next())
                    .unwrap_or_default(),
                prompt.split("ANSWER:\\n").nth(1).unwrap_or_default(),
            )
        };
        let shared = ["study", "expense"]
            .iter()
            .any(|key| left.contains(key) && right.contains(key));
        if shared { "PASS\nyes" } else { "FAIL\nno" }.to_string()
    })
}

#[test]
fn a_judge_is_measured_on_the_controls_of_a_task_set_and_its_mistakes_are_shown() {
    use splinter_agent::solve::Model;
    use splinter_pipelines::judge::measure_judge;
    use splinter_pipelines::verify::Judging;
    use std::sync::Arc;
    let (scratch, ctx) =
        scratch_context("judging-measure", Scripted::new(|_| String::new()), false);
    let set = two_families(&ctx, &scratch.0, 6);
    let good: splinter_core::model_ref::ModelRef = "local:test/good".parse().unwrap();
    let careless: splinter_core::model_ref::ModelRef = "local:test/careless".parse().unwrap();
    ctx.add_model(
        good.clone(),
        Model::new(Arc::new(keyed_judge()), "scripted/good"),
    );
    ctx.add_model(
        careless.clone(),
        Model::new(
            Arc::new(Scripted::new(|_| "PASS\nyes".into())),
            "scripted/careless",
        ),
    );
    for judging in [Judging::Reference, Judging::Fit] {
        let measured = measure_judge(&ctx, &set, &good, judging).unwrap();
        assert_eq!(measured.controls, 24, "{judging:?}");
        assert!(
            measured.trusted && measured.misjudged.is_empty(),
            "{measured:#?}"
        );
    }
    // A judge that passes everything is not trusted, and the controls it
    // passed that are wrong are listed.
    let measured = measure_judge(&ctx, &set, &careless, Judging::Reference).unwrap();
    assert!(!measured.trusted);
    assert_eq!(
        measured.calibration.precision_fail, None,
        "it never failed an answer"
    );
    assert_eq!(measured.misjudged.len(), 12, "{measured:#?}");
    assert!(measured
        .misjudged
        .iter()
        .all(|m| m.label == "fail" && m.judged == "pass"));
}
