// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements held-out examinations of what a model
// learned from a person's writing, for its clients. If your team needs
// expertise in measuring whether a model took on a writer's way of thinking,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: an exam puts the same held-out tasks to the base and to the
//! candidate closed-book and reports, for each, how often a judge says the
//! answer gives what the task's reference says and how often the answer states
//! a number or name the task's source does not hold, with a paired sign test
//! of the judged results. The judge is trusted only as far as it can be
//! measured: before it grades an arm it is calibrated on controls built from
//! the run's own verified answers - each against its own task (right) and
//! against another task's (wrong) - and a judge that does not tell them apart
//! grades nothing, so the report makes no claim from it.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::sync::Arc;

use common::{scratch_context, Scratch, Scripted};
use splinter_agent::solve::Model;
use splinter_campaign::exam::{exam, ExamRequest};
use splinter_campaign::model_ref::ModelRef;
use splinter_campaign::sources::{self, SourceTarget};
use splinter_core::clock::FixedClock;
use splinter_core::experience::{
    Environment, Experience, Privileged, PrivilegedKind, Provenance, Span, Task,
};
use sven_sdk::atif::{AgentProfile, Trajectory};
use sven_sdk::CancelToken;

const LETTER: &str = "# To a young man\n\n## Habits\n\nKeep habit1 and habit2 and habit3 and habit4 and habit5 and habit6 each morning, for a settled mind needs them.\n";

fn task(n: usize, span: &Span) -> Task {
    Task::new(
        "advise",
        vec![span.clone()],
        Environment::closed_book(),
        format!("What should I keep up, I wonder (question {n})?"),
        vec![Privileged {
            kind: PrivilegedKind::Reference,
            content: format!("Keep habit{n} each morning."),
            span: Some(span.clone()),
        }],
    )
    .unwrap()
}

fn verified(task: &Task) -> Experience {
    let answer = format!("Keep {} each morning.", reference_word(task));
    Experience::new(
        task.clone(),
        Trajectory::new("ATIF-v1.7", AgentProfile::new("teacher", "1")),
        Some(answer),
        Provenance::new(
            "scripted/teacher",
            &FixedClock::new("2026-10-01T00:00:00.000Z"),
        ),
    )
    .unwrap()
}

/// `habitN` of the task's reference.
fn reference_word(task: &Task) -> String {
    task.privileged
        .iter()
        .find(|p| p.kind == PrivilegedKind::Reference)
        .unwrap()
        .content
        .split_whitespace()
        .nth(1)
        .unwrap()
        .to_string()
}

/// A judge that passes an answer containing the reference's `habitN`, or
/// passes everything when `lenient`.
fn judge(lenient: bool) -> Scripted {
    Scripted::new(move |prompt| {
        let reference = prompt
            .split("REFERENCE:\\n")
            .nth(1)
            .and_then(|t| t.split("\\n\\nANSWER:").next())
            .unwrap_or_default();
        let answer = prompt.split("ANSWER:\\n").nth(1).unwrap_or_default();
        let word = reference
            .split_whitespace()
            .find(|w| w.starts_with("habit"))
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
            .unwrap_or("none");
        if lenient || answer.contains(word) {
            "PASS\nit gives the advice".into()
        } else {
            "FAIL\nit does not".into()
        }
    })
}

/// Answers each task by its number: `habitN`, with `extra` appended for the
/// tasks it names.
fn arm(knows: bool, invents_for: Option<usize>) -> Scripted {
    Scripted::new(move |prompt| {
        if !knows {
            return "I do not know.".into();
        }
        let n = (1..=6)
            .find(|n| prompt.contains(&format!("question {n})")))
            .unwrap_or(1);
        let mut answer = format!("Keep habit{n} each morning.");
        if invents_for == Some(n) {
            answer.push_str(" I did so from 1762 onward.");
        }
        answer
    })
}

fn setup(
    test: &str,
    judge: Scripted,
) -> (
    Scratch,
    splinter_campaign::Context,
    Vec<Task>,
    Vec<Experience>,
) {
    let (scratch, ctx) = scratch_context(test, Scripted::new(|_| String::new()), false);
    let path = scratch.0.join("letter.md");
    std::fs::write(&path, LETTER).unwrap();
    let id = sources::add(
        &ctx,
        &SourceTarget::from_learn_arg(&path.display().to_string()).unwrap(),
    )
    .unwrap()
    .source
    .id;
    let source = ctx.sources().get_source(&id).unwrap();
    let content = source.parts[0].content.clone();
    let span = Span::new(content, 0, LETTER.len() as u64).unwrap();
    let tasks: Vec<Task> = (1..=6).map(|n| task(n, &span)).collect();
    let controls = tasks.iter().map(verified).collect();
    for (name, script) in [
        ("base", arm(false, None)),
        ("tuned", arm(true, Some(3))),
        ("judge", judge),
    ] {
        ctx.add_model(
            format!("local:exam/{name}").parse().unwrap(),
            Model::new(Arc::new(script), format!("scripted/{name}")),
        );
    }
    (scratch, ctx, tasks, controls)
}

fn run(
    ctx: &splinter_campaign::Context,
    tasks: &[Task],
    controls: &[Experience],
) -> splinter_campaign::exam::Examined {
    let model = |name: &str| -> ModelRef { format!("local:exam/{name}").parse().unwrap() };
    exam(
        ctx,
        &ExamRequest {
            tasks,
            controls,
            base: &model("base"),
            candidate: &model("tuned"),
            judge: &model("judge"),
            cancel: CancelToken::new(),
        },
    )
    .unwrap()
}

#[test]
fn the_report_compares_the_arms_by_a_calibrated_judge_and_counts_invented_specifics() {
    let (_scratch, ctx, tasks, controls) = setup("exam-compare", judge(false));
    let report = run(&ctx, &tasks, &controls);

    assert!(report.judge.trusted, "{report:#?}");
    assert_eq!(report.tasks, 6);
    let (base, candidate) = (&report.base, &report.candidate);
    assert_eq!((base.judged_right, base.judged), (0, 6), "{base:#?}");
    assert_eq!(
        (candidate.judged_right, candidate.judged),
        (6, 6),
        "{candidate:#?}"
    );
    let paired = report.paired.as_ref().unwrap();
    assert_eq!((paired.discordant, paired.candidate_wins), (6, 6));
    assert!(paired.p_value < 0.05, "{paired:?}");
    // The vague base states nothing the letter lacks; the tuned arm invented a
    // year once, though the judge still says it gave the advice.
    assert_eq!((base.invented, base.checked), (0, 6));
    assert_eq!((candidate.invented, candidate.checked), (1, 6));
}

#[test]
fn a_judge_that_cannot_tell_right_from_wrong_grades_nothing() {
    let (_scratch, ctx, tasks, controls) = setup("exam-lenient", judge(true));
    let report = run(&ctx, &tasks, &controls);
    assert!(!report.judge.trusted, "{report:#?}");
    assert!(report.paired.is_none(), "no claim from an untrusted judge");
    assert_eq!(report.candidate.judged, 0);
    // What code measures does not depend on the judge.
    assert_eq!(report.candidate.invented, 1);
}
