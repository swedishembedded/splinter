// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements leakage-free held-out examinations of what a
// model learned from a person's writing, for its clients. If your team needs
// expertise in proving an exam shares no text with any training dataset, you
// can procure our services by sending an email to info@swedishembedded.com.

//! Spec: a `learn` that reserves its exam families up front trains on none of
//! their text, in any edition: no record of any dataset it builds holds a
//! word of a reserved family, the exam is written from the reserved text
//! alone, and the exam stage puts the candidate to it. A run that asks for
//! more families than the sources can spare is refused before anything is
//! generated.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::sync::Arc;

use common::gate::{gate_context, Brain, FakeTrainer, ANCHOR};
use common::Scripted;
use serde_json::json;
use splinter_agent::solve::{Model, MATERIAL_HEADING};
use splinter_core::model_ref::ModelRef;
use splinter_core::role::Role;
use splinter_pipelines::learn::{learn, ExamPlan, LearnRequest, Learned};
use splinter_pipelines::powered::PoweredExam;

fn words(seed: &str, n: usize) -> String {
    (0..n)
        .map(|i| format!("{seed}w{i}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// The number in the first word `lNw0` of `text` after `after`.
fn letter_in(text: &str, after: &str) -> Option<String> {
    let rest = text.split(after).nth(1)?;
    let at = rest.find("w0 ")?;
    let digits: String = rest[..at]
        .chars()
        .rev()
        .take_while(char::is_ascii_digit)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    (!digits.is_empty()).then_some(digits)
}

/// One scripted model for every role: it writes two tasks from the letter it
/// is shown, answers them as the teacher with the reference, and judges an
/// answer right when it holds the reference.
fn policy() -> Scripted {
    Scripted::new(|prompt| {
        if prompt.contains("You write training tasks") {
            let Some(digits) = letter_in(prompt, "## Body\\\\n\\\\nl") else {
                return json!({ "tasks": [] }).to_string();
            };
            let text = words(&format!("l{digits}"), 40);
            let reference = format!("l{digits}w0 l{digits}w1 l{digits}w2");
            let tasks: Vec<_> = (0..2)
                .map(|n| {
                    json!({
                        "instruction": format!("Question {n}: what does Letter {digits} say?"),
                        "subject": format!("Letter {digits}"),
                        "reference": reference,
                        "evidence": [{ "section": 1, "quote": text }]
                    })
                })
                .collect();
            json!({ "tasks": tasks }).to_string()
        } else if prompt.contains(MATERIAL_HEADING) {
            letter_in(prompt, MATERIAL_HEADING)
                .map_or("I do not know.".into(), |d| format!("l{d}w0 l{d}w1 l{d}w2"))
        } else {
            "I do not know.".into()
        }
    })
}

/// A judge that passes an answer holding the reference.
fn judge() -> Scripted {
    Scripted::new(|prompt| {
        let reference = prompt
            .split("REFERENCE:\\n")
            .nth(1)
            .and_then(|t| t.split("\\n\\nANSWER:").next())
            .unwrap_or_default();
        let answer = prompt.split("ANSWER:\\n").nth(1).unwrap_or_default();
        if !reference.is_empty() && answer.contains(reference) {
            "PASS\nit states the reference".into()
        } else {
            "FAIL\nit does not".into()
        }
    })
}

fn letters(scratch: &common::Scratch, count: usize) -> std::path::PathBuf {
    let dir = scratch.0.join("letters");
    std::fs::create_dir_all(&dir).unwrap();
    for n in 0..count {
        let body = words(&format!("l{n}"), 150);
        std::fs::write(
            dir.join(format!("letter-{n:02}.md")),
            format!("# Letter {n}\n\n## Body\n\n{body}\n"),
        )
        .unwrap();
        // Two letters are printed again in another edition, a word or two
        // different.
        if n < 2 {
            std::fs::write(
                dir.join(format!("reprint-{n:02}.md")),
                format!("# Letter {n} reprinted\n\n## Body\n\nDear sir, {body} yours\n"),
            )
            .unwrap();
        }
    }
    dir
}

#[test]
fn nothing_of_a_reserved_family_in_any_edition_reaches_a_training_dataset() {
    let (scratch, ctx) = gate_context("reserve-learn", Brain::Missing);
    ctx.add_model(
        ModelRef::policy_default(),
        Model::new(Arc::new(policy()), common::POLICY),
    );
    let judge_ref: ModelRef = "local:test/judge".parse().unwrap();
    ctx.add_model(
        judge_ref.clone(),
        Model::new(Arc::new(judge()), "scripted/judge"),
    );
    let dir = letters(&scratch, 12);
    let learned = learn(
        &ctx,
        &LearnRequest {
            sources: vec![dir.display().to_string()],
            kinds: vec!["recall".into()],
            persona: Some("The Writer".into()),
            exam: ExamPlan {
                families: Some(4),
                tasks: Some(8),
                resamples: Some(2),
            },
            rehearsal: Some(0.0),
            roles: [(Role::Judge, judge_ref)].into(),
            no_release: true,
            distill: true,
            ..LearnRequest::default()
        },
        &FakeTrainer::knowing(&[ANCHOR]),
    )
    .unwrap();
    let Learned::Ran(ran) = learned else {
        panic!("a learn that is not a dry run runs");
    };
    let report = ran.report;
    assert_eq!(report.stopped, None);
    let reserved = report.reserve.as_ref().unwrap();
    assert_eq!(reserved.families.len(), 4);

    // The words of every reserved part, in every edition of it.
    let sources = ctx.sources();
    let mut held = Vec::new();
    for id in &reserved.exam {
        for part in sources.get_source(id).unwrap().parts {
            let text = String::from_utf8(sources.read_blob(&part.content).unwrap()).unwrap();
            let key = text
                .split_whitespace()
                .find(|w| w.contains('w') && w.starts_with('l'))
                .unwrap()
                .to_string();
            held.push(key);
        }
    }
    assert!(held.len() >= 4, "{held:?}");
    // No record of the dialogue dataset or of the writer's own text holds any.
    let mut checked = 0;
    for built in [report.dataset.as_ref(), report.voice.as_ref()]
        .into_iter()
        .flatten()
    {
        let text = std::fs::read_to_string(&built.path).unwrap();
        assert!(!text.is_empty());
        for key in &held {
            let stem = key.trim_end_matches("w0");
            assert!(
                !text.contains(&format!("{stem}w")),
                "{stem} of a reserved family is in {}",
                built.path.display()
            );
        }
        checked += 1;
    }
    assert_eq!(checked, 2, "the dialogues and the writer's own text");

    // The exam is written from the reserved text alone and the exam stage
    // put the candidate to it.
    let exam = report.exam_set.as_ref().unwrap();
    assert_eq!(exam.tasks.len(), 8);
    let reserved_families: std::collections::BTreeSet<_> =
        reserved.families.iter().map(|f| &f.family).collect();
    assert!(exam
        .tasks
        .iter()
        .all(|t| reserved_families.contains(&t.family)));
    let Some(PoweredExam::Ran(powered)) = &report.powered else {
        panic!("the exam stage ran: {:?}", report.powered);
    };
    assert_eq!(
        (powered.tasks, powered.families, powered.resamples),
        (8, 4, 2)
    );
    assert!(
        powered.voice_error.is_some(),
        "the stand-in base has no weights to score the voice on"
    );
}

#[test]
fn a_run_that_asks_for_more_families_than_the_sources_can_spare_is_refused_up_front() {
    let (scratch, ctx) = gate_context("reserve-learn-refused", Brain::Missing);
    ctx.add_model(
        ModelRef::policy_default(),
        Model::new(Arc::new(policy()), common::POLICY),
    );
    let dir = letters(&scratch, 12);
    let refused = learn(
        &ctx,
        &LearnRequest {
            sources: vec![dir.display().to_string()],
            kinds: vec!["recall".into()],
            persona: Some("The Writer".into()),
            // The default is thirty families; twelve letters cannot spare it.
            no_release: true,
            distill: true,
            ..LearnRequest::default()
        },
        &FakeTrainer::knowing(&[ANCHOR]),
    );
    let error = refused.map(|_| ()).unwrap_err();
    assert!(error.is_refusal(), "{error}");
    assert!(error.to_string().contains("at most half"), "{error}");
}
