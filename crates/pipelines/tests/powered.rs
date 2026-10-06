// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements held-out examinations of what a model
// learned from a person's writing, for its clients. If your team needs
// expertise in measuring a fine-tune with enough power to tell a gain from
// luck, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: the powered exam puts a candidate, its base and the base told what
//! the candidate is to be to a frozen exam, keeps every answer and verdict of
//! every arm, compares the arms per task with the family as the unit of
//! evidence, and leaves out a family the candidate was trained on.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::sync::Arc;

use common::gate::{candidate_on, dataset_citing, gate_context, Brain, ANCHOR, FACTS};
use common::Scripted;
use serde_json::json;
use splinter_agent::solve::Model;
use splinter_agent::CancelToken;
use splinter_core::digest::Digest;
use splinter_core::model_ref::ModelRef;
use splinter_orchestrator::Context;
use splinter_pipelines::exam_set::{create, ExamSet, ExamSetRequest};
use splinter_pipelines::powered::{
    run, Powered, PoweredRequest, BASE, CANDIDATE, PERSONA, PROMPTED,
};
use splinter_pipelines::release::arm;
use splinter_pipelines::reserve::{reserve, ReserveRequest};
use splinter_pipelines::sources::{add, SourceTarget};

const PERSONA_PROMPT: &str = "You are a surveyor of the old school. Answer as one.";

fn words(seed: &str, n: usize) -> String {
    (0..n)
        .map(|i| format!("{seed}w{i}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// The number after `marker` in `text`.
fn number_after(text: &str, marker: &str) -> Option<String> {
    let rest = text.split(marker).nth(1)?;
    Some(rest.chars().take_while(char::is_ascii_digit).collect())
}

/// A frozen exam over twelve letters with six reserved, two tasks a family,
/// and the reserved parts' contents by family.
fn exam(ctx: &Context, scratch: &common::Scratch) -> ExamSet {
    let generator = Scripted::new(|prompt| {
        if !prompt.contains("You write training tasks") {
            return String::new();
        }
        let digits = number_after(prompt, "## Body\\\\n\\\\nl").unwrap();
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
    });
    let generator_ref: ModelRef = "local:test/generator".parse().unwrap();
    ctx.add_model(
        generator_ref.clone(),
        Model::new(Arc::new(generator), "scripted/generator"),
    );
    let dir = scratch.0.join("letters");
    std::fs::create_dir_all(&dir).unwrap();
    for n in 0..12 {
        std::fs::write(
            dir.join(format!("letter-{n:02}.md")),
            format!(
                "# Letter {n}\n\n## Body\n\n{}\n",
                words(&format!("l{n}"), 200)
            ),
        )
        .unwrap();
    }
    let id = add(ctx, &SourceTarget::Path { path: dir })
        .unwrap()
        .source
        .id;
    let reservation = reserve(
        ctx,
        &ReserveRequest {
            sources: &[id],
            families: 6,
            seed: 1,
            touched_by: &[],
        },
    )
    .unwrap();
    create(
        ctx,
        &ExamSetRequest {
            reservation: &reservation,
            kinds: &["recall".to_string()],
            generator: &generator_ref,
            goal: None,
            author: None,
            max_tasks: 12,
            cancel: CancelToken::new(),
        },
    )
    .unwrap()
}

/// Registers the arms: the candidate gives each question's reference, the
/// base does not; and a judge that passes an answer holding the reference.
fn arms(ctx: &Context, candidate_adapter: &std::path::Path) -> ModelRef {
    let knows = |gives: bool| {
        Scripted::new(move |prompt| match number_after(prompt, "Letter ") {
            Some(n) if gives => format!("l{n}w0 l{n}w1 l{n}w2"),
            _ => "I am not sure.".into(),
        })
    };
    ctx.add_model(
        arm(ctx.config(), None),
        Model::new(Arc::new(knows(false)), "scripted/base"),
    );
    ctx.add_model(
        arm(ctx.config(), Some(candidate_adapter)),
        Model::new(Arc::new(knows(true)), "scripted/tuned"),
    );
    let judge = Scripted::new(|prompt| {
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
    });
    let judge_ref: ModelRef = "local:test/judge".parse().unwrap();
    ctx.add_model(
        judge_ref.clone(),
        Model::new(Arc::new(judge), "scripted/judge"),
    );
    judge_ref
}

fn examined(ctx: &Context, exam: &ExamSet, candidate: &str, judge: &ModelRef) -> Powered {
    run(
        ctx,
        &PoweredRequest {
            exam,
            candidate,
            base: None,
            judge: Some(judge),
            goal: None,
            resamples: 3,
            voice: false,
            cancel: &CancelToken::new(),
        },
    )
    .unwrap()
}

#[test]
fn every_verdict_of_every_arm_is_kept_and_the_arms_are_compared_by_family() {
    let (scratch, ctx) = gate_context("powered", Brain::Missing);
    let exam = exam(&ctx, &scratch);
    let facts: Vec<usize> = (0..FACTS).collect();
    let data = dataset_citing(&ctx, "alpha", &facts, Some(PERSONA_PROMPT), &[]);
    let (candidate, _) = candidate_on(&ctx, data, &[ANCHOR, "alpha"]);
    let judge = arms(&ctx, &candidate.adapter);

    let report = examined(&ctx, &exam, &candidate.candidate, &judge);
    assert_eq!(
        (report.tasks, report.families, report.resamples),
        (12, 6, 3)
    );
    assert!(report.judge.trusted, "{:?}", report.judge);
    assert_eq!(report.families_trained_on, 0);
    // Four arms, each asked under the prompt it is reported under.
    assert_eq!(report.prompts[BASE], "default");
    assert_eq!(report.prompts[CANDIDATE], "default");
    assert!(report.prompts[PERSONA].starts_with("persona: You are a surveyor"));
    assert_eq!(report.prompts[PROMPTED], report.prompts[PERSONA]);
    // Every task keeps every answer of every arm, verdicts and lengths.
    assert_eq!(report.records.len(), 12);
    for record in &report.records {
        for arm in [BASE, PROMPTED, CANDIDATE, PERSONA] {
            let answers = &record.arms[arm];
            assert_eq!(answers.len(), 3, "{arm}: the greedy answer and two sampled");
            assert!(answers
                .iter()
                .all(|a| a.judged.is_some() && a.text.is_some()));
        }
        assert_eq!(record.arms[CANDIDATE][0].judged, Some(true));
        assert_eq!(record.arms[BASE][0].judged, Some(false));
    }
    let first = report
        .comparisons
        .iter()
        .find(|c| (c.first.as_str(), c.second.as_str()) == (PERSONA, BASE))
        .unwrap();
    assert_eq!(
        (first.tasks, first.first_only, first.second_only),
        (12, 12, 0)
    );
    // Twelve tasks over six families: the family-level test cannot do better
    // than six wins, which is the unit of evidence.
    assert!((first.p_families - 0.5f64.powi(6)).abs() < 1e-12);
    assert!(first.difference.unwrap().low > 0.0);
    let hard = report.hard_controls.unwrap();
    assert_eq!(hard.passed, 0, "{hard:?}");
}

#[test]
fn a_family_the_candidate_was_trained_on_is_left_out_of_the_exam() {
    let (scratch, ctx) = gate_context("powered-seen", Brain::Missing);
    let exam = exam(&ctx, &scratch);
    // The candidate's records cite the text of one reserved family.
    let family = exam.tasks[0].family.clone();
    let evidence: Digest = {
        let task = ctx.tasks().get(&exam.tasks[0].task).unwrap();
        task.evidence[0].source.clone()
    };
    let facts: Vec<usize> = (0..FACTS).collect();
    let data = dataset_citing(&ctx, "alpha", &facts, Some(PERSONA_PROMPT), &[evidence]);
    let (candidate, _) = candidate_on(&ctx, data, &[ANCHOR, "alpha"]);
    let judge = arms(&ctx, &candidate.adapter);

    let report = examined(&ctx, &exam, &candidate.candidate, &judge);
    assert_eq!(report.families_trained_on, 1);
    assert_eq!((report.tasks, report.families), (10, 5));
    assert!(report.records.iter().all(|r| r.family != family));
}
