// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning agents that learn what a person
// advised from what they wrote, for its clients. If your team needs
// expertise in grounding a model's advice in a body of writing, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: `learn` over a person's letters with the `advise` kind teaches the
//! policy the person's own advice, and only advice that quotes the letters.
//!
//! The generator is shown only the sections that read as advice. A student
//! that never saw the letters answers no predicament closed-book; a teacher
//! shown the passage answers by quoting it, which the quotation verifier
//! passes, so the training record is the predicament and the advice in the
//! writer's own words. A teacher that invents a quotation is failed by the
//! verifier and nothing is trained on its answer.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::sync::{Arc, Mutex};

use common::gate::{gate_context, Brain, FakeTrainer, ANCHOR};
use common::Scripted;
use serde_json::json;
use splinter_agent::solve::{Model, MATERIAL_HEADING};
use splinter_campaign::learn::{learn, LearnRequest, Learned};
use splinter_campaign::model_ref::ModelRef;
use splinter_campaign::release::arm;
use splinter_campaign::train::{TrainPlan, Trainer};
use splinter_campaign::{CampaignError, Context};
use splinter_policy::train::{Trained, TrainedPreference};
use sven_sdk::CancelToken;

const LETTER: &str = "# To a young man

## Business

I have received your favour of the tenth and enclose the bill of lading for the hogsheads of tobacco shipped on the brig Eliza, which should reach Havre within the month.

## Study

I advise you to fix a habit of study every morning before you do anything else. Never let a day pass without reading something of history or ethics, and always write down what you have read, for the memory fails what the pen has not fixed.

## Weather

The weather here has been mild and the roads are passable.
";

const ADVICE: &str = "I advise you to fix a habit of study every morning before you do anything else. Never let a day pass without reading something of history or ethics, and always write down what you have read, for the memory fails what the pen has not fixed.";

const SITUATION: &str = "I am nineteen and have finished my first year at college, but I find my mornings slip away in idleness; what would you advise me to do about my habits?";

/// A policy that writes one advice task from the letter, answers nothing
/// closed-book, and, shown the passage, answers with `teacher_answer`.
fn policy(teacher_answer: &'static str) -> Scripted {
    Scripted::new(move |prompt| {
        if prompt.contains("You write training tasks") {
            json!({ "tasks": [{
                "instruction": SITUATION,
                "reference": ADVICE,
                "evidence": [{ "section": 0, "quote": ADVICE }]
            }]})
            .to_string()
        } else if prompt.contains(MATERIAL_HEADING) && prompt.contains("habit of study") {
            teacher_answer.to_string()
        } else {
            "I do not know.".to_string()
        }
    })
}

/// A trainer that records what it was asked to train on.
struct Student {
    plans: Mutex<Vec<TrainPlan>>,
}

impl Trainer for Student {
    fn train(
        &self,
        ctx: &Context,
        plan: &TrainPlan,
        cancel: &CancelToken,
    ) -> Result<Trained, CampaignError> {
        self.plans.lock().unwrap().push(plan.clone());
        let trained = FakeTrainer::knowing(&[ANCHOR]).train(ctx, plan, cancel)?;
        ctx.add_model(
            arm(ctx.config(), Some(&trained.adapter)),
            Model::new(
                Arc::new(Scripted::new(|_| "I do not know.".into())),
                "scripted/learned",
            ),
        );
        Ok(trained)
    }

    fn train_preference(
        &self,
        _ctx: &Context,
        _plan: &TrainPlan,
        _cancel: &CancelToken,
    ) -> Result<TrainedPreference, CampaignError> {
        panic!("learn trains chat datasets only")
    }
}

type Ran = (
    splinter_campaign::learn::LearnReport,
    Student,
    common::Scratch,
);

fn run(test: &str, teacher_answer: &'static str) -> Ran {
    run_with(test, teacher_answer, false)
}

fn run_with(test: &str, teacher_answer: &'static str, distill: bool) -> Ran {
    let (scratch, ctx) = gate_context(test, Brain::Missing);
    ctx.add_model(
        ModelRef::policy_default(),
        Model::new(Arc::new(policy(teacher_answer)), common::POLICY),
    );
    let letter = scratch.0.join("letter.md");
    std::fs::write(&letter, LETTER).unwrap();
    let student = Student {
        plans: Mutex::new(Vec::new()),
    };
    let Learned::Ran(ran) = learn(
        &ctx,
        &LearnRequest {
            sources: vec![letter.display().to_string()],
            kinds: vec!["advise".into()],
            no_release: true,
            distill,
            ..LearnRequest::default()
        },
        &student,
    )
    .unwrap() else {
        panic!("a learn that is not a dry run runs");
    };
    (ran.report, student, scratch)
}

#[test]
fn a_teacher_that_quotes_the_letter_teaches_the_advice_in_the_writers_own_words() {
    let answer: &'static str =
        Box::leak(format!("In my letter, I advised: \"{ADVICE}\"").into_boxed_str());
    let (report, _, _scratch) = run("advice-learn-quotes", answer);

    let generated = report.tasks.as_ref().unwrap();
    assert_eq!(generated.tasks, 1, "{generated:#?}");
    assert_eq!(generated.per_kind["advise"].admitted, 1);

    // The student never saw the letter: its closed-book answers fail.
    let verified = report.verify.as_ref().unwrap();
    assert_eq!(verified.passed, 0, "{verified:#?}");
    // The teacher, shown the passage, quotes it, and the quotation verifier
    // passes the answer.
    let taught = report.teach.as_ref().unwrap();
    assert_eq!(
        (taught.verify.passed, taught.verify.failed),
        (1, 0),
        "{taught:#?}"
    );

    let dataset = report.dataset.as_ref().unwrap();
    assert_eq!(dataset.records, 1, "{dataset:#?}");
    let text = std::fs::read_to_string(&dataset.path).unwrap();
    let record: serde_json::Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
    let turns = record["messages"].as_array().unwrap();
    assert_eq!(turns[1]["content"], SITUATION);
    assert!(
        turns[2]["content"].as_str().unwrap().contains(ADVICE),
        "the record's answer holds the advice in the writer's words"
    );
    assert!(
        !text.contains(MATERIAL_HEADING),
        "the passage is teacher-only"
    );
}

#[test]
fn a_teacher_that_invents_a_quotation_teaches_nothing() {
    let answer =
        "I advised: \"the surest road to a quiet mind is to borrow boldly and repay never, \
                  for creditors forget\" and I stand by it.";
    let (report, student, _scratch) = run("advice-learn-invents", answer);

    let taught = report.teach.as_ref().unwrap();
    assert_eq!(
        (taught.verify.passed, taught.verify.failed),
        (0, 1),
        "{taught:#?}"
    );
    assert!(report.dataset.is_none(), "{report:#?}");
    assert!(report.stopped.is_some(), "no task is worth training on");
    assert!(
        student.plans.lock().unwrap().is_empty(),
        "nothing was trained"
    );
}

#[test]
fn distilling_has_the_teacher_answer_every_task_without_the_student_trying() {
    let answer: &'static str =
        Box::leak(format!("In my letter, I advised: \"{ADVICE}\"").into_boxed_str());
    let (report, student, _scratch) = run_with("advice-learn-distill", answer, true);

    assert!(report.solve.is_none(), "the student made no attempt");
    assert!(
        report.verify.is_none(),
        "there was nothing of the student's to grade"
    );
    assert!(
        report.frontier.is_none(),
        "no frontier is measured without attempts"
    );
    assert!(report.critique.is_none(), "no student attempt failed");
    let taught = report.teach.as_ref().unwrap();
    assert_eq!(
        (
            taught.solve.solved,
            taught.verify.passed,
            taught.verify.failed
        ),
        (1, 1, 0),
        "{taught:#?}"
    );

    let dataset = report.dataset.as_ref().unwrap();
    assert_eq!(dataset.records, 1, "{dataset:#?}");
    let text = std::fs::read_to_string(&dataset.path).unwrap();
    assert!(text.contains(ADVICE) && text.contains(SITUATION));
    // One record is too few to hold any out for scoring, so nothing is trained.
    assert!(report
        .stopped
        .as_deref()
        .unwrap_or_default()
        .contains("record(s) passed"));
    assert!(student.plans.lock().unwrap().is_empty());
}
