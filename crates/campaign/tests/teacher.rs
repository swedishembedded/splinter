// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning agents that acquire knowledge
// their model does not have yet, for its clients. If your team needs
// expertise in knowledge distillation or continual learning, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: `learn` teaches the policy a fact it does not know.
//!
//! A policy that never saw a document solves none of its tasks
//! closed-book. Each such task is solved again by a teacher - the same
//! policy, open-book, shown the source sections the task's evidence falls
//! in - graded by the task's own verifiers and recorded as an experience
//! whose provenance marks it the teacher's. The curriculum keeps a task the
//! student never solves when a teacher's answer to it is verified; the
//! training records are the student's: the system turn every solve runs
//! under, the instruction alone, and the verified answer - never the
//! passage. Concept mastery counts the student's closed-book attempts only,
//! and the release gate grades the trained candidate closed-book.
//!
//! The policy is scripted: it answers right only when it is shown the
//! passage; training is a test double.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::sync::{Arc, Mutex};

use common::gate::{gate_context, Brain, FakeTrainer, ANCHOR};
use common::manual::{BAUD_QUESTION, BAUD_QUOTE, MANUAL};
use common::Scripted;
use serde_json::json;
use splinter_agent::solve::{Model, MATERIAL_HEADING, SYSTEM_PROMPT};
use splinter_campaign::curriculum::mastery::weakest;
use splinter_campaign::learn::{learn, LearnRequest, Learned};
use splinter_campaign::model_ref::ModelRef;
use splinter_campaign::release::arm;
use splinter_campaign::train::{TrainPlan, Trainer};
use splinter_campaign::{CampaignError, Context};
use splinter_policy::train::{Trained, TrainedPreference};
use splinter_store::annotation::decide;
use sven_sdk::CancelToken;

/// The idle-current question.
const IDLE_QUESTION: &str = "How much current does the Frobnicator draw when idle?";

/// Text only the manual's sections hold, never a question or an answer:
/// a prompt holding it was shown the passage.
const CONSOLE_PASSAGE: &str = "115200 baud with eight data bits";
const POWER_PASSAGE: &str = "40 mA when idle and 900 mA at full load";

/// A policy that never saw the manual: it writes two recall tasks grounded
/// in it, and answers one right only when the passage that holds the
/// answer is in its prompt.
fn open_book_only_policy() -> Scripted {
    Scripted::new(|prompt| {
        if prompt.contains("You write training tasks") {
            json!({ "tasks": [
                {
                    "instruction": BAUD_QUESTION,
                    "subject": "Frobnicator",
                    "reference": "115200 baud",
                    "evidence": [{ "section": 1, "quote": BAUD_QUOTE }]
                },
                {
                    "instruction": IDLE_QUESTION,
                    "subject": "Frobnicator",
                    "reference": "40 mA",
                    "evidence": [{ "section": 2, "quote": "40 mA when idle" }]
                }
            ]})
            .to_string()
        } else if prompt.contains("You are reviewing an attempt") {
            "The answer is not what the manual says.".into()
        } else if prompt.contains(CONSOLE_PASSAGE) {
            "115200 baud".into()
        } else if prompt.contains(POWER_PASSAGE) {
            "40 mA".into()
        } else {
            "I do not know.".into()
        }
    })
}

/// What a candidate trained on the manual answers, closed-book.
fn learned_model() -> Scripted {
    Scripted::new(|prompt| {
        if prompt.contains("baud rate") {
            "115200 baud".into()
        } else if prompt.contains("when idle") {
            "40 mA".into()
        } else {
            "I do not know.".into()
        }
    })
}

/// Trains the fixtures' fake adapter and serves its candidate as `learned`.
struct Student {
    learned: Scripted,
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
            Model::new(Arc::new(self.learned.clone()), "scripted/learned"),
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

#[test]
fn learn_teaches_a_fact_the_policy_never_answers_closed_book() {
    let (scratch, ctx) = gate_context("learn-teacher", Brain::Missing);
    let policy = open_book_only_policy();
    ctx.add_model(
        ModelRef::policy_default(),
        Model::new(Arc::new(policy.clone()), common::POLICY),
    );
    let manual = scratch.0.join("manual.md");
    std::fs::write(&manual, MANUAL).unwrap();
    let student = Student {
        learned: learned_model(),
        plans: Mutex::new(Vec::new()),
    };

    let Learned::Ran(run) = learn(
        &ctx,
        &LearnRequest {
            sources: vec![manual.display().to_string()],
            ..LearnRequest::default()
        },
        &student,
    )
    .unwrap() else {
        panic!("a learn that is not a dry run runs");
    };
    let report = &run.report;

    // Closed-book, the student solves nothing; the teacher, shown each
    // task's passage, answers both, and both answers are verified.
    let verified = report.verify.as_ref().unwrap();
    assert_eq!((verified.passed, verified.failed), (0, 8), "{verified:#?}");
    let taught = report.teach.as_ref().unwrap();
    assert_eq!(taught.solve.solver, common::POLICY, "the policy teaches");
    assert_eq!(
        (
            taught.solve.solved,
            taught.verify.passed,
            taught.verify.failed
        ),
        (2, 2, 0),
        "{taught:#?}"
    );
    let store = ctx.experiences();
    for id in store.get_set(&taught.solve.experience_set).unwrap().members {
        let experience = store.get(&id).unwrap();
        assert!(experience.provenance.teacher, "{:?}", experience.provenance);
        assert!(
            decide(&store.annotations(&id).unwrap().annotations)
                .unwrap()
                .passed
        );
        assert!(
            !experience.instruction.contains(MATERIAL_HEADING),
            "the experience records the task's own instruction"
        );
    }
    assert!(
        policy
            .prompts
            .lock()
            .unwrap()
            .iter()
            .any(|p| p.contains(MATERIAL_HEADING) && p.contains(CONSOLE_PASSAGE)),
        "the teacher was shown the console section"
    );

    // Both tasks are kept: never solved closed-book, with a verified
    // answer.
    let frontier = report.frontier.as_ref().unwrap();
    let d = frontier.distribution;
    assert_eq!(
        (d.always, d.frontier, d.taught, d.never, d.unmeasured),
        (0, 0, 2, 0, 0),
        "{frontier:#?}"
    );
    let recorded: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&frontier.path).unwrap()).unwrap();
    for row in recorded["tasks"].as_array().unwrap() {
        assert_eq!(row["class"], "taught", "{row}");
        assert_eq!(
            (&row["passes"], &row["teacher"]["passes"]),
            (&json!(0), &json!(1))
        );
    }

    // The records are the student's: the system turn the solver ran
    // under, the instruction alone, and the verified answer.
    let dataset = report.dataset.as_ref().unwrap();
    assert_eq!(dataset.records, 2, "{dataset:#?}");
    let text = std::fs::read_to_string(&dataset.path).unwrap();
    let mut records: Vec<Vec<(String, String)>> = text
        .lines()
        .map(|line| {
            let record: serde_json::Value = serde_json::from_str(line).unwrap();
            record["messages"]
                .as_array()
                .unwrap()
                .iter()
                .map(|m| {
                    (
                        m["role"].as_str().unwrap().to_string(),
                        m["content"].as_str().unwrap().to_string(),
                    )
                })
                .collect()
        })
        .collect();
    records.sort();
    let turn = |role: &str, content: &str| (role.to_string(), content.to_string());
    assert_eq!(
        records,
        [
            vec![
                turn("system", SYSTEM_PROMPT),
                turn("user", BAUD_QUESTION),
                turn("assistant", "115200 baud")
            ],
            vec![
                turn("system", SYSTEM_PROMPT),
                turn("user", IDLE_QUESTION),
                turn("assistant", "40 mA")
            ],
        ]
    );
    for passage in [CONSOLE_PASSAGE, POWER_PASSAGE, MATERIAL_HEADING] {
        assert!(
            !text.contains(passage),
            "the passage is teacher-only: {text}"
        );
    }
    // Every solve - the student's, the teacher's, the critic's, a retry -
    // ran under that system turn; only the task generator's typed call
    // runs under its own role.
    let prompts = policy.prompts.lock().unwrap();
    let systems = policy.systems.lock().unwrap();
    let solves: Vec<&Vec<String>> = prompts
        .iter()
        .zip(systems.iter())
        .filter(|(prompt, _)| !prompt.contains("You write training tasks"))
        .map(|(_, system)| system)
        .collect();
    assert!(solves.len() >= 10, "{}", solves.len());
    for system in solves {
        assert_eq!(
            system,
            &[SYSTEM_PROMPT],
            "a record's system turn is the one every solve ran under"
        );
    }
    splinter_policy::train::validate_dataset(&dataset.path).unwrap();

    // Mastery is the student's closed-book record: a teacher's solve says
    // nothing about what the student knows.
    let mastery = weakest(&ctx, 5).unwrap();
    assert_eq!(mastery.measured, 2, "{mastery:#?}");
    for concept in &mastery.weakest {
        let current = concept.current.as_ref().unwrap();
        assert_eq!((current.graded, current.passes), (4, 0), "{concept:#?}");
    }

    // The candidate trained on them is graded closed-book: its held-out
    // question reaches it without the passage.
    assert_eq!(student.plans.lock().unwrap().len(), 1);
    let gated = report.release.as_ref().unwrap();
    let improvement = gated.gate.improvement.measured.as_ref().unwrap();
    assert_eq!(improvement.comparison.paired, 1, "{gated:#?}");
    let probes = student.learned.prompts.lock().unwrap();
    assert!(!probes.is_empty(), "the candidate was graded");
    for probe in probes.iter() {
        for passage in [CONSOLE_PASSAGE, POWER_PASSAGE, MATERIAL_HEADING] {
            assert!(!probe.contains(passage), "a probe is closed-book: {probe}");
        }
    }
}
