// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements contamination-free evaluation of fine-tuned
// models, for its clients. If your team needs expertise in model
// evaluation, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: what a candidate learned is measured on the same facts asked in
//! other words.
//!
//! * The variants stage has the generator model write differently worded
//!   questions about each task's fact and stores what code admits as a task
//!   set of its own, each member recording the task it is a variant of; a
//!   task that cannot be varied is counted.
//! * A variant is never trained on: it is in no task set `learn` solves,
//!   nothing it asks reaches the dataset, and an experience of one is
//!   refused when a dataset is built.
//! * `learn` writes the variants of the tasks kept for training, and the
//!   gate measures the candidate on those of the tasks it trained on.
//!
//! The policy is scripted and training is a test double.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::sync::{Arc, Mutex};

use common::gate::{gate_context, Brain, FakeTrainer, ANCHOR};
use common::manual::{
    manual_policy, BAUD_QUESTION, BAUD_VARIANTS, IDLE_QUESTION, IDLE_VARIANTS, MANUAL,
};
use common::{scratch_context, Scratch, Scripted};
use splinter_agent::solve::Model;
use splinter_agent::CancelToken;
use splinter_campaign::datasets::{build, BuildRequest, ViewName};
use splinter_campaign::learn::{learn, LearnRequest, Learned, STAGES};
use splinter_campaign::release::arm;
use splinter_campaign::solving::solve_set;
use splinter_campaign::sources::{self, SourceTarget};
use splinter_campaign::tasks::{generate, Generation};
use splinter_campaign::train::{TrainPlan, Trainer};
use splinter_campaign::variants::{generate_variants, VariantsRequest};
use splinter_campaign::verify::verify_set;
use splinter_campaign::{CampaignError, Context};
use splinter_core::model_ref::ModelRef;
use splinter_model::train::{Trained, TrainedPreference};
use splinter_store::runs::read_run;

/// A context whose policy is the manual's, and the tasks `tasks` writes
/// from the manual (recall, and denoise, which cannot be varied).
fn manual_tasks(test: &str) -> (Scratch, Context, splinter_campaign::tasks::TasksGenerated) {
    let (scratch, ctx) = scratch_context(test, manual_policy(), false);
    let manual = scratch.0.join("manual.md");
    std::fs::write(&manual, MANUAL).unwrap();
    let source = sources::add(
        &ctx,
        &SourceTarget::from_learn_arg(&manual.display().to_string()).unwrap(),
    )
    .unwrap()
    .source
    .id;
    let kinds = ["recall".to_string(), "denoise".to_string()];
    let generated = generate(
        &ctx,
        &Generation {
            sources: &[source],
            sections: &[],
            kinds: &kinds,
            generator: &ModelRef::policy_default(),
            goal: None,
            deadline: None,
            cancel: CancelToken::new(),
        },
    )
    .unwrap();
    (scratch, ctx, generated)
}

fn variants_of(
    ctx: &Context,
    generated: &splinter_campaign::tasks::TasksGenerated,
) -> splinter_campaign::variants::VariantsGenerated {
    generate_variants(
        ctx,
        &VariantsRequest {
            task_set: &generated.task_set,
            generator: &ModelRef::policy_default(),
            per_task: 3,
            deadline: None,
            cancel: CancelToken::new(),
        },
    )
    .unwrap()
}

#[test]
fn each_task_gets_variants_in_a_set_of_their_own_recording_what_they_vary() {
    let (_scratch, ctx, generated) = manual_tasks("variants-stage");
    assert_eq!(generated.tasks, 3, "two recall tasks and one denoise task");
    let variants = variants_of(&ctx, &generated);

    assert_eq!((variants.tasks, variants.variants), (2, 6), "{variants:#?}");
    assert_eq!(
        variants.ineligible.iter().collect::<Vec<_>>(),
        [(&"not_a_model_written_kind".to_string(), &1)],
        "the denoise task asks no question to reword"
    );
    let originals = ctx.tasks().get_set(&generated.task_set).unwrap();
    let set = ctx.tasks().get_set(&variants.variant_set.unwrap()).unwrap();
    assert!(originals.members.iter().all(|m| m.variant_of.is_none()));
    assert_eq!(set.members.len(), 6);
    for member in &set.members {
        let variant = ctx.tasks().get(&member.task).unwrap();
        let original = originals
            .members
            .iter()
            .find(|m| Some(&m.task) == member.variant_of.as_ref())
            .map(|m| ctx.tasks().get(&m.task).unwrap())
            .expect("a variant names a task of the set it varies");
        assert!(
            BAUD_VARIANTS.contains(&variant.instruction.as_str())
                || IDLE_VARIANTS.contains(&variant.instruction.as_str()),
            "{}",
            variant.instruction
        );
        assert_eq!(variant.task.kind, original.task.kind);
        assert_eq!(variant.evidence, original.evidence);
        assert_eq!(variant.privileged, original.privileged, "same reference");
        assert!(member
            .generator
            .as_deref()
            .unwrap()
            .ends_with(common::POLICY));
        assert!(member.prompt.is_some());
    }
}

#[test]
fn a_model_that_writes_no_usable_variant_leaves_no_set_and_says_why() {
    let (_scratch, ctx, generated) = manual_tasks("variants-none");
    let ctx_mute = {
        let mute = Scripted::new(|prompt| {
            if prompt.contains("differently worded questions") {
                "Sure! Here are some questions.".into()
            } else {
                String::new()
            }
        });
        ctx.add_model(
            ModelRef::policy_default(),
            Model::new(Arc::new(mute), common::POLICY),
        );
        ctx
    };
    let variants = variants_of(&ctx_mute, &generated);
    assert_eq!(variants.variants, 0);
    assert_eq!(variants.variant_set, None);
    assert_eq!(
        variants.rejected.get("malformed"),
        Some(&2),
        "{variants:#?}"
    );
    assert!(variants
        .rejections
        .iter()
        .all(|r| r.detail.contains("Sure!")));
}

#[test]
fn a_variant_never_enters_a_training_dataset() {
    let (_scratch, ctx, generated) = manual_tasks("variants-never-trained");
    let variants = variants_of(&ctx, &generated);
    let variant_set = variants.variant_set.unwrap();

    // Solved and verified like any task, the variants' experiences are
    // refused by every dataset.
    let solved = solve_set(
        &ctx,
        &variant_set,
        &ModelRef::policy_default(),
        None,
        &CancelToken::new(),
    )
    .unwrap();
    verify_set(&ctx, &solved.experience_set, None, &CancelToken::new()).unwrap();
    let refused = build(
        &ctx,
        &BuildRequest {
            sets: vec![solved.experience_set],
            view: ViewName::SftFinal,
            strip: None,
            min_strength: None,
            export_only: false,
        },
    )
    .unwrap_err();
    assert!(refused.is_refusal(), "{refused}");
    assert!(refused.to_string().contains("variant"), "{refused}");
}

/// Trains the fixtures' fake adapter and serves its candidate as the model
/// that learned the manual: it answers the baud rate and the idle current
/// however they are asked.
struct Student {
    learned: Scripted,
    handed: Mutex<Vec<String>>,
}

impl Trainer for Student {
    fn train(
        &self,
        ctx: &Context,
        plan: &TrainPlan,
        cancel: &CancelToken,
    ) -> Result<Trained, CampaignError> {
        for dataset in &plan.datasets {
            self.handed
                .lock()
                .unwrap()
                .push(std::fs::read_to_string(&dataset.path).unwrap());
        }
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
fn learn_measures_the_candidate_on_variants_of_what_it_trained_on() {
    let (scratch, ctx) = gate_context("learn-variants", Brain::Missing);
    ctx.add_model(
        ModelRef::policy_default(),
        Model::new(Arc::new(manual_policy()), common::POLICY),
    );
    let manual = scratch.0.join("manual.md");
    std::fs::write(&manual, MANUAL).unwrap();
    let student = Student {
        learned: Scripted::new(|prompt| {
            if prompt.contains("baud") {
                "115200 baud".into()
            } else if prompt.contains("idle") {
                "40 mA".into()
            } else {
                "I do not know.".into()
            }
        }),
        handed: Mutex::new(Vec::new()),
    };

    let Learned::Ran(run) = learn(
        &ctx,
        &LearnRequest {
            sources: vec![manual.display().to_string()],
            no_frontier: true,
            ..LearnRequest::default()
        },
        &student,
    )
    .unwrap() else {
        panic!("a learn that is not a dry run runs");
    };
    let report = &run.report;
    assert_eq!(report.stopped, None, "{report:#?}");

    // The stage sits between the tasks being kept and the training set
    // being selected, and is part of the plan.
    let stages: Vec<String> = read_run(ctx.workspace(), &run.run)
        .unwrap()
        .stages
        .into_iter()
        .map(|s| s.stage)
        .collect();
    let at = |name: &str| stages.iter().position(|s| s == name).unwrap();
    assert!(
        at("teach") < at("variants") && at("variants") < at("select"),
        "{stages:?}"
    );
    assert!(STAGES.contains(&"variants"));
    let variants = report.variants.as_ref().unwrap();
    assert_eq!((variants.tasks, variants.variants), (2, 6), "{variants:#?}");

    // Nothing a variant asks is a training record, and the tasks solved
    // are not the variants.
    let trained = student.handed.lock().unwrap().join("\n");
    assert!(trained.contains(BAUD_QUESTION) || trained.contains(IDLE_QUESTION));
    for wording in BAUD_VARIANTS.iter().chain(&IDLE_VARIANTS) {
        assert!(!trained.contains(wording), "{wording}");
    }
    let solved_tasks = ctx
        .tasks()
        .get_set(&report.tasks.as_ref().unwrap().task_set)
        .unwrap();
    assert!(solved_tasks.members.iter().all(|m| m.variant_of.is_none()));

    // The gate measured the candidate on the variants of the one task it
    // trained on, and said how many.
    let gate = &report.release.as_ref().unwrap().gate;
    let improvement = gate.improvement.measured.as_ref().unwrap();
    let measured = improvement.variants.as_ref().unwrap();
    assert_eq!(measured.tasks, 3, "{improvement:#?}");
    assert!(measured.excluded.is_empty(), "{improvement:#?}");
    assert_eq!(
        improvement.suite.tasks,
        3 + 1,
        "the variants and the held-out record"
    );
    assert_eq!(improvement.comparison.candidate_wins, 4, "{improvement:#?}");
}
