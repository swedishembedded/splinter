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
//! * A variant is never trained on: nothing it asks reaches the dataset,
//!   and an experience of one is refused when a dataset is built.
//!
//! The policy is scripted and training is a test double.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use crate::common;

use std::sync::Arc;

use crate::common::solve_set;
use common::manual::{manual_policy, BAUD_QUESTION, BAUD_VARIANTS, IDLE_VARIANTS, MANUAL};
use common::{scratch_context, Scratch, Scripted};
use splinter_agent::solve::Model;
use splinter_agent::CancelToken;
use splinter_core::model_ref::ModelRef;
use splinter_core::rng::Seed;
use splinter_orchestrator::Context;
use splinter_pipelines::datasets::{build, BuildRequest, ViewName, VoiceBuild};
use splinter_pipelines::solving::harness_protocol;
use splinter_pipelines::sources::{self, SourceTarget};
use splinter_pipelines::tasks::{generate, Generation};
use splinter_pipelines::variants::{generate_variants, VariantsRequest};
use splinter_pipelines::verify::{verify_set, Grading};
use splinter_store::tasks::{TaskEntry, TaskSet};

/// A context whose policy is the manual's, and the tasks `tasks` writes
/// from the manual (recall, and denoise, which cannot be varied).
fn manual_tasks(test: &str) -> (Scratch, Context, splinter_pipelines::tasks::TasksGenerated) {
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
            author: None,
            // seed: the denoise kind refuses to build without one; any value serves
            written: &[],
            denoise_seed: Some(Seed::new(1)),
            deadline: None,
            cancel: CancelToken::new(),
        },
    )
    .unwrap();
    (scratch, ctx, generated)
}

fn variants_of(
    ctx: &Context,
    generated: &splinter_pipelines::tasks::TasksGenerated,
) -> splinter_pipelines::variants::VariantsGenerated {
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

/// Tasks are addressed by content, so a variant worded exactly as another
/// task of the set would be that task: recorded as a variant, it would make
/// an original something that is "never trained on".
#[test]
fn a_variant_worded_as_another_task_of_the_set_is_left_out() {
    let (_scratch, ctx, generated) = manual_tasks("variants-twin");
    let originals = ctx.tasks().get_set(&generated.task_set).unwrap();
    let baud = originals
        .members
        .iter()
        .find(|m| ctx.tasks().get(&m.task).unwrap().instruction == BAUD_QUESTION)
        .unwrap()
        .clone();
    let twin = ctx
        .tasks()
        .get(&baud.task)
        .unwrap()
        .with_instruction(BAUD_VARIANTS[0])
        .unwrap();
    let twin = ctx.tasks().put(&twin).unwrap();
    let set = ctx
        .tasks()
        .put_set(&TaskSet {
            name: "a fact asked twice".into(),
            members: vec![
                baud.clone(),
                TaskEntry {
                    task: twin.clone(),
                    ..baud.clone()
                },
            ],
        })
        .unwrap();
    let variants = generate_variants(
        &ctx,
        &VariantsRequest {
            task_set: &set,
            generator: &ModelRef::policy_default(),
            per_task: 3,
            deadline: None,
            cancel: CancelToken::new(),
        },
    )
    .unwrap();
    let recorded = ctx.tasks().get_set(&variants.variant_set.unwrap()).unwrap();
    assert!(
        recorded
            .members
            .iter()
            .all(|m| m.task != twin && m.task != baud.task),
        "{recorded:#?}"
    );
    assert_eq!(recorded.members.len(), 2, "the two wordings no task has");
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
        None,
        harness_protocol(),
    )
    .unwrap();
    verify_set(
        &ctx,
        &solved.experience_set,
        Grading::ActiveJudge,
        &CancelToken::new(),
    )
    .unwrap();
    let refused = build(
        &ctx,
        &BuildRequest {
            sets: vec![solved.experience_set],
            view: ViewName::SftFinal,
            strip: None,
            min_strength: None,
            system_prompt: None,
            export_only: false,
            limit: None,
            voice: VoiceBuild::default(),
            max_block: None,
            pack: false,
            split_over_block: false,
            control: Default::default(),
            serialise_calls: false,
            only_actions: None,
            roles: None,
            images: false,
            rl: Default::default(),
            transition: Default::default(),
            practice: Vec::new(),
            split_seed: None,
            history: Default::default(),
            pass_curation: Default::default(),
            base: None,
        },
    )
    .unwrap_err();
    assert!(refused.is_refusal(), "{refused}");
    assert!(refused.to_string().contains("variant"), "{refused}");
}

/// A task some set lists as an original is one: an older store that
/// recorded it as the variant of another (a variant worded as it was) does
/// not make it untrainable.
#[test]
fn a_task_listed_as_an_original_is_trained_on_though_recorded_as_a_variant() {
    let (_scratch, ctx, generated) = manual_tasks("variants-recorded-twin");
    let originals = ctx.tasks().get_set(&generated.task_set).unwrap();
    let baud = originals
        .members
        .iter()
        .find(|m| ctx.tasks().get(&m.task).unwrap().instruction == BAUD_QUESTION)
        .unwrap()
        .clone();
    let twin = ctx
        .tasks()
        .get(&baud.task)
        .unwrap()
        .with_instruction(BAUD_VARIANTS[0])
        .unwrap();
    let twin = ctx.tasks().put(&twin).unwrap();
    let twin_entry = TaskEntry {
        task: twin,
        ..baud.clone()
    };
    let facts = ctx
        .tasks()
        .put_set(&TaskSet {
            name: "a fact asked twice".into(),
            members: vec![baud.clone(), twin_entry.clone()],
        })
        .unwrap();
    ctx.tasks()
        .put_set(&TaskSet {
            name: "recorded by an older run".into(),
            members: vec![TaskEntry {
                variant_of: Some(baud.task.clone()),
                ..twin_entry
            }],
        })
        .unwrap();
    let solved = solve_set(
        &ctx,
        &facts,
        &ModelRef::policy_default(),
        None,
        &CancelToken::new(),
        None,
        harness_protocol(),
    )
    .unwrap();
    verify_set(
        &ctx,
        &solved.experience_set,
        Grading::ActiveJudge,
        &CancelToken::new(),
    )
    .unwrap();
    let built = build(
        &ctx,
        &BuildRequest {
            sets: vec![solved.experience_set],
            view: ViewName::SftFinal,
            strip: None,
            min_strength: None,
            system_prompt: None,
            export_only: false,
            limit: None,
            voice: VoiceBuild::default(),
            max_block: None,
            pack: false,
            split_over_block: false,
            control: Default::default(),
            serialise_calls: false,
            only_actions: None,
            roles: None,
            images: false,
            rl: Default::default(),
            transition: Default::default(),
            practice: Vec::new(),
            split_seed: None,
            history: Default::default(),
            pass_curation: Default::default(),
            base: None,
        },
    )
    .unwrap();
    assert_eq!(built.records, 2);
}
