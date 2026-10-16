// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements leakage-free held-out examinations of
// code-editing agents, for its clients. If your team needs expertise in
// keeping an exam's files out of the data a model is trained on, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: `dataset build` over workspace attempts keeps the exam sealed.
//!
//! * Building from an attempt at a task of a reserved family is refused,
//!   naming the family.
//! * A supervised action whose tool-call arguments name a file of a
//!   reserved family is refused, however clean the attempt's own family is.
//! * A clean build remembers the families it trained on, so that a later
//!   run never reserves one of them.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::collections::BTreeSet;

use atif::{
    AgentProfile, ObservationEntry, StepObservation, StepOrigin, ToolInvocation, TraceStep,
    Trajectory,
};
use common::{scratch_context, Scripted};
use serde_json::json;
use splinter_core::annotation::{Annotation, AnnotationBody, Outcome, Producer, Strength};
use splinter_core::clock::FixedClock;
use splinter_core::experience::{Environment, Experience, Provenance, Task};
use splinter_core::kinds::WORKSPACE_CHANGE;
use splinter_core::rng::Seed;
use splinter_orchestrator::{Context, OrchestratorError};
use splinter_pipelines::datasets::ViewName;
use splinter_pipelines::datasets::{build, BuildRequest, VoiceBuild};
use splinter_pipelines::exam_seal;
use splinter_store::experiences::{ExperienceSet, SetId};

/// An attempt at a task whose fault is in `family`; the agent edits
/// `edited` and the checks then pass.
fn attempt(ctx: &Context, family: &str, edited: &str) -> Experience {
    let task = Task::new(
        WORKSPACE_CHANGE,
        Vec::new(),
        Environment::new(Environment::WORKSPACE, json!({"snapshot": "tree-1"})),
        "Make the failing test pass.",
        Vec::new(),
    )
    .unwrap()
    .with_concepts([family])
    .unwrap();
    let mut trajectory = Trajectory::new("ATIF-v1.7", AgentProfile::new("scripted", "1"));
    let mut edit = TraceStep::new(2, StepOrigin::Agent, "");
    edit.tool_calls =
        Some(vec![ToolInvocation::new("c1", "edit")
            .with_arguments(json!({"path": edited, "text": "fixed"}))]);
    edit.observation = Some(StepObservation::single(ObservationEntry::for_call(
        "c1", "ok",
    )));
    trajectory.steps = vec![
        TraceStep::new(1, StepOrigin::User, "Make the failing test pass."),
        edit,
        TraceStep::new(3, StepOrigin::Agent, "Done."),
    ];
    let experience = Experience::new(
        task,
        trajectory,
        Some("PATCH".into()),
        Provenance::new(
            "scripted/solver-1",
            &FixedClock::new("2026-10-16T01:00:00.000Z"),
        ),
    )
    .unwrap();
    let id = ctx.experiences().put(&experience).unwrap();
    ctx.experiences()
        .annotate(&Annotation {
            experience: id,
            producer: Producer {
                name: "grader".into(),
                version: "1".into(),
            },
            body: AnnotationBody::Verdict {
                outcome: Outcome::Pass,
                strength: Strength::Executable,
                evidence: json!({}),
            },
        })
        .unwrap();
    experience
}

fn set_of(ctx: &Context, attempts: &[Experience]) -> SetId {
    ctx.experiences()
        .put_set(&ExperienceSet {
            name: "attempts".into(),
            members: attempts.iter().map(|a| a.id().unwrap()).collect(),
        })
        .unwrap()
}

fn build_sft_step(ctx: &Context, set: SetId) -> Result<(), OrchestratorError> {
    build(
        ctx,
        &BuildRequest {
            sets: vec![set],
            view: ViewName::SftStep,
            strip: None,
            min_strength: Some(Strength::Executable),
            system_prompt: None,
            export_only: false,
            limit: None,
            voice: VoiceBuild::default(),
            max_block: None,
        },
    )
    .map(|_| ())
}

fn families(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|n| (*n).to_string()).collect()
}

fn context(test: &str) -> Context {
    let (scratch, ctx) = scratch_context(test, Scripted::new(|_| String::new()), false);
    std::mem::forget(scratch);
    // seed: the reserved record needs one; no outcome depends on its value
    exam_seal::remember_reserved(&ctx, &families(&["exam.py"]), Seed::new(1)).unwrap();
    ctx
}

#[test]
fn an_attempt_of_a_reserved_family_is_refused_naming_the_family() {
    let ctx = context("seal-family");
    let set = set_of(&ctx, &[attempt(&ctx, "exam.py", "exam.py")]);
    let refused = build_sft_step(&ctx, set).unwrap_err().to_string();
    assert!(refused.contains("exam.py"), "{refused}");
    assert!(refused.contains("leakage"), "{refused}");
}

#[test]
fn an_action_editing_a_file_of_a_reserved_family_is_refused() {
    let ctx = context("seal-action");
    let set = set_of(&ctx, &[attempt(&ctx, "train.py", "src/exam.py")]);
    let refused = build_sft_step(&ctx, set).unwrap_err().to_string();
    assert!(refused.contains("exam.py"), "{refused}");
    assert!(exam_seal::trained(&ctx).unwrap().is_empty());
}

#[test]
fn a_file_that_only_ends_like_an_exam_file_is_not_that_file() {
    let ctx = context("seal-suffix");
    let set = set_of(&ctx, &[attempt(&ctx, "train.py", "not_exam.py")]);
    build_sft_step(&ctx, set).unwrap();
}

#[test]
fn a_clean_build_records_the_families_it_trained_on() {
    let ctx = context("seal-clean");
    let set = set_of(&ctx, &[attempt(&ctx, "train.py", "train.py")]);
    build_sft_step(&ctx, set).unwrap();
    assert_eq!(exam_seal::trained(&ctx).unwrap(), families(&["train.py"]));
    assert_eq!(exam_seal::reserved(&ctx).unwrap(), families(&["exam.py"]));
}
