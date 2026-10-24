// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training-set curation that keeps a long
// verified trajectory in the dataset instead of dropping it, for its
// clients. If your team needs expertise in turning long agent trajectories
// into trainable windows then you can procure our services by sending an
// email to info@swedishembedded.com.

//! Spec: `dataset build` splits a record over the training block only when
//! it is asked to, and the unasked build keeps leaving it out.
//!
//! * With `split_over_block` false, a record longer than `--max-block` is
//!   still left out and counted as `OverBlock`.
//! * With it true, the same record is split into windows that each fit, and
//!   nothing is counted as `OverBlock`.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

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
use splinter_data::{chars_as_tokens, Exclusion, Record};
use splinter_model::TokenCounter;
use splinter_orchestrator::Context;
use splinter_pipelines::datasets::{project_with, BuildRequest, ViewName, VoiceBuild};
use splinter_store::experiences::{ExperienceSet, SetId};

const INSTRUCTION: &str = "Make the failing test pass.";

/// An action that edits `path` and is answered with `ok`.
fn edit(step: u64, call: &str, path: &str) -> TraceStep {
    let mut step = TraceStep::new(step, StepOrigin::Agent, "");
    step.tool_calls =
        Some(vec![ToolInvocation::new(call, "edit")
            .with_arguments(json!({"path": path, "text": "fixed"}))]);
    step.observation = Some(StepObservation::single(ObservationEntry::for_call(
        call, "ok",
    )));
    step
}

/// Two actions and a final answer: a trajectory long enough that its packed
/// record can be split into more than one window.
fn attempt(ctx: &Context) -> Experience {
    let task = Task::new(
        WORKSPACE_CHANGE,
        Vec::new(),
        Environment::new(Environment::WORKSPACE, json!({"snapshot": "tree-1"})),
        INSTRUCTION,
        Vec::new(),
    )
    .unwrap();
    let mut trajectory = Trajectory::new("ATIF-v1.7", AgentProfile::new("scripted", "1"));
    trajectory.steps = vec![
        TraceStep::new(1, StepOrigin::User, INSTRUCTION),
        edit(2, "c1", "a.py"),
        edit(3, "c2", "b.py"),
        TraceStep::new(4, StepOrigin::Agent, "Done."),
    ];
    let experience = Experience::new(
        task,
        trajectory,
        Some("PATCH".into()),
        Provenance::new(
            "scripted/solver-1",
            &FixedClock::new("2026-10-24T01:00:00.000Z"),
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

fn set_of(ctx: &Context, experience: &Experience) -> SetId {
    ctx.experiences()
        .put_set(&ExperienceSet {
            name: "attempts".into(),
            members: vec![experience.id().unwrap()],
        })
        .unwrap()
}

fn request(set: SetId, max_block: Option<usize>, split_over_block: bool) -> BuildRequest {
    BuildRequest {
        sets: vec![set],
        view: ViewName::SftStep,
        strip: None,
        min_strength: Some(Strength::Executable),
        system_prompt: None,
        export_only: false,
        limit: None,
        voice: VoiceBuild::default(),
        max_block,
        pack: true,
        control: Default::default(),
        serialise_calls: false,
        split_over_block,
    }
}

/// The bound the build measures a record by, as the build itself does.
fn measure(ctx: &Context) -> impl Fn(&Record) -> usize {
    let counter = TokenCounter::for_model(&ctx.config().policy_base).ok();
    move |record: &Record| {
        let text = serde_json::to_string(&record.body).unwrap_or_default();
        counter
            .as_ref()
            .map_or_else(|| chars_as_tokens(&text), |c| c.count(&text))
    }
}

#[test]
fn a_build_splits_an_over_block_record_only_when_asked() {
    let (scratch, ctx) = scratch_context("dataset-split", Scripted::new(|_| String::new()), false);
    std::mem::forget(scratch);
    let experience = attempt(&ctx);
    let set = set_of(&ctx, &experience);

    let (unbounded, _) = project_with(&ctx, &request(set.clone(), None, false), None).unwrap();
    assert_eq!(unbounded.records.len(), 1, "one record per trajectory");
    let measure = measure(&ctx);
    let bound = measure(&unbounded.records[0]) - 1;

    let (kept, _) = project_with(&ctx, &request(set.clone(), Some(bound), false), None).unwrap();
    assert!(kept.records.is_empty(), "unasked, the record is left out");
    assert_eq!(kept.count(Exclusion::OverBlock), 1);

    let (split, _) = project_with(&ctx, &request(set, Some(bound), true), None).unwrap();
    assert!(split.records.len() > 1, "asked, the record is split");
    assert_eq!(split.count(Exclusion::OverBlock), 0);
    for record in &split.records {
        assert!(measure(record) <= bound, "every window fits the bound");
    }
}
