// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning pipelines whose every random choice
// can be reproduced from the record of the run, for its clients. If your team
// needs expertise in experiment provenance for fine-tuning pipelines, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: a random choice a pipeline makes is a function of a seed the run
//! states, and the artifact it decides records it.
//!
//! * The denoise tasks of a task set are built from a seed the generation is
//!   given and reports; the same seed over the same text builds the same
//!   tasks, another seed other ones, and none at all is refused.
//! * The subjects of a dataset that keep their privileged context under
//!   `mix:F` are picked by the seed the policy is resolved with; the dataset's
//!   manifest records it and building again with it gives the same dataset.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use common::manual::{manual_policy, MANUAL};
use common::{scratch_context, Scratch};
use splinter_agent::CancelToken;
use splinter_core::model_ref::ModelRef;
use splinter_core::rng::Seed;
use splinter_data::Strip;
use splinter_orchestrator::Context;
use splinter_pipelines::datasets::{build, parse_strip, BuildRequest, ViewName, VoiceBuild};
use splinter_pipelines::solving::solve_set;
use splinter_pipelines::sources::{self, SourceTarget};
use splinter_pipelines::tasks::{generate, Generation, TasksGenerated};
use splinter_pipelines::verify::{verify_set, Grading};

// seed: two arbitrary seeds; only that the same one repeats and another differs matters.
const FIRST: Seed = Seed::new(21);
const OTHER: Seed = Seed::new(22);

fn manual(test: &str) -> (Scratch, Context, splinter_core::source::SourceId) {
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
    (scratch, ctx, source)
}

fn tasks_of(
    ctx: &Context,
    source: &splinter_core::source::SourceId,
    kinds: &[String],
    denoise_seed: Option<Seed>,
) -> Result<TasksGenerated, splinter_orchestrator::OrchestratorError> {
    generate(
        ctx,
        &Generation {
            sources: std::slice::from_ref(source),
            sections: &[],
            kinds,
            generator: &ModelRef::policy_default(),
            goal: None,
            author: None,
            denoise_seed,
            deadline: None,
            cancel: CancelToken::new(),
        },
    )
}

#[test]
fn denoise_tasks_are_built_from_a_seed_the_generation_reports() {
    let (_scratch, ctx, source) = manual("denoise-seed");
    let kinds = ["denoise".to_string()];

    let refused = tasks_of(&ctx, &source, &kinds, None).unwrap_err();
    assert!(refused.is_refusal(), "{refused}");
    assert!(refused.to_string().contains("seed"), "{refused}");

    let made = tasks_of(&ctx, &source, &kinds, Some(FIRST)).unwrap();
    assert_eq!(made.denoise_seed, Some(FIRST));
    let again = tasks_of(&ctx, &source, &kinds, made.denoise_seed).unwrap();
    assert_eq!(again.task_set, made.task_set);
    let other = tasks_of(&ctx, &source, &kinds, Some(OTHER)).unwrap();
    assert_ne!(other.task_set, made.task_set);

    // No denoise task, no seed to state.
    let recall = tasks_of(&ctx, &source, &["recall".to_string()], None).unwrap();
    assert_eq!(recall.denoise_seed, None);
}

#[test]
fn a_mixed_dataset_records_the_seed_that_picked_its_subjects_and_rebuilds_from_it() {
    let (_scratch, ctx, source) = manual("mix-seed");
    let kinds = ["recall".to_string()];
    let tasks = tasks_of(&ctx, &source, &kinds, None).unwrap();
    let solved = solve_set(
        &ctx,
        &tasks.task_set,
        &ModelRef::policy_default(),
        None,
        &CancelToken::new(),
        None,
    )
    .unwrap();
    verify_set(
        &ctx,
        &solved.experience_set,
        Grading::ActiveJudge,
        &CancelToken::new(),
    )
    .unwrap();
    let built = |seed: Seed| {
        let strip = parse_strip("mix:0.5")
            .unwrap()
            .with_seed::<()>(|| Ok(seed))
            .unwrap();
        let built = build(
            &ctx,
            &BuildRequest {
                sets: vec![solved.experience_set.clone()],
                view: ViewName::SftFinal,
                strip: Some(strip),
                min_strength: None,
                system_prompt: None,
                export_only: false,
                limit: None,
                voice: VoiceBuild::default(),
                max_block: None,
                pack: false,
                control: Default::default(),
            },
        )
        .unwrap();
        let manifest = ctx.datasets().get(&built.dataset).unwrap().manifest;
        (built.dataset, manifest.strip)
    };
    let (dataset, strip) = built(FIRST);
    assert!(matches!(strip, Some(Strip::Mix { seed, .. }) if seed == FIRST.value()));
    assert_eq!(built(FIRST).0, dataset);
    let (_, other) = built(OTHER);
    assert!(matches!(other, Some(Strip::Mix { seed, .. }) if seed == OTHER.value()));
}
