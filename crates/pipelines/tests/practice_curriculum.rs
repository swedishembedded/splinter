// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements curricula that spend a learner's GPU time on
// the task families it is still improving on, for its clients. If your team
// needs expertise in budget-limited training of code-editing agents, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: a curriculum strategy other than the frontier order adds a recorded
//! schedule stage before the solve, over the families of the tasks that are
//! left after the exam is reserved.
//!
//! * the stage sits between reserve and solve, and the run's report names
//!   every family in the order tasks are attempted, with what each weight
//!   was made of;
//! * a learning-progress draw has a recorded seed, and the same seed over
//!   the same state gives the same order;
//! * the schedule reads the competence history the store already holds: a
//!   family a release has attempted is no longer a novelty;
//! * whichever strategy ran, the report states the device time the attempts
//!   took against the tasks kept.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use common::gate::{put_base, FakeTrainer};
use common::repo::{project, Agent};
use common::{config, Scratch};
use splinter_agent::solve::Model;
use splinter_core::clock::FixedClock;
use splinter_core::model_ref::ModelRef;
use splinter_core::rng::Seed;
use splinter_orchestrator::Context;
use splinter_pipelines::curriculum::frontier::PassAtK;
use splinter_pipelines::curriculum::progress::Params;
use splinter_pipelines::curriculum::schedule::{schedule, Strategy};
use splinter_pipelines::practice::{practice, PracticeReport, PracticeRequest};
use splinter_store::runs::read_run;

fn context(test: &str) -> (Scratch, Context) {
    let scratch = Scratch::new(test);
    let ctx = Context::new(config(&scratch), false)
        .unwrap()
        .with_clock(Box::new(FixedClock::new("2026-10-13T08:00:00.000Z")))
        .with_model(
            ModelRef::policy_default(),
            Model::new(Agent::new(), "scripted/agent-1"),
        );
    put_base(&ctx);
    (scratch, ctx)
}

fn practised(
    ctx: &Context,
    repo: &std::path::Path,
    strategy: Strategy,
    seed: Option<u64>,
) -> (String, PracticeReport) {
    let recorded = practice(
        ctx,
        &PracticeRequest {
            repository: repo.to_path_buf(),
            revision: "HEAD".into(),
            max_tasks: Some(32),
            mutants_per_file: Some(1),
            exam_families: Some(1),
            // seed: both runs of the determinism test must reserve the same exam.
            exam_seed: Some(Seed::new(3)),
            pass_at_k: PassAtK::new(2, None, None),
            no_release: true,
            curriculum: strategy,
            // seed: the schedule seed under test is the caller's.
            curriculum_seed: seed.map(Seed::new),
            ..PracticeRequest::default()
        },
        &FakeTrainer::knowing(&[]),
    )
    .unwrap();
    (recorded.run, recorded.report)
}

#[test]
fn a_learning_progress_run_records_its_schedule_seed_and_yield() {
    let (scratch, ctx) = context("lp-schedule");
    let repo = scratch.0.join("repo");
    project(&repo, &[]);

    let (run, report) = practised(&ctx, &repo, Strategy::LearningProgress, Some(11));
    assert_eq!(report.stopped, None, "{report:#?}");

    let stages: Vec<String> = read_run(ctx.workspace(), &run)
        .unwrap()
        .stages
        .into_iter()
        .map(|s| s.stage)
        .collect();
    let at = |name: &str| stages.iter().position(|s| s == name).unwrap();
    assert_eq!(at("schedule"), at("reserve") + 1, "{stages:?}");
    assert_eq!(at("solve"), at("schedule") + 1, "{stages:?}");
    assert_eq!(
        read_run(ctx.workspace(), &run).unwrap().seeds["curriculum-seed"],
        11u64
    );

    let schedule = report.schedule.as_ref().unwrap();
    assert_eq!(schedule.seed.map(Seed::value), Some(11));
    assert_eq!(schedule.order.len(), schedule.families.len());
    assert!(!schedule.order.is_empty());
    // Nothing has attempted these files before: none has a progress to speak of.
    assert_eq!(schedule.informative, 0);
    let total: f64 = schedule.families.values().map(|s| s.weight).sum();
    assert!((total - 1.0).abs() < 1e-9);

    let spent = report.yield_.as_ref().unwrap();
    assert_eq!(
        spent.kept,
        report.frontier.as_ref().unwrap().distribution.kept()
    );
}

#[test]
fn the_same_seed_over_the_same_state_draws_the_same_order() {
    let orders: Vec<Vec<String>> = ["a", "b"]
        .iter()
        .map(|name| {
            let (scratch, ctx) = context(&format!("lp-seed-{name}"));
            let repo = scratch.0.join("repo");
            project(&repo, &[]);
            let (_, report) = practised(&ctx, &repo, Strategy::LearningProgress, Some(5));
            report.schedule.unwrap().order
        })
        .collect();
    assert_eq!(orders[0], orders[1]);
}

#[test]
fn the_frontier_order_adds_no_stage_and_a_missing_seed_is_drawn_and_recorded() {
    let (scratch, ctx) = context("lp-frontier");
    let repo = scratch.0.join("repo");
    project(&repo, &[]);
    let (run, report) = practised(&ctx, &repo, Strategy::Frontier, None);
    assert!(report.schedule.is_none());
    assert!(report.yield_.is_some());
    let recorded = read_run(ctx.workspace(), &run).unwrap();
    assert!(recorded.stages.iter().all(|s| s.stage != "schedule"));

    let (scratch, ctx) = context("lp-drawn");
    let repo = scratch.0.join("repo");
    project(&repo, &[]);
    let (run, report) = practised(&ctx, &repo, Strategy::LearningProgress, None);
    let seed = report.schedule.unwrap().seed.unwrap();
    assert_eq!(
        read_run(ctx.workspace(), &run).unwrap().seeds["curriculum-seed"],
        seed.value()
    );
}

#[test]
fn a_family_a_release_attempted_is_no_longer_a_novelty() {
    let (scratch, ctx) = context("lp-history");
    let repo = scratch.0.join("repo");
    project(&repo, &[]);
    let (_, report) = practised(&ctx, &repo, Strategy::Frontier, None);
    let kept = report.frontier.unwrap().frontier_task_set;
    let params = Params::default();

    let scheduled = schedule(&ctx, &kept, Strategy::GainRanked, &params, None).unwrap();
    assert!(!scheduled.report.families.is_empty());
    for (family, score) in &scheduled.report.families {
        assert!(score.novelty < params.novelty, "{family}: {score:?}");
    }
    // Ranking draws nothing; a draw without a seed is refused.
    assert_eq!(scheduled.report.seed, None);
    let refused = schedule(&ctx, &kept, Strategy::LearningProgress, &params, None).unwrap_err();
    assert!(refused.to_string().contains("seed"), "{refused}");
}
