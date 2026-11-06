// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reinforcement learning from the verified
// results of agents working in a repository, for its clients. If your team
// needs expertise in training code-editing agents against hidden tests, you
// can procure our services by sending an email to info@swedishembedded.com.

//! Spec: `practice` trains by the objective it is asked for.
//!
//! * the default is supervised imitation of the passing attempts, and a run
//!   that names nothing is unchanged: the supervised trainer, no rollout
//!   report;
//! * `policy-gradient` makes the k attempts at a task a group through the
//!   canonical policy-gradient view (reward: a verified pass), trains with
//!   the clipped objective and brain's defaults for it, samples the
//!   importance ratio at the temperature the attempts were drawn at, leaves
//!   the attempts the exposure audit failed out and counts them, and reports
//!   the groups by kind, the ones the frontier would drop included;
//! * `self-distill` also distils the groups whose attempts all failed, with
//!   what the workspace verifier knows, in a teacher context the student's
//!   own messages never hold;
//! * a request the objective cannot honour is refused before any attempt.

#![allow(clippy::unwrap_used)]

mod common;

use std::path::Path;

use common::gate::{put_base, FakeTrainer};
use common::repo::{project, Agent, Mender, MODULES, PEEKING};
use common::{config, Scratch};
use splinter_agent::solve::Model;
use splinter_core::clock::FixedClock;
use splinter_core::model_ref::ModelRef;
use splinter_core::training::Regime;
use splinter_model::policy_gradient::PolicyGradientOptions;
use splinter_orchestrator::Context;
use splinter_pipelines::curriculum::frontier::{PassAtK, DEFAULT_SAMPLING};
use splinter_pipelines::practice::{practice, PracticeObjective, PracticeReport, PracticeRequest};
use splinter_pipelines::train::Tuning;
use sven_sdk::model::ModelProvider;

/// A context over a scratch state root whose `policy:default` is `agent`.
fn context(test: &str, agent: std::sync::Arc<dyn ModelProvider>) -> (Scratch, Context) {
    let scratch = Scratch::new(test);
    let ctx = Context::new(config(&scratch), false)
        .unwrap()
        .with_clock(Box::new(FixedClock::new("2026-11-06T08:00:00.000Z")))
        .with_model(
            ModelRef::policy_default(),
            Model::new(agent, "scripted/agent-1"),
        );
    put_base(&ctx);
    (scratch, ctx)
}

fn request(repo: &Path, objective: PracticeObjective) -> PracticeRequest {
    PracticeRequest {
        repository: repo.to_path_buf(),
        revision: "HEAD".into(),
        max_tasks: Some(32),
        mutants_per_file: Some(1),
        exam_families: Some(1),
        pass_at_k: PassAtK::new(2, None, None),
        no_release: true,
        objective,
        ..PracticeRequest::default()
    }
}

fn run(ctx: &Context, request: &PracticeRequest, trainer: &FakeTrainer) -> PracticeReport {
    practice(ctx, request, trainer).unwrap().report
}

/// The lines of the report's dataset.
fn records(report: &PracticeReport) -> Vec<serde_json::Value> {
    std::fs::read_to_string(&report.dataset.as_ref().unwrap().path)
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

#[test]
fn a_run_that_names_no_objective_trains_supervised_exactly_as_before() {
    assert_eq!(
        PracticeRequest::default().objective,
        PracticeObjective::SftStep
    );
    let (scratch, ctx) = context("objective-default", Agent::new());
    let repo = scratch.0.join("repo");
    project(&repo, &[]);
    let trainer = FakeTrainer::knowing(&[]);
    let report = run(
        &ctx,
        &request(&repo, PracticeObjective::default()),
        &trainer,
    );

    assert_eq!(report.stopped, None, "{report:#?}");
    assert!(report.rollouts.is_none(), "{:#?}", report.rollouts);
    assert_eq!(report.dataset.as_ref().unwrap().view, "sft-step");
    assert_eq!(*trainer.called.lock().unwrap(), [Regime::Sft]);
}

#[test]
fn policy_gradient_makes_the_attempts_at_a_task_a_group_and_trains_the_clipped_objective() {
    let (scratch, ctx) = context("objective-pg", Agent::new());
    let repo = scratch.0.join("repo");
    project(&repo, &[]);
    let trainer = FakeTrainer::knowing(&[]);
    let report = run(
        &ctx,
        &request(&repo, PracticeObjective::PolicyGradient),
        &trainer,
    );
    assert_eq!(report.stopped, None, "{report:#?}");

    // The view and the trainer are the canonical ones.
    let dataset = report.dataset.as_ref().unwrap();
    assert_eq!(dataset.view, "policy-gradient");
    assert_eq!(*trainer.called.lock().unwrap(), [Regime::PolicyGradient]);
    let lines = records(&report);
    assert!(!lines.is_empty());
    assert!(
        lines.iter().all(|l| l["advantage"].is_f64()),
        "every record carries its attempt's advantage"
    );
    assert!(lines.iter().all(|l| l.get("teacher_context").is_none()));
    splinter_model::train::validate_dataset(&dataset.path).unwrap();

    // The scripted agent peeks at the hidden tests in the attempt that would
    // pass at three modules: the audit fails those attempts, they are left
    // out whole and counted, and their families are not trained on.
    let rollouts = report.rollouts.as_ref().unwrap();
    let verified = report.verify.as_ref().unwrap();
    assert_eq!(rollouts.objective, PracticeObjective::PolicyGradient);
    assert!(verified.exposed > 0);
    assert_eq!(rollouts.exposed, verified.exposed, "{rollouts:#?}");
    let solved = report.solve.as_ref().unwrap();
    assert_eq!(rollouts.attempts, solved.solved);
    let families: Vec<String> = lines
        .iter()
        .map(|l| l["metadata"]["group"].as_str().unwrap().to_string())
        .collect();
    for n in PEEKING {
        assert!(
            !families.contains(&format!("shop/tier{n}.py")),
            "{families:?}"
        );
    }

    // Every task is in a group of some kind: the groups the frontier does
    // not keep are counted, not dropped unseen.
    let groups = rollouts.groups.unwrap();
    let tasks = report.reserve.as_ref().unwrap().training_tasks;
    assert_eq!(groups.graded + groups.all_pass + groups.all_fail, tasks);
    assert!(groups.graded > 0 && groups.graded < MODULES);
    assert_eq!(groups.distilled, 0, "plain policy gradient distils nothing");

    // The trainer gets the objective's settings, brain's defaults resolved,
    // and the temperature the attempts were sampled at.
    let plans = trainer.plans.lock().unwrap();
    let defaults = PolicyGradientOptions::defaults();
    let used = plans[0].tuning.policy_gradient;
    assert_eq!(used.clip_eps, Some(defaults.clip_eps));
    assert_eq!(used.kl_beta, Some(defaults.kl_beta));
    assert_eq!(used.temperature, Some(DEFAULT_SAMPLING.temperature));
    assert_eq!(plans[0].regime, Regime::PolicyGradient);
}

#[test]
fn the_objectives_own_settings_reach_the_trainer_and_a_named_temperature_is_kept() {
    let (scratch, ctx) = context("objective-pg-flags", Agent::new());
    let repo = scratch.0.join("repo");
    project(&repo, &[]);
    let trainer = FakeTrainer::knowing(&[]);
    let asked = PracticeRequest {
        tuning: Tuning {
            policy_gradient: PolicyGradientOptions {
                clip_eps: Some(0.1),
                kl_beta: Some(0.5),
                ..PolicyGradientOptions::default()
            },
            ..Tuning::default()
        },
        ..request(&repo, PracticeObjective::PolicyGradient)
    };
    run(&ctx, &asked, &trainer);
    let used = trainer.plans.lock().unwrap()[0].tuning.policy_gradient;
    assert_eq!((used.clip_eps, used.kl_beta), (Some(0.1), Some(0.5)));

    let (scratch, ctx) = context("objective-pg-temperature", Agent::new());
    let repo = scratch.0.join("repo");
    project(&repo, &[]);
    let trainer = FakeTrainer::knowing(&[]);
    let named = PracticeRequest {
        tuning: Tuning {
            policy_gradient: PolicyGradientOptions {
                temperature: Some(0.3),
                ..PolicyGradientOptions::default()
            },
            ..Tuning::default()
        },
        ..request(&repo, PracticeObjective::PolicyGradient)
    };
    run(&ctx, &named, &trainer);
    let used = trainer.plans.lock().unwrap()[0].tuning.policy_gradient;
    assert_eq!(used.temperature, Some(0.3));
}

/// A project every attempt fails: the cliff of every task.
fn idle(test: &str) -> (Scratch, Context, std::path::PathBuf) {
    let (scratch, ctx) = context(test, Mender::new(false));
    let repo = scratch.0.join("repo");
    project(&repo, &[]);
    (scratch, ctx, repo)
}

#[test]
fn policy_gradient_learns_nothing_from_tasks_every_attempt_failed_and_says_so() {
    let (_scratch, ctx, repo) = idle("objective-pg-cliff");
    let trainer = FakeTrainer::knowing(&[]);
    let report = run(
        &ctx,
        &request(&repo, PracticeObjective::PolicyGradient),
        &trainer,
    );
    let why = report.stopped.as_deref().unwrap();
    assert!(
        why.contains("policy-gradient") && why.contains("NoGroupSignal"),
        "{why}"
    );
    assert!(trainer.called.lock().unwrap().is_empty());
    assert!(report.candidate.is_none());
    // Unmeasured is absent, not zero: no dataset, no group counts.
    assert!(report.rollouts.as_ref().unwrap().groups.is_none());
}

#[test]
fn self_distill_distils_the_tasks_every_attempt_failed_with_what_the_verifier_knows() {
    let (_scratch, ctx, repo) = idle("objective-distill");
    let trainer = FakeTrainer::knowing(&[]);
    let report = run(
        &ctx,
        &request(&repo, PracticeObjective::SelfDistill),
        &trainer,
    );
    assert_eq!(report.stopped, None, "{report:#?}");
    assert_eq!(*trainer.called.lock().unwrap(), [Regime::PolicyGradient]);

    let groups = report.rollouts.as_ref().unwrap().groups.unwrap();
    let tasks = report.reserve.as_ref().unwrap().training_tasks;
    assert_eq!(groups.all_fail, tasks, "{groups:?}");
    assert_eq!(groups.distilled, tasks);
    assert_eq!(groups.graded, 0);

    let lines = records(&report);
    assert!(!lines.is_empty());
    for line in &lines {
        assert!(line.get("advantage").is_none(), "{line}");
        let context = line["teacher_context"]["messages"][0]["content"]
            .as_str()
            .unwrap_or_else(|| panic!("no teacher context: {line}"));
        assert!(
            context.to_lowercase().contains("hidden checks"),
            "{context}"
        );
        // What the verifier knows is the teacher's, never the student's.
        assert!(!line["messages"].to_string().contains("hidden checks"));
    }
    assert_eq!(report.dataset.as_ref().unwrap().view, "policy-gradient");
}

#[test]
fn self_distill_keeps_the_scalar_advantage_where_the_attempts_differ() {
    let (scratch, ctx) = context("objective-distill-mixed", Agent::new());
    let repo = scratch.0.join("repo");
    project(&repo, &[]);
    let trainer = FakeTrainer::knowing(&[]);
    let report = run(
        &ctx,
        &request(&repo, PracticeObjective::SelfDistill),
        &trainer,
    );
    let groups = report.rollouts.as_ref().unwrap().groups.unwrap();
    assert!(groups.graded > 0, "{groups:?}");
    let lines = records(&report);
    let (distilled, scored): (Vec<_>, Vec<_>) = lines
        .iter()
        .partition(|l| l.get("teacher_context").is_some());
    assert!(!scored.is_empty());
    assert!(scored.iter().all(|l| l["advantage"].is_f64()));
    assert_eq!(distilled.is_empty(), groups.all_fail == 0, "{groups:?}");
}

#[test]
fn a_request_the_objective_cannot_honour_is_refused_before_anything_is_attempted() {
    let (scratch, ctx) = context("objective-refused", Agent::new());
    let repo = scratch.0.join("repo");
    project(&repo, &[]);
    let refused = |asked: PracticeRequest, flag: &str| {
        let error = practice(&ctx, &asked, &FakeTrainer::knowing(&[])).unwrap_err();
        assert!(
            error.is_refusal() && error.to_string().contains(flag),
            "{flag}: {error}"
        );
        assert!(ctx.sources().list().unwrap().is_empty(), "{flag}");
    };
    let pg = |objective| request(&repo, objective);

    // Imitation of a teacher, and a fresh adapter, are not what a rollout
    // objective trains.
    refused(
        PracticeRequest {
            teacher: Some("reference".parse().unwrap()),
            ..pg(PracticeObjective::PolicyGradient)
        },
        "--teacher",
    );
    refused(
        PracticeRequest {
            rebuild: true,
            ..pg(PracticeObjective::SelfDistill)
        },
        "--rebuild",
    );
    // A rollout run monitors nothing.
    refused(
        PracticeRequest {
            tuning: Tuning {
                eval_every: Some(5),
                ..Tuning::default()
            },
            ..pg(PracticeObjective::PolicyGradient)
        },
        "--eval-every",
    );
    // The objective's settings belong to a rollout objective; named beside
    // the supervised one they would be dropped.
    refused(
        PracticeRequest {
            tuning: Tuning {
                policy_gradient: PolicyGradientOptions {
                    clip_eps: Some(0.2),
                    ..PolicyGradientOptions::default()
                },
                ..Tuning::default()
            },
            ..pg(PracticeObjective::SftStep)
        },
        "--clip-eps",
    );
}
