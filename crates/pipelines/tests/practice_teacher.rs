// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that learn tasks they cannot yet
// solve from verified demonstrations, for its clients. If your team needs
// expertise in curricula for code-editing agents you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: a repository task the policy never solves is not dropped when a
//! teacher is named; it is learned from a demonstration that the same
//! independent verifier passed.
//!
//! * with no teacher, `practice` keeps nothing from a policy that solves
//!   nothing (the zero-success problem);
//! * the task's reference change, replayed as the actions of an agent with
//!   sven's own tools, is a teacher's experience, graded by the workspace
//!   verifier, and the never-solved task it answers is kept and trained on;
//! * a stronger model solving the task as an agent is the same teacher kind
//!   by another name, graded the same way;
//! * the teacher never grades itself: a demonstration that does not make the
//!   hidden checks pass is not kept, whoever proposed it;
//! * no exam task is taught.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::collections::BTreeSet;

use common::gate::{put_base, FakeTrainer};
use common::repo::{project, Mender};
use common::{config, Scratch};
use splinter_agent::demonstration::DEMONSTRATOR;
use splinter_agent::solve::Model;
use splinter_agent::work::WorkProtocol;
use splinter_core::clock::FixedClock;
use splinter_core::experience::{Privileged, PrivilegedKind, Task};
use splinter_core::model_ref::ModelRef;
use splinter_orchestrator::Context;
use splinter_pipelines::curriculum::frontier::PassAtK;
use splinter_pipelines::curriculum::repository_teacher::{
    teach_repository, RepositoryTeacher, TeachRepositoryRequest,
};
use splinter_pipelines::practice::{practice, PracticeReport, PracticeRequest};
use splinter_store::runs::read_run;
use splinter_store::tasks::{TaskEntry, TaskSet};

/// A context whose policy never changes anything, and whose stronger model
/// restores every module.
fn context(test: &str) -> (Scratch, Context) {
    let scratch = Scratch::new(test);
    let ctx = Context::new(config(&scratch), false)
        .unwrap()
        .with_clock(Box::new(FixedClock::new("2026-10-19T08:00:00.000Z")))
        .with_model(
            ModelRef::policy_default(),
            Model::new(Mender::new(false), "scripted/idle-1"),
        )
        .with_model(strong(), Model::new(Mender::new(true), "scripted/strong-1"));
    put_base(&ctx);
    (scratch, ctx)
}

fn strong() -> ModelRef {
    "local:strong".parse().unwrap()
}

fn request(repo: &std::path::Path, teacher: Option<RepositoryTeacher>) -> PracticeRequest {
    PracticeRequest {
        repository: repo.to_path_buf(),
        revision: "HEAD".into(),
        max_tasks: Some(32),
        mutants_per_file: Some(1),
        exam_families: Some(1),
        pass_at_k: PassAtK::new(2, None, None),
        no_release: true,
        teacher,
        ..PracticeRequest::default()
    }
}

fn run(ctx: &Context, request: &PracticeRequest) -> (String, PracticeReport) {
    let recorded = practice(ctx, request, &FakeTrainer::knowing(&[])).unwrap();
    (recorded.run, recorded.report)
}

#[test]
fn without_a_teacher_a_policy_that_solves_nothing_leaves_nothing_to_train_on() {
    let (scratch, ctx) = context("teacher-none");
    let repo = scratch.0.join("repo");
    project(&repo, &[]);
    let (_, report) = run(&ctx, &request(&repo, None));
    assert!(report.teach.is_none());
    assert!(report.dataset.is_none(), "{report:#?}");
    assert!(
        report
            .stopped
            .as_deref()
            .is_some_and(|why| why.contains("never solved")),
        "{:?}",
        report.stopped
    );
}

#[test]
fn a_reference_demonstration_the_verifier_passed_teaches_a_task_nobody_solved() {
    let (scratch, ctx) = context("teacher-reference");
    let repo = scratch.0.join("repo");
    project(&repo, &[]);
    let (run_id, report) = run(&ctx, &request(&repo, Some(RepositoryTeacher::Reference)));
    assert_eq!(report.stopped, None, "{report:#?}");

    let recorded = read_run(ctx.workspace(), &run_id).unwrap();
    let stages: Vec<&str> = recorded.stages.iter().map(|s| s.stage.as_str()).collect();
    assert!(
        stages.windows(2).any(|w| w == ["verify", "teach"]),
        "{stages:?}"
    );

    // The policy solved none; every task it was given was demonstrated and
    // the demonstration passed the workspace verifier.
    let reserve = report.reserve.as_ref().unwrap();
    let taught = report.teach.as_ref().unwrap();
    assert_eq!(taught.solve.solver, DEMONSTRATOR);
    assert!(taught.solve.teacher);
    assert_eq!(taught.solve.solved, reserve.training_tasks, "{taught:#?}");
    assert_eq!(taught.verify.passed, reserve.training_tasks, "{taught:#?}");
    let distribution = report.frontier.as_ref().unwrap().distribution;
    assert_eq!(distribution.taught, reserve.training_tasks);
    assert_eq!(distribution.never, 0);

    // What was kept is the teacher's experience: a trajectory of sven's own
    // tools, marked as the teacher's, of no exam task.
    let exam: BTreeSet<_> = ctx
        .tasks()
        .get_set(reserve.exam_task_set.as_ref().unwrap())
        .unwrap()
        .members
        .into_iter()
        .map(|e| e.task)
        .collect();
    let demonstrations = ctx
        .experiences()
        .get_set(&taught.solve.experience_set)
        .unwrap()
        .members;
    assert!(!demonstrations.is_empty());
    for id in demonstrations {
        let experience = ctx.experiences().get(&id).unwrap();
        assert!(experience.provenance.teacher);
        assert_eq!(experience.provenance.solver, DEMONSTRATOR);
        assert!(
            !exam.contains(&experience.task.id),
            "an exam task was taught"
        );
        let calls: Vec<String> = experience
            .trajectory
            .steps
            .iter()
            .flat_map(|s| s.tool_calls.iter().flatten())
            .map(|c| c.function_name.clone())
            .collect();
        assert_eq!(calls, ["read_file", "edit_file"], "{id}");
    }
    let dataset = report.dataset.as_ref().unwrap();
    assert!(dataset.records > 0, "{dataset:#?}");
    splinter_model::train::validate_dataset(&dataset.path).unwrap();
    let candidate = report.candidate.as_ref().unwrap();
    // What the demonstrations and the training cost is on record, for a
    // choice between this and another way of improving.
    let cost = splinter_pipelines::improve::training_seconds(&ctx, &candidate.candidate).unwrap();
    assert!(
        cost.is_some(),
        "the run that trained a candidate says what it cost"
    );
}

#[test]
fn a_stronger_model_solving_the_task_is_graded_like_the_student() {
    let (scratch, ctx) = context("teacher-model");
    let repo = scratch.0.join("repo");
    project(&repo, &[]);
    let teacher = RepositoryTeacher::Model(strong());
    let (_, report) = run(&ctx, &request(&repo, Some(teacher)));
    assert_eq!(report.stopped, None, "{report:#?}");
    let taught = report.teach.as_ref().unwrap();
    assert_eq!(taught.solve.solver, "scripted/strong-1");
    assert!(taught.solve.teacher);
    let training = report.reserve.as_ref().unwrap().training_tasks;
    assert_eq!(taught.verify.passed, training, "{taught:#?}");
    assert_eq!(
        report.frontier.as_ref().unwrap().distribution.taught,
        training
    );
}

/// The tasks of a practice run's training set with each reference change
/// replaced by the one of the next task: a teacher that proposes a change
/// that does not fix the task it is asked about.
fn mismatched(ctx: &Context, report: &PracticeReport) -> splinter_store::tasks::TaskSetId {
    let set = report.tasks.as_ref().unwrap().task_set.as_ref().unwrap();
    let members = ctx.tasks().get_set(set).unwrap().members;
    let reference = |task: &Task| {
        task.privileged
            .iter()
            .find(|p| p.kind == PrivilegedKind::Reference)
            .unwrap()
            .content
            .clone()
    };
    let tasks: Vec<Task> = members
        .iter()
        .map(|m| ctx.tasks().get(&m.task).unwrap())
        .collect();
    let swapped: Vec<TaskEntry> = tasks
        .iter()
        .enumerate()
        .map(|(i, task)| {
            let other = reference(&tasks[(i + 1) % tasks.len()]);
            let privileged = task
                .privileged
                .iter()
                .map(|p| match p.kind {
                    PrivilegedKind::Reference => Privileged {
                        content: other.clone(),
                        ..p.clone()
                    },
                    _ => p.clone(),
                })
                .collect();
            let made = Task::new(
                task.task.kind.clone(),
                task.evidence.clone(),
                task.environment.clone(),
                &task.instruction,
                privileged,
            )
            .unwrap()
            .with_concepts(task.concepts.clone())
            .unwrap();
            ctx.tasks().put(&made).unwrap();
            TaskEntry {
                task: made.task.id,
                ..members[i].clone()
            }
        })
        .collect();
    ctx.tasks()
        .put_set(&TaskSet {
            name: "references that do not fit".into(),
            members: swapped,
        })
        .unwrap()
}

#[test]
fn a_demonstration_that_does_not_make_the_hidden_checks_pass_is_not_kept() {
    let (scratch, ctx) = context("teacher-wrong");
    let repo = scratch.0.join("repo");
    project(&repo, &[]);
    let (_, report) = run(&ctx, &request(&repo, None));
    let wrong = mismatched(&ctx, &report);

    let taught = teach_repository(
        &ctx,
        &TeachRepositoryRequest {
            task_set: &wrong,
            attempts: None,
            teacher: &RepositoryTeacher::Reference,
            work: WorkProtocol::default(),
            deadline: None,
            cancel: splinter_agent::CancelToken::new(),
        },
    )
    .unwrap();
    assert!(taught.solve.solved > 0, "{taught:#?}");
    assert_eq!(
        taught.verify.passed, 0,
        "the verifier, not the teacher, decides: {taught:#?}"
    );
}
