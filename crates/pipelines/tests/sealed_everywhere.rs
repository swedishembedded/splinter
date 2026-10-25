// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements examinations that stay sealed across every
// run of a learning pipeline, for its clients. If your team needs expertise
// in protecting a held-out final exam from the stages that read a repository,
// you can procure our services by sending an email to info@swedishembedded.com.

//! Spec: a sealed final-exam family is sealed in the state root, not only in
//! the plan this run was given.
//!
//! * A plan that is validated or accepted writes its sealed families (their
//!   ids and tree ids) to the state root.
//! * From then on a run without a plan refuses the sealed repository before
//!   it is taken in as a source, by its tree id and never by the name the
//!   catalog happens to use now.
//! * A family the plan did not seal is read as before.
//! * Task admission refuses a task whose environment is a sealed family's
//!   tree, whatever its family is called.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::collections::BTreeSet;
use std::path::Path;
use std::process::Command;

use common::gate::{put_base, FakeTrainer};
use common::repo::project;
use common::{config, Scratch};
use splinter_agent::solve::Model;
use splinter_core::clock::FixedClock;
use splinter_core::experience::{Environment, Task};
use splinter_core::kinds::WORKSPACE_CHANGE;
use splinter_core::model_ref::ModelRef;
use splinter_orchestrator::Context;
use splinter_pipelines::curriculum::frontier::PassAtK;
use splinter_pipelines::experiment::ExperimentPlan;
use splinter_pipelines::practice::{practice, PlanSlot, PracticeRequest};
use splinter_pipelines::sealed;
use splinter_store::tasks::{TaskEntry, TaskSet};

fn context(test: &str) -> (Scratch, Context) {
    let scratch = Scratch::new(test);
    let ctx = Context::new(config(&scratch), false)
        .unwrap()
        .with_clock(Box::new(FixedClock::new("2026-10-13T08:00:00.000Z")))
        .with_model(
            ModelRef::policy_default(),
            Model::new(
                std::sync::Arc::new(common::Scripted::new(|_| String::new())),
                "scripted/agent-1",
            ),
        );
    put_base(&ctx);
    (scratch, ctx)
}

fn request(repo: &Path) -> PracticeRequest {
    PracticeRequest {
        repository: repo.to_path_buf(),
        revision: "HEAD".into(),
        max_tasks: Some(1),
        mutants_per_file: Some(1),
        exam_families: Some(0),
        pass_at_k: PassAtK::new(2, None, None),
        no_release: true,
        ..PracticeRequest::default()
    }
}

fn tree_of(repo: &Path) -> String {
    let out = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(["rev-parse", "HEAD^{tree}"])
        .output()
        .unwrap();
    assert!(out.status.success());
    String::from_utf8(out.stdout).unwrap().trim().to_string()
}

/// A plan whose generation 1 trains the repository at `train` and whose
/// sealed final exam is the repository at `sealed`, under the family ids
/// `trained_as` and `sealed_as`.
fn plan_for(train: &Path, trained_as: &str, sealed: &Path, sealed_as: &str) -> ExperimentPlan {
    let catalog = format!(
        "[[family]]\nid = \"{trained_as}\"\nlanguage = \"python\"\ndomain = \"training\"\n\
         tree = \"{}\"\n\
         [[family]]\nid = \"{sealed_as}\"\nlanguage = \"python\"\ndomain = \"exam\"\n\
         tree = \"{}\"\nenvironment = \"snap-exam\"\n",
        tree_of(train),
        tree_of(sealed)
    );
    let plan = format!(
        "[plan]\nname = \"p\"\n[[generation]]\ntrain = [\"{trained_as}\"]\nretention = []\n\
         [sealed]\nfamilies = [\"{sealed_as}\"]\n"
    );
    ExperimentPlan::parse(&plan, &catalog).unwrap()
}

#[test]
fn a_run_without_the_plan_cannot_read_a_sealed_family_once_the_seal_is_recorded() {
    let (scratch, ctx) = context("sealed-no-plan");
    let train = scratch.0.join("train");
    let exam = scratch.0.join("exam");
    project(&train, &[("train_only.py", "")]);
    project(&exam, &[]);
    sealed::record(&ctx, &plan_for(&train, "w", &exam, "y")).unwrap();
    assert_eq!(
        sealed::trees(&ctx).unwrap(),
        BTreeSet::from([tree_of(&exam)])
    );

    // A run given no plan at all still refuses the exam's repository, before
    // it is taken in as a source and before a task is made of it.
    let refused = practice(&ctx, &request(&exam), &FakeTrainer::knowing(&[])).unwrap_err();
    assert!(
        refused.is_refusal()
            && refused.to_string().contains("sealed")
            && refused.to_string().contains("y"),
        "{refused}"
    );
    assert!(
        ctx.sources().list().unwrap().is_empty(),
        "a sealed repository was taken in as a source"
    );
    assert!(
        ctx.tasks().list().unwrap().is_empty(),
        "a task was made of the sealed repository"
    );
}

#[test]
fn an_unsealed_family_is_unaffected_by_the_seal() {
    let (scratch, ctx) = context("sealed-unaffected");
    let train = scratch.0.join("train");
    let exam = scratch.0.join("exam");
    project(&train, &[("train_only.py", "")]);
    project(&exam, &[]);
    sealed::record(&ctx, &plan_for(&train, "w", &exam, "y")).unwrap();

    let recorded = practice(&ctx, &request(&train), &FakeTrainer::knowing(&[])).unwrap();
    assert!(
        recorded
            .report
            .tasks
            .as_ref()
            .is_some_and(|t| t.admitted > 0),
        "{:#?}",
        recorded.report
    );
}

#[test]
fn renaming_a_family_does_not_open_it() {
    let (scratch, ctx) = context("sealed-rename");
    let train = scratch.0.join("train");
    let exam = scratch.0.join("exam");
    project(&train, &[("train_only.py", "")]);
    project(&exam, &[]);
    // The seal is recorded under the name the catalog uses today.
    sealed::record(&ctx, &plan_for(&train, "w", &exam, "y")).unwrap();
    // A later plan gives the same tree another name and trains it. The
    // recorded seal is by tree, so the repository stays shut.
    sealed::record(&ctx, &plan_for(&exam, "renamed", &train, "other")).unwrap();

    let refused = practice(&ctx, &request(&exam), &FakeTrainer::knowing(&[])).unwrap_err();
    assert!(
        refused.is_refusal(),
        "a renamed sealed family was opened: {refused}"
    );
    assert!(ctx.sources().list().unwrap().is_empty());
}

#[test]
fn a_practice_given_a_plan_writes_its_seal_for_every_later_run() {
    let (scratch, ctx) = context("sealed-accepted");
    let train = scratch.0.join("train");
    let exam = scratch.0.join("exam");
    project(&train, &[("train_only.py", "")]);
    project(&exam, &[]);
    let plan = plan_for(&train, "w", &exam, "y");
    assert_eq!(sealed::trees(&ctx).unwrap(), BTreeSet::new());

    // Accepting the plan is the run that records it.
    let accepted = practice(
        &ctx,
        &PracticeRequest {
            experiment: Some(PlanSlot {
                plan,
                generation: 1,
            }),
            ..request(&train)
        },
        &FakeTrainer::knowing(&[]),
    )
    .unwrap();
    assert!(accepted.report.tasks.is_some(), "{:#?}", accepted.report);
    assert_eq!(
        sealed::trees(&ctx).unwrap(),
        BTreeSet::from([tree_of(&exam)])
    );

    // A later run without the plan still cannot read the exam.
    let refused = practice(&ctx, &request(&exam), &FakeTrainer::knowing(&[])).unwrap_err();
    assert!(refused.is_refusal(), "{refused}");
}

#[test]
fn task_admission_refuses_a_task_of_a_sealed_tree_whatever_its_family_is_called() {
    let (scratch, ctx) = context("sealed-task");
    let train = scratch.0.join("train");
    let exam = scratch.0.join("exam");
    project(&train, &[("train_only.py", "")]);
    project(&exam, &[]);
    sealed::record(&ctx, &plan_for(&train, "w", &exam, "y")).unwrap();

    let task = Task::new(
        WORKSPACE_CHANGE,
        Vec::new(),
        Environment::new(
            Environment::WORKSPACE,
            serde_json::json!({"tree_id": tree_of(&exam), "snapshot": "tree-1"}),
        ),
        "Make the failing test pass.",
        Vec::new(),
    )
    .unwrap()
    .with_concepts(["renamed-family.py"])
    .unwrap();
    let refused = sealed::admit_task(&ctx, &task).unwrap_err();
    assert!(
        refused.is_refusal() && refused.to_string().contains("sealed"),
        "{refused}"
    );

    // The same task set, admitted through the store, is refused as a set.
    let id = ctx
        .tasks()
        .put(&task)
        .and_then(|id| {
            ctx.tasks().put_set(&TaskSet {
                name: "sealed tasks".into(),
                members: vec![TaskEntry {
                    task: id,
                    generator: None,
                    prompt: None,
                    variant_of: None,
                    subject: None,
                }],
            })
        })
        .unwrap();
    let refused = sealed::admit_set(&ctx, &id).unwrap_err();
    assert!(refused.is_refusal(), "{refused}");

    // A task of the training repository is admitted.
    let open = Task::new(
        WORKSPACE_CHANGE,
        Vec::new(),
        Environment::new(
            Environment::WORKSPACE,
            serde_json::json!({"tree_id": tree_of(&train), "snapshot": "tree-2"}),
        ),
        "Make the failing test pass.",
        Vec::new(),
    )
    .unwrap();
    sealed::admit_task(&ctx, &open).unwrap();
    // A file a task declares under a sealed family id is refused too, even
    // when its environment names no tree.
    let by_name = Task::new(
        WORKSPACE_CHANGE,
        Vec::new(),
        Environment::closed_book(),
        "Say what is broken.",
        Vec::new(),
    )
    .unwrap()
    .with_concepts(["y"])
    .unwrap();
    assert!(sealed::admit_task(&ctx, &by_name).is_err());
    // A task in the snapshot the catalog named for the sealed family is
    // refused even when its tree id is not the one recorded.
    let by_snapshot = Task::new(
        WORKSPACE_CHANGE,
        Vec::new(),
        Environment::new(
            Environment::WORKSPACE,
            serde_json::json!({"tree_id": "another-tree", "snapshot": "snap-exam"}),
        ),
        "Make the failing test pass.",
        Vec::new(),
    )
    .unwrap();
    assert!(sealed::admit_task(&ctx, &by_snapshot).is_err());
}
