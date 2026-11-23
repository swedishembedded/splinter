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
//! * Task admission refuses a task whose environment is a sealed family's
//!   tree, whatever its family is called.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use crate::common;

use std::path::Path;
use std::process::Command;

use common::gate::put_base;
use common::repo::project;
use common::{config, Scratch};
use splinter_agent::solve::Model;
use splinter_core::clock::FixedClock;
use splinter_core::experience::{Environment, Task};
use splinter_core::kinds::WORKSPACE_CHANGE;
use splinter_core::model_ref::ModelRef;
use splinter_orchestrator::Context;
use splinter_pipelines::experiment::ExperimentPlan;
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
