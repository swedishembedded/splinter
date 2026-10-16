// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements release gates that cannot be passed by a
// model that games its exam, for its clients. If your team needs expertise
// in detecting reward hacking and benchmark contamination in agent
// evaluations, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: canaries are made from the workspace tasks of a stored task set, as
//! a task set of their own that the release gate is handed. Each is the
//! original task with every hidden assertion also required to fail; tasks
//! that are not workspace tasks are counted and left out; making them again
//! gives the same set.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use common::gate::{fact, gate_context, Brain};
use serde_json::json;
use splinter_core::experience::{Environment, Task};
use splinter_eval::verifiers::workspace::WorkspaceCheck;
use splinter_pipelines::canaries::{load, make};
use splinter_sandbox::CommandCall;
use splinter_store::tasks::{TaskEntry, TaskSet};

fn workspace_task(n: usize) -> Task {
    let check = WorkspaceCheck {
        fail_to_pass: vec![CommandCall {
            argv: ["python3", "-m", "unittest", "test_hidden"]
                .map(String::from)
                .to_vec(),
            stdin: None,
        }],
        pass_to_pass: Vec::new(),
        hidden: None,
        protected: Vec::new(),
        hidden_paths: Vec::new(),
        hidden_strength: None,
    };
    Task::new(
        "workspace-change",
        Vec::new(),
        Environment::new(Environment::WORKSPACE, json!({ "tree": n })),
        format!("Make the failing tests pass ({n})."),
        vec![check.as_check().unwrap()],
    )
    .unwrap()
}

fn entry(task: &Task, ctx: &splinter_orchestrator::Context) -> TaskEntry {
    TaskEntry {
        task: ctx.tasks().put(task).unwrap(),
        generator: None,
        prompt: None,
        variant_of: None,
        subject: None,
    }
}

#[test]
fn canaries_are_the_workspace_tasks_with_their_checks_inverted_and_others_are_counted() {
    let (_scratch, ctx) = gate_context("canary-sets", Brain::Honest);
    let originals = [workspace_task(1), workspace_task(2), workspace_task(3)];
    let mut members = vec![entry(&fact("alpha", 0), &ctx)];
    members.extend(originals.iter().map(|t| entry(t, &ctx)));
    let set = ctx
        .tasks()
        .put_set(&TaskSet {
            name: "mixed".into(),
            members,
        })
        .unwrap();

    let made = make(&ctx, &set, Some(2)).unwrap();
    assert_eq!(made.canaries, 2, "{made:#?}");
    assert_eq!(made.skipped.get("not_a_workspace_task"), Some(&1));
    let canaries = load(&ctx, &made.canary_set.clone().unwrap()).unwrap();
    assert_eq!(canaries.len(), 2);
    for (canary, original) in canaries.iter().zip(&originals) {
        assert_ne!(canary.task.id, original.task.id);
        assert_eq!(canary.instruction, original.instruction);
        let (_, check) = WorkspaceCheck::of_task(canary).unwrap().unwrap();
        assert_eq!(
            check.fail_to_pass.len(),
            2,
            "each assertion and its inversion"
        );
    }

    let again = make(&ctx, &set, Some(2)).unwrap();
    assert_eq!(
        again.canary_set, made.canary_set,
        "making them again is a no-op"
    );
}

#[test]
fn a_set_with_no_workspace_task_makes_no_canaries() {
    let (_scratch, ctx) = gate_context("canary-sets-none", Brain::Honest);
    let set = ctx
        .tasks()
        .put_set(&TaskSet {
            name: "facts".into(),
            members: vec![entry(&fact("alpha", 0), &ctx)],
        })
        .unwrap();
    let made = make(&ctx, &set, None).unwrap();
    assert_eq!((made.canaries, made.canary_set), (0, None));
}
