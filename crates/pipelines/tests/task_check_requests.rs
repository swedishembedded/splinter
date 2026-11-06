// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solvable, leakage-free code-editing task
// sets for training and measuring agents, for its clients. If your team needs
// expertise in curating benchmark tasks that a solver can actually solve from
// their request, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: `tasks check-requests` keeps, unchanged, the tasks whose request
//! states the contract their reference change restores, drops and counts the
//! ones whose request does not, and leaves the source set as it was.

#![allow(clippy::unwrap_used)]

mod common;

use common::gate::{gate_context, Brain};
use splinter_core::experience::{Environment, Privileged, PrivilegedKind, Task};
use splinter_pipelines::tasks::check_requests::check_requests;
use splinter_store::tasks::{TaskEntry, TaskSet};

fn request(title: &str) -> String {
    format!(
        "Some behaviour of this repository is broken. The tests that check it are not in the \
         tree you were given; they fail on it, and each names a symptom:\n\n- {title}\n\nFind the \
         cause in the repository's code and fix it so that this behaviour works again."
    )
}

fn task(title: &str, reference: &str) -> Task {
    Task::new(
        "workspace-change",
        Vec::new(),
        Environment::closed_book(),
        request(title),
        vec![Privileged {
            kind: PrivilegedKind::Reference,
            content: reference.into(),
            span: None,
        }],
    )
    .unwrap()
}

const RESTORES_RETURN: &str =
    "--- a/s.py\n+++ b/s.py\n@@ -1,2 +1,3 @@\n def restore(x):\n     put(x)\n+    return len(x)\n";

#[test]
fn tasks_with_an_unstated_contract_are_dropped_and_counted_and_the_rest_kept() {
    let (_scratch, ctx) = gate_context("task-check-requests", Brain::Honest);
    let unfair = task(
        "restore puts each entry with remaining ttl",
        RESTORES_RETURN,
    );
    let fair = task("restore returns how many entries it put", RESTORES_RETURN);
    let other = task("restore puts each entry with remaining ttl", "");
    let entry = |t: &Task| TaskEntry {
        task: ctx.tasks().put(t).unwrap(),
        generator: None,
        prompt: None,
        variant_of: None,
        subject: None,
    };
    let members = vec![entry(&unfair), entry(&fair), entry(&other)];
    let kept_ids: Vec<_> = members[1..].iter().map(|m| m.task.clone()).collect();
    let set = ctx
        .tasks()
        .put_set(&TaskSet {
            name: "seeded".into(),
            members,
        })
        .unwrap();

    let checked = check_requests(&ctx, &set).unwrap();

    assert_eq!(checked.kept_tasks, 2);
    assert_eq!(checked.unstated_return_value, 1);
    assert_eq!(checked.unstated_raised_error, 0);
    let mut kept: Vec<_> = ctx
        .tasks()
        .get_set(&checked.kept)
        .unwrap()
        .members
        .into_iter()
        .map(|m| m.task)
        .collect();
    let mut expected = kept_ids;
    kept.sort_by_key(ToString::to_string);
    expected.sort_by_key(ToString::to_string);
    assert_eq!(kept, expected, "the kept tasks are the same tasks");
    assert_eq!(ctx.tasks().get_set(&set).unwrap().members.len(), 3);
}
