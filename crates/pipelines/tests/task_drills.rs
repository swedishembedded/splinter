// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements leakage-free held-out splits of
// code-editing task sets, for its clients. If your team needs expertise in
// holding out whole task families before any training data is built, you
// can procure our services by sending an email to info@swedishembedded.com.

//! Spec: `tasks drills` makes comprehension drills from a practised
//! repository and keeps the exam sealed.
//!
//! * a drill of a file the exam reserved is an exam drill, a drill of any
//!   other file a training drill, and the two sets partition the drills;
//! * the same seed makes the same sets;
//! * an answer to an exam drill cannot be built into a training set, and
//!   one to a training drill can, which remembers its file as trained on;
//! * with `for_tasks`, only the functions that contain a line a repair task's
//!   reference change changes are drilled, and a reference change that is not
//!   of the revision is refused;
//! * drills export as Harbor tasks like the other tasks of a set.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::collections::BTreeSet;

use common::gate::put_base;
use common::repo::{git, project, write, Agent};
use common::{config, workspace_tasks, Scratch};
use splinter_agent::solve::Model;
use splinter_core::clock::FixedClock;
use splinter_core::model_ref::ModelRef;
use splinter_core::rng::Seed;
use splinter_knowledge::tasks::workspace::DrillOrigin;
use splinter_orchestrator::Context;
use splinter_pipelines::harbor::export_harbor;
use splinter_pipelines::tasks::drills::{drills, DrillRequest, Drilled};
use splinter_store::tasks::TaskSetId;

fn context(test: &str) -> (Scratch, Context) {
    let scratch = Scratch::new(test);
    let ctx = Context::new(config(&scratch), false)
        .unwrap()
        .with_clock(Box::new(FixedClock::new("2026-11-06T08:00:00.000Z")))
        .with_model(
            ModelRef::policy_default(),
            Model::new(Agent::new(), "scripted/agent-1"),
        );
    put_base(&ctx);
    (scratch, ctx)
}

/// A repository made into tasks: its admitted set.
fn practised(test: &str) -> (Scratch, Context, TaskSetId) {
    let (scratch, ctx) = context(test);
    let repo = scratch.0.join("repo");
    project(&repo, &[]);
    let set = workspace_tasks::practised(&ctx, &repo);
    (scratch, ctx, set)
}

fn request(scratch: &Scratch, seed: u64) -> DrillRequest {
    DrillRequest {
        repository: scratch.0.join("repo"),
        revision: "HEAD".into(),
        allow_dirty: false,
        per_function: 2,
        // seed: the second run must draw the same drills as the first
        seed: Seed::new(seed),
        files: Vec::new(),
        for_tasks: None,
    }
}

fn members(ctx: &Context, set: &Option<TaskSetId>) -> Vec<splinter_core::experience::Task> {
    set.as_ref()
        .map(|s| ctx.tasks().get_set(s).unwrap().members)
        .unwrap_or_default()
        .into_iter()
        .map(|m| ctx.tasks().get(&m.task).unwrap())
        .collect()
}

#[test]
fn every_drill_is_named_by_the_file_it_exercises_and_the_same_seed_makes_the_same_set() {
    let (scratch, ctx, _) = practised("task-drills");

    let made = drills(&ctx, &request(&scratch, 7)).unwrap();
    assert!(made.admitted > 0, "{made:#?}");
    let all = members(&ctx, &made.task_set);
    assert_eq!(all.len(), made.admitted);
    for task in &all {
        let origin = DrillOrigin::of_task(task).unwrap();
        assert_eq!(task.concepts[0], origin.file);
    }
    assert_eq!(made.families.values().sum::<usize>(), made.admitted);

    // Again with the same seed, the same set.
    let again = drills(&ctx, &request(&scratch, 7)).unwrap();
    assert_eq!(made.task_set, again.task_set);
}

fn lines_of(drilled: &Drilled) -> BTreeSet<String> {
    drilled.drilled.keys().cloned().collect()
}

#[test]
fn drills_for_tasks_cover_exactly_the_functions_their_reference_changes_touch() {
    let (scratch, ctx, tasks) = practised("task-drills-for-tasks");
    let mut asked = request(&scratch, 7);
    asked.for_tasks = Some(tasks.clone());
    let made = drills(&ctx, &asked).unwrap();
    let scope = made.scope.clone().unwrap();
    assert!(scope.tasks > 0 && scope.without_reference == 0, "{scope:?}");
    assert!(!scope.functions.is_empty(), "{scope:?}");
    // Every drill is of a function a change lies in, and of nothing else.
    assert!(
        lines_of(&made).is_subset(&scope.functions.iter().cloned().collect()),
        "{made:#?}"
    );
    assert!(made.admitted > 0, "{made:#?}");

    // The same drills of the whole repository are more than those.
    let whole = drills(&ctx, &request(&scratch, 7)).unwrap();
    assert!(whole.admitted >= made.admitted);
}

#[test]
fn a_reference_change_made_for_another_revision_is_refused_naming_the_revision() {
    let (scratch, ctx, tasks) = practised("task-drills-other-revision");
    let repo = scratch.0.join("repo");
    for n in 0..common::repo::MODULES {
        let path = repo.join(format!("shop/tier{n}.py"));
        let text = std::fs::read_to_string(&path).unwrap();
        write(&path, &format!("# moved down a line\n{text}"));
    }
    git(&repo, &["add", "-A"]);
    git(&repo, &["commit", "-q", "-m", "shift"]);
    let mut asked = request(&scratch, 7);
    asked.for_tasks = Some(tasks);
    let refused = drills(&ctx, &asked).unwrap_err().to_string();
    assert!(refused.contains("drill the revision"), "{refused}");
}

#[test]
fn drills_export_as_harbor_tasks_like_the_other_tasks_of_a_set() {
    let (scratch, ctx, _) = practised("task-drills-harbor");
    let made = drills(&ctx, &request(&scratch, 7)).unwrap();
    let out = scratch.0.join("harbor");
    let exports = export_harbor(&ctx, &made.task_set.unwrap(), &out, "python:3.12-slim").unwrap();
    assert_eq!(exports.written.len(), made.admitted);
    for exported in &exports.written {
        for file in [
            "instruction.md",
            "task.toml",
            "tests/test.sh",
            "tests/expected_output.txt",
        ] {
            assert!(exported.directory.join(file).is_file(), "{file}");
        }
    }
}
