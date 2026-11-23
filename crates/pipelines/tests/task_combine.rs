// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements leakage-free held-out splits of
// code-editing task sets, for its clients. If your team needs expertise in
// holding out whole task families before any training data is built, you
// can procure our services by sending an email to info@swedishembedded.com.

//! Spec: `tasks combine` makes tasks of several faults from the admitted
//! tasks of a practised repository, and keeps the exam sealed.
//!
//! * a combined task whose faults include a file of the exam is an exam task,
//!   never a training one, however many of its files are training files;
//! * the exam set and the training set partition the combined tasks;
//! * combining again with the same seed makes the same sets;
//! * a plan that needs more files than the tasks lie in is refused by name.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::collections::BTreeSet;

use common::gate::put_base;
use common::repo::{project, Agent};
use common::{config, workspace_tasks, Scratch};
use splinter_agent::solve::Model;
use splinter_core::clock::FixedClock;
use splinter_core::family::{members, touches};
use splinter_core::model_ref::ModelRef;
use splinter_core::rng::Seed;
use splinter_knowledge::tasks::workspace::CombinePlan;
use splinter_orchestrator::Context;
use splinter_pipelines::tasks::combine::{combine, CombineRequest};
use splinter_store::tasks::TaskSetId;

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

/// A repository made into tasks: its admitted set and the families its exam holds.
fn practised(test: &str) -> (Scratch, Context, TaskSetId, BTreeSet<String>) {
    let (scratch, ctx) = context(test);
    let repo = scratch.0.join("repo");
    project(&repo, &[]);
    let (set, exam) = workspace_tasks::practised(&ctx, &repo, 2);
    (scratch, ctx, set, exam)
}

fn request(set: &TaskSetId) -> CombineRequest {
    CombineRequest {
        task_set: set.clone(),
        plan: CombinePlan {
            faults: 2,
            per_family: 2,
            max_tasks: 5,
            // seed: the second run must make the same sets as the first
            seed: Seed::new(5),
        },
    }
}

#[test]
fn a_task_with_a_fault_in_an_exam_file_is_an_exam_task_and_the_rest_may_be_trained_on() {
    let (_scratch, ctx, set, exam) = practised("task-combine-exam");
    assert_eq!(exam.len(), 2);

    let made = combine(&ctx, &request(&set)).unwrap();
    assert!(made.admitted > 0, "{made:#?}");
    assert_eq!(made.exam_tasks + made.training_tasks, made.admitted);
    assert!(made.exam_tasks > 0 && made.training_tasks > 0, "{made:#?}");

    // The rule, task by task: any exam file makes an exam task.
    for record in &made.tasks {
        assert_eq!(record.members.len(), 2);
        assert_eq!(
            record.exam,
            touches(&record.family, &exam).is_some(),
            "{} is {} of the exam",
            record.family,
            if record.exam { "part" } else { "no part" }
        );
        let task = ctx.tasks().get(&record.task).unwrap();
        assert_eq!(task.concepts, std::slice::from_ref(&record.family));
        assert_eq!(members(&record.family).count(), 2);
    }

    // The two sets partition the whole one, and no training task draws on an
    // exam file.
    let ids = |set: &Option<TaskSetId>| -> BTreeSet<_> {
        set.as_ref()
            .map(|s| ctx.tasks().get_set(s).unwrap().members)
            .unwrap_or_default()
            .into_iter()
            .map(|m| m.task)
            .collect()
    };
    let (all, in_exam, in_training) = (
        ids(&made.task_set),
        ids(&made.exam_task_set),
        ids(&made.training_task_set),
    );
    assert!(in_exam.is_disjoint(&in_training));
    assert_eq!(&in_exam | &in_training, all);
    for task in &in_training {
        let family = ctx.tasks().get(task).unwrap().concepts[0].clone();
        assert_eq!(touches(&family, &exam), None, "{family} was trained on");
    }

    // Again with the same seed, the same sets.
    let again = combine(&ctx, &request(&set)).unwrap();
    assert_eq!(made.task_set, again.task_set);
    assert_eq!(made.exam_task_set, again.exam_task_set);
    assert_eq!(made.training_task_set, again.training_task_set);
}

#[test]
fn a_set_of_one_file_cannot_be_combined_and_the_refusal_names_the_families() {
    let (_scratch, ctx, set, _) = practised("task-combine-one-file");
    let mut asked = request(&set);
    asked.plan.faults = 99;
    let refused = combine(&ctx, &asked).unwrap_err().to_string();
    assert!(refused.contains("families"), "{refused}");
}
