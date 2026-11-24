// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements curricula that spend a learner's solver and
// training budget where it still learns, for its clients. If your team needs
// expertise in curriculum design or continual learning, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: the curriculum spends solver and training budget where learning
//! happens.
//!
//! * Before its attempts become training data, a task's pass@k is
//!   measured: k solves, each graded by the task's own verifiers and kept
//!   as an experience. A task the policy never solves is solved once more
//!   by a teacher, open-book, shown the task's grounding material. A task
//!   the policy always solves is dropped (no signal), one it solves
//!   sometimes is kept, one it never solves is kept when the teacher's
//!   answer is verified (taught) and dropped otherwise (nothing to learn
//!   from); the distribution is reported and the measurement recorded per
//!   task.
//! * Concept mastery is tallied per release from the policy's recorded
//!   solves, and `status` lists the weakest concepts.
//! * A retention suite the gate saw drop queues its forgotten concepts for
//!   new tasks, which the next `learn` generates from.
//! * A training set is deduplicated and capped per concept, kind and
//!   strength, deterministically (the selection's own spec is
//!   `curriculum_quotas.rs`).
//!
//! Models are scripted, so every pass@k outcome is controlled.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use crate::common;

use crate::common::solve_set;
use common::gate::releases::released;
use common::gate::{anchor_file, fact, gate_context, knower, Brain, ANCHOR};
use splinter_agent::CancelToken;
use splinter_core::experience::Task;
use splinter_core::model_ref::ModelRef;
use splinter_core::release::ReleaseId;
use splinter_knowledge::concepts::Concept;
use splinter_orchestrator::Context;
use splinter_pipelines::curriculum::mastery::weakest;
use splinter_pipelines::graded::Verdict;
use splinter_pipelines::release::anchor;
use splinter_pipelines::solving::{harness_protocol, Solved};
use splinter_pipelines::status::status;
use splinter_pipelines::verify::{verify_set, Grading};
use splinter_store::tasks::{TaskEntry, TaskSet, TaskSetId};

/// Stores `tasks` as a task set.
fn task_set(ctx: &Context, tasks: &[Task]) -> TaskSetId {
    let members = tasks
        .iter()
        .map(|task| TaskEntry {
            task: ctx.tasks().put(task).unwrap(),
            generator: None,
            prompt: None,
            variant_of: None,
            subject: None,
        })
        .collect();
    ctx.tasks()
        .put_set(&TaskSet {
            name: "curriculum spec".into(),
            members,
        })
        .unwrap()
}

/// The fact tasks of `topics`, `n` each, as one set.
fn facts(ctx: &Context, topics: &[&str], n: usize) -> TaskSetId {
    let tasks: Vec<Task> = topics
        .iter()
        .flat_map(|topic| (0..n).map(move |i| fact(topic, i)))
        .collect();
    task_set(ctx, &tasks)
}

/// The set solved once by `policy`, as `solve` solves it.
fn solve_once(ctx: &Context, set: &TaskSetId, policy: &ModelRef) -> Solved {
    solve_set(
        ctx,
        set,
        policy,
        None,
        &CancelToken::new(),
        None,
        harness_protocol(),
    )
    .unwrap()
}

#[test]
fn mastery_is_tallied_per_release_and_status_lists_the_weakest() {
    let (scratch, ctx) = gate_context("curriculum-mastery", Brain::Honest);
    anchor::freeze(&ctx, &[anchor_file(&scratch.0, 4)]).unwrap();
    let policy = ModelRef::policy_default();
    ctx.add_model(policy.clone(), knower(&[ANCHOR, "gamma"]));
    let set = facts(&ctx, &["alpha", "gamma"], 3);
    let solved = solve_once(&ctx, &set, &policy);
    let verified = verify_set(
        &ctx,
        &solved.experience_set,
        Grading::ActiveJudge,
        &CancelToken::new(),
    )
    .unwrap();
    // The report keeps every graded experience, not only the failed ones.
    let count = |verdict: Verdict| {
        verified
            .items
            .iter()
            .filter(|i| i.verdict == verdict)
            .count()
    };
    assert_eq!(verified.items.len(), verified.experiences);
    assert_eq!(
        (
            count(Verdict::Passed),
            count(Verdict::Failed),
            count(Verdict::Undecided)
        ),
        (verified.passed, verified.failed, verified.undecided)
    );

    let base = weakest(&ctx, 5).unwrap();
    assert_eq!(base.policy.release, None);
    assert_eq!((base.concepts, base.measured), (2, 2));
    let names: Vec<String> = base.weakest.iter().map(|c| c.concept.to_string()).collect();
    assert_eq!(names, ["declared:alpha", "declared:gamma"]);
    assert_eq!(base.weakest[0].current.as_ref().unwrap().rate, Some(0.0));

    // A release that learned alpha; the policy now answers as it does.
    let release = released(&ctx, "alpha", &[ANCHOR, "alpha", "gamma"]);
    ctx.add_model(policy.clone(), knower(&[ANCHOR, "alpha"]));
    let solved = solve_once(&ctx, &set, &policy);
    verify_set(
        &ctx,
        &solved.experience_set,
        Grading::ActiveJudge,
        &CancelToken::new(),
    )
    .unwrap();

    let now = status(&ctx).unwrap().concepts;
    assert_eq!(now.policy.release, Some(release.clone()));
    let weakest = &now.weakest[0];
    assert_eq!(
        weakest.concept,
        Concept::Declared {
            name: "gamma".into()
        }
    );
    let current = weakest.current.as_ref().unwrap();
    assert_eq!(
        (current.release.as_ref(), current.rate),
        (Some(&release), Some(0.0))
    );
    let earlier: Vec<(Option<&ReleaseId>, Option<f64>)> = weakest
        .releases
        .iter()
        .map(|m| (m.release.as_ref(), m.rate))
        .collect();
    assert_eq!(earlier, [(None, Some(1.0)), (Some(&release), Some(0.0))]);
    let alpha = &now.weakest[1];
    assert_eq!(
        alpha.concept,
        Concept::Declared {
            name: "alpha".into()
        }
    );
    assert_eq!(alpha.current.as_ref().unwrap().rate, Some(1.0));
    assert!(
        weakest.releases.iter().all(|m| m.graded == 3),
        "each release's own attempts, never pooled: {weakest:#?}"
    );
}
