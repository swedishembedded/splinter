// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning agents that acquire knowledge
// their model does not have yet, for its clients. If your team needs
// expertise in knowledge distillation or continual learning, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The teacher: a verified answer to a task the student never solves.
//!
//! A policy cannot learn from its own closed-book attempts what it does not
//! know: on a document it never saw, none of them passes. [`teach`] takes
//! the tasks whose graded student attempts all failed and has a teacher
//! solve each once - the policy itself unless another model is named -
//! with the task's grounding material shown before its instruction (the
//! solve stage's teacher mode, [`crate::solving`]). Each teacher's solve is
//! an experience of the task, its provenance marked the teacher's, graded
//! by the task's own verifiers (no judge), like any solve.
//!
//! A verified teacher's answer is what a never-solved task is learned
//! from: the frontier keeps the task ([`super::frontier`]), and the
//! training view shows the student the instruction alone with that answer.
//! That is privileged-information stripping: context distillation, the
//! context being what only the teacher saw. A teacher's solve never counts toward concept mastery
//! ([`super::mastery`]): it says what the material lets a model answer, not
//! what the student knows.

use std::time::Instant;

use serde::Serialize;
use splinter_record::experiences::SetId;
use splinter_record::tasks::{TaskSet, TaskSetId};
use sven_sdk::CancelToken;

use crate::context::Context;
use crate::curriculum::frontier::tally;
use crate::error::CampaignError;
use crate::model_ref::ModelRef;
use crate::solving::{solve_tasks, SamplingChoice, SolveRequest, Solved};
use crate::verify::{verify_set, Verified};

/// One teach stage's inputs and bounds.
pub struct TeachRequest<'a> {
    /// The tasks the student attempted.
    pub task_set: &'a TaskSetId,
    /// The student's graded attempts at them; `None` when it made none, and
    /// every task is taught.
    pub attempts: Option<&'a SetId>,
    /// The model that teaches.
    pub teacher: &'a ModelRef,
    /// No solve starts after this, and none runs past it.
    pub deadline: Option<Instant>,
    /// Stops the stage.
    pub cancel: CancelToken,
}

/// What the teach stage reports.
#[derive(Clone, Debug, Serialize)]
pub struct Taught {
    /// The tasks the student never solved, each given to the teacher.
    pub task_set: TaskSetId,
    /// The teacher's solves (`experience_set` is their set).
    pub solve: Solved,
    /// Their verdicts.
    pub verify: Verified,
}

/// Has `request.teacher` solve, open-book, every task of
/// `request.task_set` that the student's graded attempts never solved, and
/// grades each solve; see the module documentation.
pub fn teach(ctx: &Context, request: &TeachRequest<'_>) -> Result<Taught, CampaignError> {
    let set = ctx.tasks().get_set(request.task_set)?;
    let (members, name) = match request.attempts {
        Some(attempts) => {
            let student = tally(ctx, attempts)?;
            let never_solved = set
                .members
                .into_iter()
                .filter(|entry| {
                    student
                        .get(&entry.task)
                        .is_some_and(|(count, _)| count.graded > 0 && count.passes == 0)
                })
                .collect();
            (
                never_solved,
                format!("never solved closed-book of {}", request.task_set),
            )
        }
        None => (set.members, format!("every task of {}", request.task_set)),
    };
    let task_set = ctx.tasks().put_set(&TaskSet { name, members })?;
    let solve = solve_tasks(
        ctx,
        &SolveRequest {
            task_set: &task_set,
            solver: request.teacher,
            attempts: 1,
            sampling: SamplingChoice::Own,
            teacher: true,
            deadline: request.deadline,
            cancel: request.cancel.clone(),
        },
    )?;
    let verify = verify_set(ctx, &solve.experience_set, None, &request.cancel)?;
    Ok(Taught {
        task_set,
        solve,
        verify,
    })
}
