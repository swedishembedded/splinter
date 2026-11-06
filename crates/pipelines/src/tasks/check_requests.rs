// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solvable, leakage-free code-editing task
// sets for training and measuring agents, for its clients. If your team needs
// expertise in curating benchmark tasks that a solver can actually solve from
// their request, you can procure our services by sending an email to
// info@swedishembedded.com.

//! `tasks check-requests`: a task set without the tasks whose request does
//! not state the contract their hidden check enforces.
//!
//! The same rule that refuses such a task at generation
//! ([`splinter_knowledge::tasks::workspace::contract`]) is applied to the
//! stored tasks, from their request and their reference change, so a
//! collection that exists is cleaned without regenerating it. No task is
//! changed: the kept tasks are the same tasks (same ids) in a new set, and
//! the dropped ones are counted by what they leave unstated.

use serde::Serialize;
use splinter_core::experience::PrivilegedKind;
use splinter_knowledge::tasks::workspace::contract::{unstated_contract, Unstated};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_store::tasks::{TaskSet, TaskSetId};

/// What `tasks check-requests` did.
#[derive(Clone, Debug, Serialize)]
pub struct RequestsChecked {
    /// The set that was read.
    pub source: TaskSetId,
    /// The set of the tasks kept.
    pub kept: TaskSetId,
    /// Tasks in it.
    pub kept_tasks: usize,
    /// Tasks dropped because no title speaks of a value the fix makes the
    /// code return again.
    pub unstated_return_value: usize,
    /// Tasks dropped because no title speaks of an error the fix makes the
    /// code raise again.
    pub unstated_raised_error: usize,
}

/// Writes the tasks of `set` whose request states their contract as a task
/// set of their own, and counts the others.
pub fn check_requests(
    ctx: &Context,
    set: &TaskSetId,
) -> Result<RequestsChecked, OrchestratorError> {
    let source = ctx.tasks().get_set(set)?;
    let (mut kept, mut returns, mut errors) = (Vec::new(), 0, 0);
    for entry in source.members {
        let task = ctx.tasks().get(&entry.task)?;
        let reference = task
            .privileged
            .iter()
            .find(|p| p.kind == PrivilegedKind::Reference)
            .map_or("", |p| p.content.as_str());
        match unstated_contract(&task.instruction, reference) {
            None => kept.push(entry),
            Some(Unstated::ReturnValue) => returns += 1,
            Some(Unstated::RaisedError(_)) => errors += 1,
        }
    }
    let kept_tasks = kept.len();
    let kept = ctx.tasks().put_set(&TaskSet {
        name: format!("{} (requests checked)", source.name),
        members: kept,
    })?;
    Ok(RequestsChecked {
        source: set.clone(),
        kept,
        kept_tasks,
        unstated_return_value: returns,
        unstated_raised_error: errors,
    })
}
