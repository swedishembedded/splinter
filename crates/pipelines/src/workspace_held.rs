// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that learn a codebase from its own
// tests, for its clients. If your team needs expertise in training
// code-editing agents on executable tasks drawn from a repository, you can
// procure our services by sending an email to info@swedishembedded.com.

//! What earlier runs already hold, kept in the task store beside every other
//! task set: a workspace task is stored under the generator that made it,
//! with the key that identifies it across runs as the entry's prompt digest
//! (what the generator was asked to make: a change to a family of a
//! snapshot), and the family it belongs to as its declared concept. The next
//! run reads them back as [`Held`], so a task already made is not made
//! twice and a request that repeats one made for the same file is not
//! admitted again.

use std::collections::BTreeSet;

use splinter_core::digest::Digest;
use splinter_knowledge::tasks::workspace::{Held, WorkspaceTask, WORKSPACE_GENERATOR};
use splinter_store::tasks::{TaskEntry, TaskSet, TaskSetId};

use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

/// Everything the task store holds that the workspace generator made, less
/// the tasks of a sealed final-exam family: a seal recorded after a task was
/// stored keeps that task out of every later run.
pub(crate) fn held_tasks(ctx: &Context) -> Result<Held, OrchestratorError> {
    let store = ctx.tasks();
    let sealed = crate::sealed::families(ctx)?;
    let mut held = Held::none();
    let mut seen: BTreeSet<Digest> = BTreeSet::new();
    for set in store.list_sets()? {
        for entry in store.get_set(&set)?.members {
            let (Some(WORKSPACE_GENERATOR), Some(key)) =
                (entry.generator.as_deref(), entry.prompt.clone())
            else {
                continue;
            };
            if !seen.insert(entry.task.clone()) {
                continue;
            }
            let task = store.get(&entry.task)?;
            let Some(family) = task.concepts.first() else {
                continue;
            };
            // The recorded seal is by tree id and family id; a task of a
            // sealed family is not a task this run may read.
            if crate::sealed::refuses_task(&sealed, &task).is_some() {
                continue;
            }
            held.insert(key, family, &task.instruction);
        }
    }
    Ok(held)
}

/// Stores `tasks` and the set they make, named for `commit`; a task of a
/// sealed family is refused, as task admission.
pub(crate) fn keep_tasks(
    ctx: &Context,
    commit: &str,
    tasks: &[WorkspaceTask],
) -> Result<TaskSetId, OrchestratorError> {
    let store = ctx.tasks();
    let sealed = crate::sealed::families(ctx)?;
    let mut members = Vec::with_capacity(tasks.len());
    for made in tasks {
        if let Some(why) = crate::sealed::refuses_task(&sealed, &made.task) {
            return Err(OrchestratorError::Refused(why));
        }
        members.push(TaskEntry {
            task: store.put(&made.task)?,
            generator: Some(WORKSPACE_GENERATOR.to_string()),
            prompt: Some(made.key.clone()),
            variant_of: None,
            subject: None,
        });
    }
    Ok(store.put_set(&TaskSet {
        name: format!("workspace tasks of {commit}"),
        members,
    })?)
}
