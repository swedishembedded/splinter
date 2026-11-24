// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements leakage-free held-out splits of
// code-editing task sets, for its clients. If your team needs expertise in
// holding out whole task families before any training data is built, you
// can procure our services by sending an email to info@swedishembedded.com.

//! `tasks combine`: harder tasks made of the admitted single-fault tasks of
//! a stored set.
//!
//! The tasks are made by [`splinter_knowledge::tasks::workspace`] (see
//! `WorkspaceGenerator::combine` for what a combined task is and what it is
//! admitted on). This stage stores them: a combined task's family joins the
//! families of the files it holds a fault of ([`splinter_core::family`]), and
//! a task of a family an experiment plan seals is refused ([`crate::sealed`]).
//!
//! One set is written, of every combined task. The members of each combined task are kept as a document
//! (`combined_task`), outside the task, so its address is that of any
//! workspace task with the same content.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use splinter_core::digest::Digest;
use splinter_core::rng::Seed;
use splinter_knowledge::tasks::workspace::{
    CombinePlan, CombineReason, WorkspaceGenerator, COMBINED_GENERATOR,
};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_store::tasks::{TaskEntry, TaskSet, TaskSetId};

/// The class the record of a combined task is kept under.
const COMBINED_TASK: &str = "combined_task";

/// What `tasks combine` is asked.
#[derive(Clone, Debug)]
pub struct CombineRequest {
    /// The set of admitted single-fault tasks, of one snapshot.
    pub task_set: TaskSetId,
    /// How many faults a task holds, how often a file takes part, how many
    /// tasks in all, and the seed the choice is a function of.
    pub plan: CombinePlan,
}

/// A combined task and what it was made of, as it is kept.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CombinedRecord {
    /// The combined task's address.
    pub task: Digest,
    /// Its family: the sorted member families joined.
    pub family: String,
    /// The single tasks it is made of, in family order.
    pub members: Vec<Digest>,
}

/// What `tasks combine` did.
#[derive(Clone, Debug, Serialize)]
pub struct Combined {
    /// The seed the choice was made by.
    pub seed: Seed,
    /// Faults in each task.
    pub faults: usize,
    /// The most tasks any one file takes part in.
    pub per_family: usize,
    /// Every combined task admitted; `None` when none was.
    pub task_set: Option<TaskSetId>,
    /// Combined tasks admitted.
    pub admitted: usize,
    /// Combinations tried.
    pub attempted: usize,
    /// Combinations not admitted, by reason.
    pub refused: BTreeMap<CombineReason, usize>,
    /// Single tasks that could not be combined at all, by reason.
    pub unusable: BTreeMap<CombineReason, usize>,
    /// Admitted tasks of each family.
    pub families: BTreeMap<String, usize>,
    /// Each admitted task and its members.
    pub tasks: Vec<CombinedRecord>,
}

/// Combines the tasks of `request.task_set` (see the module documentation).
///
/// # Errors
///
/// [`OrchestratorError`] when the set names a sealed family, the plan does
/// not fit the tasks (fewer files than faults), or the workspace backend or
/// the stores fail.
pub fn combine(ctx: &Context, request: &CombineRequest) -> Result<Combined, OrchestratorError> {
    crate::sealed::admit_set(ctx, &request.task_set)?;
    let store = ctx.tasks();
    let tasks = store
        .get_set(&request.task_set)?
        .members
        .iter()
        .map(|entry| store.get(&entry.task))
        .collect::<Result<Vec<_>, _>>()?;
    let backend = ctx.environments().workspace()?;
    let generator = WorkspaceGenerator::new(
        backend.sandbox,
        backend.host,
        ctx.artifacts(),
        backend.scratch.join("combine"),
    );
    let made = generator
        .combine(&tasks, &request.plan, ctx.clock())
        .map_err(|e| OrchestratorError::Refused(e.to_string()))?;

    let sealed = crate::sealed::families(ctx)?;
    let mut all = Vec::new();
    let mut records = Vec::new();
    let mut families: BTreeMap<String, usize> = BTreeMap::new();
    for combined in made.admitted {
        if let Some(why) = crate::sealed::refuses_task(&sealed, &combined.task.task) {
            return Err(OrchestratorError::Refused(why));
        }
        let entry = TaskEntry {
            task: store.put(&combined.task.task)?,
            generator: Some(COMBINED_GENERATOR.to_string()),
            prompt: Some(combined.task.key.clone()),
            variant_of: None,
            subject: None,
        };
        let record = CombinedRecord {
            task: entry.task.clone(),
            family: combined.task.family.clone(),
            members: combined.members,
        };
        ctx.workspace().put_document(COMBINED_TASK, &record)?;
        *families.entry(record.family.clone()).or_default() += 1;
        all.push(entry);
        records.push(record);
    }
    let name = |what: &str| format!("{what} (from {})", request.task_set);
    let put =
        |what: &str, members: &Vec<TaskEntry>| -> Result<Option<TaskSetId>, OrchestratorError> {
            if members.is_empty() {
                return Ok(None);
            }
            Ok(Some(store.put_set(&TaskSet {
                name: name(what),
                members: members.clone(),
            })?))
        };
    Ok(Combined {
        seed: request.plan.seed,
        faults: request.plan.faults,
        per_family: request.plan.per_family,
        task_set: put("combined workspace tasks", &all)?,
        admitted: records.len(),
        attempted: made.attempted,
        refused: made.refused,
        unusable: made.unusable,
        families,
        tasks: records,
    })
}
