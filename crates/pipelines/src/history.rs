// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements executable tasks derived from a repository's
// own history, for its clients. If your team needs expertise in turning a
// codebase's commits into verified training tasks, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Reading tasks from a repository's history as the rest of the pipeline
//! reads them: the knowledge layer's history source under the workspace
//! generator this state root provides, held against every task an earlier
//! run stored, and, when asked, kept as a task set.
//!
//! The same source `practice --history` selects; this module is the entry a
//! command that has no rollout calls.

use splinter_knowledge::tasks::workspace::{
    HistoryGeneration, HistoryRequest, WorkspaceGenerator, WorkspacePlan,
};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_store::tasks::TaskSetId;

use crate::practice::held::{held_tasks, keep_tasks};

/// What a history run admitted and where it was stored.
pub struct HistoryRun {
    /// The admitted tasks and the funnel.
    pub generation: HistoryGeneration,
    /// The task set the admitted tasks were stored in, when asked for.
    pub task_set: Option<TaskSetId>,
}

/// The workspace generator over this state root, under `plan`.
pub(crate) fn generator(
    ctx: &Context,
    plan: WorkspacePlan,
) -> Result<WorkspaceGenerator, OrchestratorError> {
    let backend = ctx.environments().workspace()?;
    WorkspaceGenerator::new(
        backend.sandbox,
        backend.host,
        ctx.artifacts(),
        backend.scratch.join("generate"),
    )
    .with_plan(plan)
    .map_err(|e| refused("the task generator", &e))
}

/// Reads `request`'s history, adds no task an earlier run holds, and stores
/// the admitted tasks as a task set when `store` is set.
pub fn generate(
    ctx: &Context,
    request: &HistoryRequest<'_>,
    store: bool,
) -> Result<HistoryRun, OrchestratorError> {
    let generator = generator(ctx, WorkspacePlan::default())?;
    let held = held_tasks(ctx)?;
    let generation = generator
        .from_history(request, &held, ctx.clock())
        .map_err(|e| refused("making tasks from the history", &e))?;
    let task_set = if !store || generation.admitted.is_empty() {
        None
    } else {
        let tasks: Vec<_> = generation.admitted.iter().map(|t| t.task.clone()).collect();
        Some(keep_tasks(ctx, request.range, &tasks)?)
    };
    Ok(HistoryRun {
        generation,
        task_set,
    })
}

fn refused(what: &str, e: &dyn std::fmt::Display) -> OrchestratorError {
    OrchestratorError::Refused(format!("{what}: {e}"))
}
