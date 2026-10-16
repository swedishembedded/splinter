// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements release gates that cannot be passed by a
// model that games its exam, for its clients. If your team needs expertise
// in detecting reward hacking and benchmark contamination in agent
// evaluations, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Canary task sets: the workspace tasks of a stored set, each with its hidden
//! assertions also required to fail
//! ([`splinter_eval::verifiers::workspace_canary`]), stored as a set of their
//! own for the release gate to be handed ([`load`]).
//!
//! Canaries are never a variant of the task they were made from: a variant
//! is measured by the gate as held-out evidence, and a canary carries no
//! comparison, only a refusal.

use std::collections::BTreeMap;

use serde::Serialize;
use splinter_core::experience::{Environment, Task};
use splinter_eval::verifiers::workspace::WorkspaceCheck;
use splinter_eval::verifiers::workspace_canary::canary_task;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_store::tasks::{TaskEntry, TaskSet, TaskSetId};

/// The generator name canaries carry in their set.
pub const CANARY_GENERATOR: &str = "splinter-pipelines/canaries@1";

/// How many canaries the gate is handed when nothing names another number:
/// the count the cheating acceptance (a cheater passing nine in ten is
/// flagged) was pre-registered on.
pub const CANARY_COUNT: usize = 20;

/// What making canaries reported.
#[derive(Clone, Debug, Serialize)]
pub struct CanariesMade {
    /// The set of canaries (`tasks show <set>`); `None` when none was made.
    pub canary_set: Option<TaskSetId>,
    /// Canaries in it.
    pub canaries: usize,
    /// Tasks of the original set that were not made into one, by reason.
    pub skipped: BTreeMap<String, usize>,
}

/// The canaries of the workspace tasks of `set`, the first `limit` of them
/// ([`CANARY_COUNT`] when `None`), as a task set of their own. Content
/// addressed, so making them again gives the same set.
pub fn make(
    ctx: &Context,
    set: &TaskSetId,
    limit: Option<usize>,
) -> Result<CanariesMade, OrchestratorError> {
    let limit = limit.unwrap_or(CANARY_COUNT);
    let mut skipped: BTreeMap<String, usize> = BTreeMap::new();
    let mut members: Vec<TaskEntry> = Vec::new();
    for entry in ctx.tasks().get_set(set)?.members {
        if members.len() == limit {
            break;
        }
        let task = ctx.tasks().get(&entry.task)?;
        if task.environment.kind != Environment::WORKSPACE {
            *skipped.entry("not_a_workspace_task".into()).or_default() += 1;
            continue;
        }
        if !matches!(WorkspaceCheck::of_task(&task), Ok(Some(_))) {
            *skipped
                .entry("no_usable_workspace_check".into())
                .or_default() += 1;
            continue;
        }
        let canary = canary_task(&task).map_err(|e| OrchestratorError::Refused(e.to_string()))?;
        members.push(TaskEntry {
            task: ctx.tasks().put(&canary)?,
            generator: Some(CANARY_GENERATOR.into()),
            prompt: Some(entry.task),
            variant_of: None,
            subject: None,
        });
    }
    let canaries = members.len();
    let canary_set = if members.is_empty() {
        None
    } else {
        Some(ctx.tasks().put_set(&TaskSet {
            name: format!("canaries of {set}"),
            members,
        })?)
    };
    Ok(CanariesMade {
        canary_set,
        canaries,
        skipped,
    })
}

/// The tasks of the canary set `set`, in order, for a release request.
pub fn load(ctx: &Context, set: &TaskSetId) -> Result<Vec<Task>, OrchestratorError> {
    let members = ctx.tasks().get_set(set)?.members;
    if members.is_empty() {
        return Err(OrchestratorError::Refused(format!(
            "the canary set {set} holds no task"
        )));
    }
    members
        .iter()
        .map(|entry| Ok(ctx.tasks().get(&entry.task)?))
        .collect()
}
