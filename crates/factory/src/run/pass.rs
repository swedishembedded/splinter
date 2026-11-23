// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! A rollout pass as an arm runs it: an `attempt.collect` stage's
//! parameters with its model role resolved and the stages whose solved
//! tasks it leaves out, or an evaluation's arm.

use std::collections::BTreeMap;

use serde::Serialize;
use splinter_core::factory::trial::{RunBudget, TimeoutPolicy};

use super::{stage_error, Factory, RunError};
use crate::ops::{AttemptParams, Hint};
use crate::plan::PlannedStage;
use crate::recipe::ModelRef;

/// Which tasks a pass takes.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct TaskSelector {
    /// Only the tasks with no verified success in any of these stages,
    /// frozen when the pass starts.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub unsolved_by: Vec<String>,
    /// Only these task ids.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub ids: Vec<String>,
}

/// A pass as an arm runs it: an `attempt.collect` stage's parameters with
/// its model resolved and the stages whose solved tasks it leaves out, or an
/// evaluation's arm.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RolloutPass {
    /// The slot namespace of its attempts (the stage's id).
    pub name: String,
    /// The agent adapter.
    pub agent: String,
    /// The agent's own arguments.
    pub args: Vec<String>,
    /// Files the arguments refer to.
    pub files: Vec<String>,
    /// The model, unless the agent brings its own.
    pub model: Option<ModelRef>,
    /// A system prompt file appended to the agent's own.
    pub system_prompt: Option<String>,
    /// Environment variables for the agent.
    pub environment_variables: BTreeMap<String, String>,
    /// The environment file.
    pub environment: String,
    /// Which tasks.
    pub tasks: TaskSelector,
    /// Attempts per task.
    pub attempts: u32,
    /// Attempts run at once.
    pub concurrency: u32,
    /// The per-trial budget.
    pub budget: Option<RunBudget>,
    /// The timeout policy.
    pub timeout_policy: TimeoutPolicy,
    /// Extra attempts the infrastructure's failures may take.
    pub retries: u32,
    /// The seed attempt seeds are drawn from.
    pub seed: u64,
    /// Played by people.
    pub interactive: bool,
    /// A hinted pass.
    pub hint: Option<Hint>,
}

/// The pass an `attempt.collect` stage runs.
pub(crate) fn pass(factory: &Factory<'_>, stage: &PlannedStage) -> Result<RolloutPass, RunError> {
    let params: AttemptParams = stage.params().map_err(|e| stage_error(&stage.key, e))?;
    let model =
        params
            .model
            .as_deref()
            .map(|role| {
                factory.resolved.model(role).cloned().ok_or_else(|| {
                    stage_error(&stage.key, format!("model role {role} is not bound"))
                })
            })
            .transpose()?;
    Ok(RolloutPass {
        name: stage.key.clone(),
        agent: params.agent,
        args: params.args,
        files: params.files,
        model,
        system_prompt: params.system_prompt,
        environment_variables: params.environment_variables,
        environment: params.environment,
        tasks: TaskSelector {
            unsolved_by: stage
                .sources("unsolved_by")
                .into_iter()
                .map(str::to_owned)
                .collect(),
            ids: params.tasks,
        },
        attempts: params.attempts,
        concurrency: params.concurrency,
        budget: params.budget,
        timeout_policy: params.timeout_policy,
        retries: params.retries,
        seed: params.seed,
        interactive: params.interactive,
        hint: params.hint,
    })
}

/// The `tasks.pin` stage feeding `stage`.
pub(crate) fn pin(stage: &PlannedStage) -> Result<&str, RunError> {
    stage
        .source("tasks")
        .ok_or_else(|| stage_error(&stage.key, "its tasks come from no stage"))
}
