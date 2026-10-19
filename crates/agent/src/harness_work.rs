// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements coding agents that change a repository
// without ever seeing the tests that judge them, for its clients. If your
// team needs expertise in measuring what a code-editing agent really did,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! A candidate sven harness run on a task that changes a repository.
//!
//! [`solve_harness_in_workspace`] is [`crate::work::solve_in_workspace`] with a
//! programmable harness in place of the conversational agent loop: the same
//! fresh copy of the task's snapshot, the same confinement (sven's file tools
//! resolve inside the tree, the shell is the sandbox's), the same withheld
//! tools, and the same result - a [`WorkspaceSolution`] whose final output is
//! the patch read back from git. An attempt by a harness therefore flows
//! through the verification and storage that attempts by the agent loop do,
//! and what the harness claims to have done is never evidence: the tree is.
//!
//! The harness document is given the task's instruction as its `task` input
//! when it declares one. Documents are compiled by sven; one that does not
//! compile is an error, not a failed attempt.

use std::sync::Arc;

use serde_json::json;
use splinter_core::experience::Task;
use splinter_sandbox::workspace::DEFAULT_EXCLUDED;
use splinter_sandbox::{ResolvedEnvironment, SandboxError, WorkspaceEnvironment};
use sven_sdk::model::ModelProvider;
use sven_sdk::tool::Tool;

use crate::solve::{Solution, SolveError};
use crate::work::{engine, WorkOptions, WorkReport, WorkspaceSolution};
use crate::workspace_shell::SandboxShell;

/// The name of the input a harness may declare to receive the task's
/// instruction.
pub const TASK_INPUT: &str = "task";

/// Runs `harness` (a sven harness document) on a fresh copy of the snapshot
/// `task` records, laid out from `archive`, on `model`.
///
/// # Errors
///
/// [`SolveError::EnvironmentMismatch`] when `env` is not the environment the
/// task records, [`SolveError::Environment`] when the tree cannot be laid out
/// or read back, and [`SolveError::Engine`] when sven cannot build the
/// engine, compile the document or start the run. A run stopped by a bound,
/// or ending in the harness's own `fail`, is a solution with that
/// conclusion, whatever tree it left.
pub async fn solve_harness_in_workspace(
    task: &Task,
    env: &WorkspaceEnvironment,
    archive: &std::path::Path,
    harness: &str,
    model: Arc<dyn ModelProvider>,
    options: WorkOptions,
) -> Result<WorkspaceSolution, SolveError> {
    let offered = ResolvedEnvironment::Workspace(env.clone()).record()?;
    if offered != task.environment {
        return Err(SolveError::EnvironmentMismatch {
            task_kind: task.environment.kind.clone(),
            task_snapshot: task.environment.snapshot.clone(),
            offered_kind: offered.kind,
            offered_snapshot: offered.snapshot,
        });
    }
    let scratch = tempfile::tempdir().map_err(|source| SandboxError::Io {
        path: "temporary directory".into(),
        source,
    })?;
    let tree = scratch.path().join("tree");
    let baseline = env.materialize(archive, &tree)?;

    let confined: Vec<Arc<dyn Tool>> = vec![Arc::new(SandboxShell::new(env.clone(), tree.clone()))];
    let engine = engine(&tree, model, &options, &confined, false)?;
    let harness = engine.load_harness_from_str(harness)?;
    let declares_task = harness
        .plan()
        .inputs
        .as_ref()
        .is_some_and(|inputs| inputs.contains_key(TASK_INPUT));
    let inputs = if declares_task {
        json!({ TASK_INPUT: task.instruction })
    } else {
        json!({})
    };
    let mut run = harness.start(inputs)?;

    let outcome = run.execute(options.solve.run_options()).await?;

    let patch = {
        let (env, tree) = (env.clone(), tree.clone());
        tokio::task::spawn_blocking(move || env.diff_since(&tree, &baseline, &DEFAULT_EXCLUDED))
            .await
            .map_err(|e| SandboxError::Workspace {
                what: "reading the change back".to_string(),
                detail: e.to_string(),
            })??
    };
    let mut trajectory = run.trajectory().ok_or_else(|| SandboxError::Workspace {
        what: "reading the harness trajectory".to_string(),
        detail: "the run left none".to_string(),
    })?;
    if let serde_json::Value::Object(extra) =
        trajectory.agent.extra.get_or_insert_with(|| json!({}))
    {
        extra.insert(
            splinter_eval::exposure::WORKSPACE_TREE_KEY.into(),
            json!(tree.display().to_string()),
        );
    }
    let solution = Solution {
        trajectory,
        final_output: Some(String::from_utf8_lossy(&patch.diff).into_owned()),
        conclusion: outcome.conclusion,
        usage: outcome.usage,
    };
    Ok(WorkspaceSolution {
        report: WorkReport {
            solution: solution.clone(),
            tool_calls: u32::try_from(outcome.usage.tool_calls).unwrap_or(u32::MAX),
            stopped_by_tool_cap: false,
            stopped_by_repetition: false,
            repeated_call: None,
            dropped_events: 0,
            memory: None,
        },
        solution,
        patch,
        feedback_rounds: 0,
    })
}
