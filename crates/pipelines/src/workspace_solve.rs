// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements coding agents that change a repository
// without ever seeing the tests that judge them, for its clients. If your
// team needs expertise in measuring what a code-editing agent really did,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! What the solve stage adds for a task that is a change to a repository:
//! the archive its snapshot is laid out from, the checks the task shows its
//! solver, the seed an attempt draws from, and the episode itself.
//!
//! The patch the worker leaves in its tree is the solve's final output.
//! The checks the solver is shown are the task's guarding commands
//! (`pass_to_pass`), which run over tests that are in its tree; the commands
//! that must go from failing to passing run over the hidden tests and never
//! reach the solver, not even as feedback.

use splinter_agent::solve::{Model, Solution, SolveError, SolveOptions};
use std::time::Instant;

use splinter_agent::harness_work::solve_harness_in_workspace;
use splinter_agent::work::{solve_in_workspace, WorkOptions, WorkProtocol};
use splinter_core::digest::Digest;
use splinter_core::experience::Task;
use splinter_core::outcome::AttemptReport;
use splinter_eval::verifiers::workspace::WorkspaceCheck;
use splinter_sandbox::{CommandCall, WorkspaceEnvironment};
use std::path::PathBuf;

use splinter_orchestrator::context::Context;

/// What a solve in a workspace needs besides the task and its environment.
#[derive(Clone, Debug)]
pub(crate) struct WorkspaceInputs {
    archive: PathBuf,
    visible: Vec<CommandCall>,
}

impl WorkspaceInputs {
    /// The inputs for `task`, a change to the repository `env` is a
    /// snapshot of; the reason when it cannot be solved: its archive is not
    /// kept, or its check is not one.
    pub(crate) fn of(
        ctx: &Context,
        env: &WorkspaceEnvironment,
        task: &Task,
    ) -> Result<Self, String> {
        let archive = ctx
            .artifacts()
            .path(&env.snapshot().digest)
            .map_err(|e| format!("the snapshot of the task's workspace is not kept: {e}"))?;
        let check = WorkspaceCheck::of_task(task)
            .map_err(|e| format!("the task's workspace check is unusable: {e}"))?;
        Ok(Self {
            archive,
            visible: check
                .map(|(_, check)| check.pass_to_pass)
                .unwrap_or_default(),
        })
    }
}

/// The seed of the draws of attempt `attempt` at task `task`: a function of
/// the two alone, so the same attempt at the same task decodes the same way
/// on the same weights, and another attempt, or another task, draws
/// differently.
#[must_use]
pub(crate) fn attempt_seed(task: &Digest, attempt: u32) -> u64 {
    let digest = Digest::of(format!("{task}/{attempt}").as_bytes());
    // The first 16 hex digits of the address: 64 bits.
    u64::from_str_radix(&digest.hex()[..16], 16).unwrap_or_default()
}

/// One attempt at `task` in the workspace `env`: the worker's episode under
/// `protocol`, as a solution whose final output is the patch, and the facts
/// of the episode the solution does not carry.
pub(crate) async fn solve_attempt(
    task: &Task,
    env: &WorkspaceEnvironment,
    inputs: &WorkspaceInputs,
    model: &Model,
    options: SolveOptions,
    protocol: &WorkProtocol,
) -> Result<(Solution, AttemptReport), SolveError> {
    let started = Instant::now();
    let solved = solve_in_workspace(
        task,
        env,
        &inputs.archive,
        &inputs.visible,
        model.provider.clone(),
        WorkOptions::new(options),
        protocol,
    )
    .await?;
    let report = solved.report.attempt_report(started.elapsed());
    Ok((solved.solution, report))
}

/// One attempt at `task` by the sven harness `harness` in place of the agent
/// loop: [`solve_attempt`]'s result for the same environment, so the attempt
/// is verified and stored as any other.
pub(crate) async fn solve_harness_attempt(
    task: &Task,
    env: &WorkspaceEnvironment,
    inputs: &WorkspaceInputs,
    harness: &str,
    model: &Model,
    options: SolveOptions,
) -> Result<(Solution, AttemptReport), SolveError> {
    let started = Instant::now();
    let solved = solve_harness_in_workspace(
        task,
        env,
        &inputs.archive,
        harness,
        model.provider.clone(),
        WorkOptions::new(options),
    )
    .await?;
    let report = solved.report.attempt_report(started.elapsed());
    Ok((solved.solution, report))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_attempt_has_a_seed_of_its_own_and_the_same_one_again() {
        let task = Digest::of(b"a task");
        let other = Digest::of(b"another task");
        assert_eq!(attempt_seed(&task, 0), attempt_seed(&task, 0));
        assert_ne!(attempt_seed(&task, 0), attempt_seed(&task, 1));
        assert_ne!(attempt_seed(&task, 0), attempt_seed(&other, 0));
    }
}
