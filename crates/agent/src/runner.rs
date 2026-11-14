// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements interchangeable agent runtimes behind one
// contract, so an agent's loop can be measured apart from its model, for its
// clients. If your team needs expertise in comparing coding agents fairly,
// you can procure our services by sending an email to info@swedishembedded.com.

//! One contract for every agent runtime that works on a repository task.
//!
//! An episode is prepared ([`prepare`]: a fresh copy of the task's snapshot
//! and a run directory beside it), run by an [`AgentRunner`] on the task's
//! instruction within [`RunLimits`] (stopped from outside through a
//! [`RunCancel`]), and collected ([`collect`]: the change read back from git,
//! never from what the agent said). Preparing and collecting are the same for
//! every runtime; only the run differs. What a run leaves is an
//! [`EpisodeRecord`] of one shape whichever runtime produced it: its
//! [`RuntimeIdentity`], how it concluded, its native record (kept as the
//! runtime wrote it, by path and SHA-256), its normalized steps and the patch.
//! Completion is judged from the patch by the task's verifiers; nothing that
//! interprets an episode looks at which runtime produced it.
//!
//! Two runtimes implement it: [`sven::SvenRunner`] (the [`crate::work::Worker`]
//! on sven's engine) and [`claude_code::ClaudeCodeRunner`] (the pinned Claude
//! Code build, started only under the launch rules in that module, answered
//! only by a local brain server).

pub mod claude_code;
pub mod sven;

use std::path::{Path, PathBuf};
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use splinter_sandbox::workspace::DEFAULT_EXCLUDED;
use splinter_sandbox::{Baseline, Patch, SandboxError, WorkspaceEnvironment};
use sven_sdk::RunConclusion;
use tokio::sync::watch;

/// Which runtime ran an episode, as the episode's record keeps it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeIdentity {
    /// The runtime: `sven` or `claude-code`.
    pub runtime: String,
    /// Its version.
    pub version: String,
    /// The SHA-256 of the binary that ran, when the runtime is a pinned binary.
    pub binary_sha256: Option<String>,
    /// How it was run, when that changes what it does (Claude Code's `bare`).
    pub mode: Option<String>,
}

/// The bounds of one run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RunLimits {
    /// Wall-clock time the run may take.
    pub deadline: Duration,
}

/// Stops a run from outside.
#[derive(Clone, Debug)]
pub struct RunCancel(watch::Sender<bool>);

impl Default for RunCancel {
    fn default() -> Self {
        Self(watch::channel(false).0)
    }
}

impl RunCancel {
    /// A token that has not fired.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Stops every run holding this token.
    pub fn cancel(&self) {
        self.0.send_replace(true);
    }

    /// Whether [`Self::cancel`] was called.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        *self.0.borrow()
    }

    /// Completes once the token is cancelled.
    pub async fn cancelled(&self) {
        let mut rx = self.0.subscribe();
        // The sender lives as long as `self`, so the wait cannot fail.
        let _ = rx.wait_for(|fired| *fired).await;
    }
}

/// A prepared episode: the tree the agent works in, the baseline its change
/// is read back against, and the run directory the runtime keeps its own
/// state and record in.
#[derive(Clone, Debug)]
pub struct Episode {
    /// The working tree.
    pub tree: PathBuf,
    /// The baseline commit of the tree.
    pub baseline: Baseline,
    /// The runtime's own directory, beside the tree.
    pub run_dir: PathBuf,
}

/// One normalized step: an assistant turn and the tool calls it made.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Step {
    /// What the assistant said.
    pub text: String,
    /// Its reasoning, when the runtime records it.
    pub reasoning: String,
    /// The tool calls of the turn, with their results.
    pub tool_calls: Vec<StepCall>,
}

/// One tool call of a [`Step`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct StepCall {
    /// The tool's name, as the runtime names it.
    pub name: String,
    /// Its input.
    pub input: serde_json::Value,
    /// What it returned, when it returned.
    pub result: Option<String>,
    /// Whether the runtime marked the result an error.
    pub error: bool,
}

/// What a run produced, before its change is read back.
#[derive(Clone, Debug)]
pub struct RunOutcome {
    /// How the run ended.
    pub conclusion: RunConclusion,
    /// The runtime's own record of the run, as it wrote it.
    pub native_record: PathBuf,
    /// The normalized steps of that record.
    pub steps: Vec<Step>,
    /// How long the run took.
    pub wall: Duration,
}

/// A whole episode, in the one shape every runtime's episodes have.
#[derive(Clone, Debug)]
pub struct EpisodeRecord {
    /// Which runtime ran it.
    pub runtime: RuntimeIdentity,
    /// How it ended.
    pub conclusion: RunConclusion,
    /// The runtime's own record, by path.
    pub native_record: PathBuf,
    /// The SHA-256 of that record when it was collected.
    pub native_sha256: String,
    /// The normalized steps.
    pub steps: Vec<Step>,
    /// The change, read back from git.
    pub patch: Patch,
    /// How long the run took.
    pub wall: Duration,
}

/// Why an episode could not be prepared, run or collected. A run that fails
/// or is stopped is an outcome ([`RunConclusion`]), not an error.
#[derive(Debug, thiserror::Error)]
pub enum RunnerError {
    /// The workspace could not be laid out or read back.
    #[error(transparent)]
    Sandbox(#[from] SandboxError),
    /// The runtime refused to start: a launch rule would be broken.
    #[error("the {runtime} launch was refused: {reason}")]
    Refused {
        /// The runtime.
        runtime: &'static str,
        /// Which rule.
        reason: String,
    },
    /// Reading or writing the run's files failed.
    #[error("{path}: {source}")]
    Io {
        /// The file.
        path: PathBuf,
        /// The underlying error.
        source: std::io::Error,
    },
    /// The runtime itself failed before it could conclude.
    #[error("the {runtime} run failed: {reason}")]
    Runtime {
        /// The runtime.
        runtime: &'static str,
        /// What failed.
        reason: String,
    },
}

/// An agent runtime that works on a prepared episode.
#[async_trait]
pub trait AgentRunner: Send + Sync {
    /// Which runtime this is.
    fn identity(&self) -> RuntimeIdentity;

    /// Works on `episode` on `instruction` within `limits` until it
    /// concludes, a limit ends it or `cancel` fires.
    async fn run(
        &self,
        episode: &Episode,
        instruction: &str,
        limits: RunLimits,
        cancel: RunCancel,
    ) -> Result<RunOutcome, RunnerError>;
}

/// Lays out an episode under `root` (which must be empty or absent): the
/// snapshot `archive` of `env` as `root/tree`, and `root/run` for the runtime.
pub fn prepare(
    env: &WorkspaceEnvironment,
    archive: &Path,
    root: &Path,
) -> Result<Episode, RunnerError> {
    let tree = root.join("tree");
    let run_dir = root.join("run");
    let baseline = env.materialize(archive, &tree)?;
    std::fs::create_dir_all(&run_dir).map_err(|source| RunnerError::Io {
        path: run_dir.clone(),
        source,
    })?;
    Ok(Episode {
        tree,
        baseline,
        run_dir,
    })
}

/// The record of a finished run: its change read back from git against the
/// episode's baseline, its native record digested.
pub fn collect(
    env: &WorkspaceEnvironment,
    episode: &Episode,
    runtime: RuntimeIdentity,
    outcome: RunOutcome,
) -> Result<EpisodeRecord, RunnerError> {
    let patch = env.diff_since(&episode.tree, &episode.baseline, &DEFAULT_EXCLUDED)?;
    let bytes = std::fs::read(&outcome.native_record).map_err(|source| RunnerError::Io {
        path: outcome.native_record.clone(),
        source,
    })?;
    Ok(EpisodeRecord {
        runtime,
        conclusion: outcome.conclusion,
        native_sha256: format!("{:x}", Sha256::digest(&bytes)),
        native_record: outcome.native_record,
        steps: outcome.steps,
        patch,
        wall: outcome.wall,
    })
}

/// Prepares, runs and collects one episode with `runner`.
pub async fn run_episode(
    runner: &dyn AgentRunner,
    env: &WorkspaceEnvironment,
    archive: &Path,
    root: &Path,
    instruction: &str,
    limits: RunLimits,
    cancel: RunCancel,
) -> Result<EpisodeRecord, RunnerError> {
    let episode = prepare(env, archive, root)?;
    let outcome = runner.run(&episode, instruction, limits, cancel).await?;
    collect(env, &episode, runner.identity(), outcome)
}
