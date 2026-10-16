// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements self-improving coding agents whose every
// step is auditable, for its clients. If your team needs expertise in agent
// evaluation or locally operated coding agents, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The repository as Splinter's shared machinery sees it, and the
//! supervisor's judgement of a candidate through that machinery.
//!
//! A run starts by capturing the tracked files of the baseline revision as
//! a snapshot (kept in Splinter's artifact store, identified by content).
//! Every attempt is a worker in a fresh copy of that snapshot, and a
//! candidate is judged by the workspace grading the `practice` pipeline
//! uses: the snapshot laid out again, the candidate's patch applied, the
//! protected paths fenced, the acceptance commands run in the sandbox. The
//! repository the task names is read once and never written; the checks
//! and the scripts they run are not in any tree a worker has.

use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};
use splinter_sdk::knowledge::capture::{capture_tree, DirtyPolicy, TreeRequest};
use splinter_sdk::measure::verifiers::workspace::{
    grade, CommandRun, Graded, Refusal, Touched, WorkspaceCheck,
};
use splinter_sdk::sandbox::{Patch, ResolvedEnvironment, WorkspaceEnvironment};
use splinter_sdk::store::artifacts::ArtifactStore;
use splinter_sdk::vocabulary::experience::Task;
use splinter_sdk::vocabulary::kinds::WORKSPACE_CHANGE;
use splinter_sdk::Context;

use crate::contract::Contract;
use crate::outcome::{CheckResult, FileChange};
use crate::redact::redact;

/// The repository at the baseline, as a workspace environment.
pub struct Workspace {
    /// The environment every worker and every judgement runs in.
    pub env: WorkspaceEnvironment,
    /// The archive of the snapshot, in the artifact store.
    pub archive: PathBuf,
    /// The commit that was captured.
    pub commit: String,
    store: ArtifactStore,
    scratch: PathBuf,
}

impl Workspace {
    /// The tracked files of `revision` of `repository`, captured. With
    /// `DirtyPolicy::Refuse` a repository whose tracked files differ from
    /// the revision checked out is refused: the loop works from a committed
    /// baseline and leaves the person's changes alone.
    pub fn capture(
        ctx: &Context,
        repository: &Path,
        revision: &str,
        dirty: DirtyPolicy,
    ) -> Result<Self> {
        let backend = ctx.environments().workspace()?;
        let store = ctx.artifacts();
        let captured = capture_tree(
            &TreeRequest {
                repository,
                revision,
                dirty,
                terms: None,
            },
            &backend.host,
            &store,
            &backend.scratch,
            ctx.clock(),
        )
        .with_context(|| format!("capturing {revision} of {}", repository.display()))?;
        Ok(Self {
            env: backend.environment(captured.snapshot),
            archive: captured.archive_path,
            commit: captured.commit,
            store,
            scratch: grading_scratch(),
        })
    }

    /// The digest of the snapshot, as the artifact store names it.
    #[must_use]
    pub fn snapshot_digest(&self) -> String {
        self.env.snapshot().digest.to_string()
    }

    /// The task of one attempt: `instruction` to a worker in this
    /// environment, graded by `check`.
    pub fn task(&self, instruction: &str, check: &WorkspaceCheck) -> Result<Task> {
        let record = ResolvedEnvironment::Workspace(self.env.clone()).record()?;
        Ok(Task::new(
            WORKSPACE_CHANGE,
            Vec::new(),
            record,
            instruction,
            vec![check.as_check()?],
        )?)
    }

    /// Judges `patch` by the contract's checks. An empty patch judges the
    /// snapshot as it is, which is how a baseline is validated.
    pub fn judge(&self, contract: &Contract, patch: &Patch) -> Result<Judgement> {
        let graded = grade(
            &self.env,
            &self.store,
            &self.scratch,
            &workspace_check(contract),
            patch,
        )?;
        Ok(Judgement::of(contract, graded))
    }
}

/// Where a candidate is graded. Not under Splinter's state: a failing
/// check prints the path of the tree it ran in, and that output is what a
/// retry is shown, which must not name the state root.
fn grading_scratch() -> PathBuf {
    std::env::temp_dir().join("splinter-agent-loop")
}

/// The contract's checks as the workspace grading takes them: every check
/// must pass after the change, in contract order, and the protected paths
/// are fenced. Nothing is laid over the tree: a check's script is wherever
/// the supervisor keeps it.
#[must_use]
pub fn workspace_check(contract: &Contract) -> WorkspaceCheck {
    WorkspaceCheck {
        fail_to_pass: contract.acceptance.iter().map(|c| c.call()).collect(),
        pass_to_pass: Vec::new(),
        hidden: None,
        protected: contract.protected.clone(),
        hidden_paths: Vec::new(),
        hidden_strength: None,
        expectations: Vec::new(),
        pinned: Vec::new(),
    }
}

/// What judging a candidate found, in the loop's own terms.
#[derive(Debug)]
pub struct Judgement {
    /// Why the candidate was set aside before any check ran.
    pub refusal: Option<Refusal>,
    /// Each check's result, in contract order.
    pub checks: Vec<CheckResult>,
    /// The files the candidate changed.
    pub changes: Vec<FileChange>,
}

impl Judgement {
    fn of(contract: &Contract, graded: Graded) -> Self {
        let checks = graded
            .commands
            .iter()
            .map(|run| check_result(contract, run))
            .collect();
        let changes = graded.touched.iter().map(file_change).collect();
        Self {
            refusal: graded.refusal,
            checks,
            changes,
        }
    }

    /// Whether every check passed.
    #[must_use]
    pub fn all_passed(&self) -> bool {
        self.refusal.is_none() && !self.checks.is_empty() && self.checks.iter().all(|c| c.passed)
    }
}

fn check_result(contract: &Contract, run: &CommandRun) -> CheckResult {
    let check = &contract.acceptance[run.index];
    CheckResult {
        name: check.name.clone(),
        command: check.command.clone(),
        passed: run.passed,
        exit_code: run.result.exit_code,
        timed_out: run.result.timed_out,
        duration_ms: u64::try_from(run.elapsed.as_millis()).unwrap_or(u64::MAX),
        stdout_tail: redact(&splinter_sdk::agent::work::tail(&run.result.stdout)),
        stderr_tail: redact(&splinter_sdk::agent::work::tail(&run.result.stderr)),
    }
}

fn file_change(touched: &Touched) -> FileChange {
    let status = match (&touched.before, &touched.after) {
        (None, _) => "added",
        (_, None) => "deleted",
        _ => "modified",
    };
    FileChange {
        path: touched.path.clone(),
        status: status.into(),
        before: touched.before.as_ref().map(ToString::to_string),
        after: touched.after.as_ref().map(ToString::to_string),
    }
}
