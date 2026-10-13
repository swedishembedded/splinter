// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reproducible, auditable workspaces for
// agents that change code, for its clients. If your team needs expertise in
// sandboxed code-editing agents or reproducible agent runs, you can procure
// our services by sending an email to info@swedishembedded.com.

//! A workspace: a source tree a solver changes by editing files and running
//! commands, pinned by the digest of the archive it is laid out from.
//!
//! * [`TreeSnapshot`] - the archive's content digest and the id of the git
//!   tree it makes. Identical archives are the same environment.
//! * [`WorkspaceEnvironment::materialize`] - lays the archive out in a
//!   directory byte for byte (executable bits included) and records it as
//!   the [`Baseline`] of a repository kept beside the directory.
//! * [`WorkspaceEnvironment::run`] - one command in the tree, in the
//!   environment's sandbox.
//! * [`WorkspaceEnvironment::diff_since`] and
//!   [`WorkspaceEnvironment::apply`] - what the solver changed, read back
//!   from git as a [`Patch`] (never from what the solver says it did), and
//!   that patch applied to another tree of the same snapshot.
//!
//! The environment's record pins the snapshot, the sandbox and the git that
//! reads changes back; see [`crate::environment`].

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use splinter_core::digest::Digest;

use crate::backend::{CodeResult, CommandCall, Sandbox, SandboxError};
use crate::workspace_host::WorkspaceHost;

/// The paths a diff leaves out unless the caller says otherwise: sven's own
/// working directory (its audit log and logs, written into the project
/// root) and Python bytecode caches. Git glob patterns, relative to the
/// tree.
pub const DEFAULT_EXCLUDED: [&str; 2] = [".sven/**", "**/__pycache__/**"];

/// A source tree as an environment pins it: the archive and what it makes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TreeSnapshot {
    /// The content digest of the archive file (a tar, as `git archive`
    /// writes it).
    pub digest: Digest,
    /// The id of the git tree the archive's files make, executable bits
    /// and symbolic links included.
    pub tree_id: String,
}

/// The state a tree was laid out in, to read changes against.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Baseline {
    /// The id of the baseline commit (the same for the same tree).
    pub commit: String,
    /// The id of the baseline commit's tree: [`TreeSnapshot::tree_id`].
    pub tree_id: String,
    git_dir: PathBuf,
}

impl Baseline {
    pub(crate) fn new(commit: String, tree_id: String, git_dir: PathBuf) -> Self {
        Self {
            commit,
            tree_id,
            git_dir,
        }
    }

    /// The repository that records the baseline. It lives beside the tree
    /// (`<tree>.git`), out of reach of a command that only sees the tree,
    /// and goes with the directory that holds both.
    #[must_use]
    pub fn git_dir(&self) -> &Path {
        &self.git_dir
    }
}

/// What a solver changed in a tree: a git patch, with binary files and
/// mode changes, and the paths it touches.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Patch {
    /// The patch, as `git diff --binary` writes it.
    pub diff: Vec<u8>,
    /// The paths it touches, relative to the tree, in order. A rename is a
    /// deletion and an addition.
    pub files: Vec<PathBuf>,
}

impl Patch {
    /// Whether nothing changed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.diff.is_empty()
    }
}

/// A workspace resolved: its snapshot, the sandbox commands run in, and the
/// host tools that lay a tree out and read its changes back. Cheap to
/// clone: clones share the sandbox and the tools.
#[derive(Clone, Debug)]
pub struct WorkspaceEnvironment {
    snapshot: TreeSnapshot,
    sandbox: Arc<dyn Sandbox>,
    host: Arc<WorkspaceHost>,
}

impl WorkspaceEnvironment {
    /// The workspace of `snapshot` whose commands run in `sandbox`, laid
    /// out and read back with `host`.
    #[must_use]
    pub fn new(
        snapshot: TreeSnapshot,
        sandbox: Arc<dyn Sandbox>,
        host: Arc<WorkspaceHost>,
    ) -> Self {
        Self {
            snapshot,
            sandbox,
            host,
        }
    }

    /// The tree this workspace is laid out from.
    #[must_use]
    pub fn snapshot(&self) -> &TreeSnapshot {
        &self.snapshot
    }

    /// The sandbox commands run in.
    #[must_use]
    pub fn sandbox(&self) -> &dyn Sandbox {
        self.sandbox.as_ref()
    }

    /// The host tools that lay a tree out and read its changes back.
    #[must_use]
    pub fn host(&self) -> &WorkspaceHost {
        &self.host
    }

    /// Lays `archive` out in `into`, which must not exist or be empty, and
    /// records it as the baseline. Refuses an archive that is not the
    /// snapshot's (a different digest, or a tree of a different id), so a
    /// tree is always exactly the one the environment names.
    pub fn materialize(&self, archive: &Path, into: &Path) -> Result<Baseline, SandboxError> {
        let actual = std::fs::File::open(archive)
            .and_then(Digest::of_reader)
            .map_err(crate::backend::io(archive))?;
        if actual != self.snapshot.digest {
            return Err(SandboxError::Workspace {
                what: format!("laying out {}", archive.display()),
                detail: format!(
                    "the archive has digest {actual}, the snapshot names {}",
                    self.snapshot.digest
                ),
            });
        }
        let baseline = self.host.import(archive, into)?;
        if baseline.tree_id != self.snapshot.tree_id {
            return Err(SandboxError::Workspace {
                what: format!("laying out {}", archive.display()),
                detail: format!(
                    "the extracted tree is {}, the snapshot names {}",
                    baseline.tree_id, self.snapshot.tree_id
                ),
            });
        }
        Ok(baseline)
    }

    /// Runs one command with `tree` as its working directory, in this
    /// environment's sandbox.
    pub fn run(&self, tree: &Path, call: &CommandCall) -> Result<CodeResult, SandboxError> {
        self.sandbox.run_command(tree, call)
    }

    /// What `tree` differs from `baseline` by, leaving out the paths that
    /// match `excluded` (git glob patterns relative to the tree; see
    /// [`DEFAULT_EXCLUDED`]) and any repository inside the tree. A new file
    /// that the tree's own ignore rules hide is a cache, not a change.
    pub fn diff_since(
        &self,
        tree: &Path,
        baseline: &Baseline,
        excluded: &[&str],
    ) -> Result<Patch, SandboxError> {
        self.host.diff(tree, baseline, excluded)
    }

    /// Applies `patch` to `tree`, all or nothing, and returns the paths it
    /// touched. A patch that does not apply cleanly leaves the tree as it
    /// was.
    pub fn apply(&self, tree: &Path, patch: &Patch) -> Result<Vec<PathBuf>, SandboxError> {
        self.host.apply(tree, patch)
    }
}
