// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements executable evaluation of code changes
// against a project's own test suite, for its clients. If your team needs
// expertise in test-driven benchmarks for code-editing agents, you can
// procure our services by sending an email to info@swedishembedded.com.

//! What every workspace task is built from and checked with, whatever
//! seeded it: the archives of the solver's tree and of the hidden overlay,
//! the task made of them, and the proof that the reference change makes its
//! check pass.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use splinter_core::annotation::Outcome;
use splinter_core::clock::Clock;
use splinter_core::digest::Digest;
use splinter_core::experience::{Experience, Privileged, PrivilegedKind, Provenance, Task};
use splinter_core::kinds::WORKSPACE_CHANGE;
use splinter_eval::verifiers::workspace::{WorkspaceCheck, WorkspaceVerifier};
use splinter_eval::verifiers::Verifier;
use splinter_sandbox::{ResolvedEnvironment, TreeSnapshot, WorkspaceEnvironment};
use splinter_store::artifacts::ArtifactSpec;

use super::pack::{pack_dir, pack_files};
use super::{WorkspaceError, WorkspaceGenerator, WORKSPACE_GENERATOR};

/// The role of a snapshot's archive in the artifact store.
const SNAPSHOT_ROLE: &str = "workspace-snapshot";

/// The role of a hidden overlay in the artifact store.
const OVERLAY_ROLE: &str = "workspace-hidden";

/// A hidden overlay: the archive of the files only the check sees.
pub(super) struct Overlay {
    pub(super) digest: Digest,
    pub(super) path: PathBuf,
}

/// The archives of a candidate, in a scratch directory until the candidate
/// is known to be sound and they are kept in the artifact store.
pub(super) struct Staged {
    _work: tempfile::TempDir,
    pub(super) snapshot: TreeSnapshot,
    pub(super) archive: PathBuf,
    pub(super) overlay: Option<Overlay>,
}

impl Staged {
    /// The hidden overlay's digest, when there is one.
    pub(super) fn hidden(&self) -> Option<Digest> {
        self.overlay.as_ref().map(|o| o.digest.clone())
    }
}

impl WorkspaceGenerator {
    /// A new directory under the scratch root, removed when dropped.
    pub(super) fn temp_dir(&self, prefix: &str) -> Result<tempfile::TempDir, WorkspaceError> {
        let io = |source| WorkspaceError::Io {
            path: self.scratch.clone(),
            source,
        };
        std::fs::create_dir_all(&self.scratch).map_err(io)?;
        tempfile::Builder::new()
            .prefix(prefix)
            .tempdir_in(&self.scratch)
            .map_err(io)
    }

    /// Archives `tree` (leaving out the entries named `exclude`) as the
    /// solver's snapshot and, when `overlay` names a directory and files
    /// below it, those files as the hidden overlay.
    pub(super) fn stage(
        &self,
        tree: &Path,
        exclude: &[&str],
        overlay: Option<(&Path, &[String])>,
    ) -> Result<Staged, WorkspaceError> {
        let io = |path: &Path| {
            let path = path.to_path_buf();
            move |source| WorkspaceError::Io { path, source }
        };
        let work = self.temp_dir("splinter-staged-")?;
        let archive = work.path().join("tree.tar");
        pack_dir(tree, exclude, &archive)?;
        let snapshot = self.host.snapshot(&archive)?;
        let overlay = match overlay {
            Some((dir, files)) if !files.is_empty() => {
                let path = work.path().join("hidden.tar");
                pack_files(dir, files, &path)?;
                let digest = std::fs::File::open(&path)
                    .and_then(Digest::of_reader)
                    .map_err(io(&path))?;
                Some(Overlay { digest, path })
            }
            _ => None,
        };
        Ok(Staged {
            _work: work,
            snapshot,
            archive,
            overlay,
        })
    }

    /// Keeps the archives of `staged` in the artifact store, where the
    /// verifier and a solver's environment find them by digest.
    pub(super) fn keep(&self, staged: &Staged) -> Result<(), WorkspaceError> {
        let mut kept = vec![(&staged.archive, SNAPSHOT_ROLE, &staged.snapshot.digest)];
        if let Some(overlay) = &staged.overlay {
            kept.push((&overlay.path, OVERLAY_ROLE, &overlay.digest));
        }
        for (file, role, digest) in kept {
            let artifact = self
                .store
                .put_file(file, &ArtifactSpec::new(role, WORKSPACE_GENERATOR))?;
            debug_assert_eq!(&artifact.digest, digest);
        }
        Ok(())
    }

    /// The task: the solver's environment, the request, the check, the
    /// reference change when there is one, and the family as the concept it
    /// exercises.
    pub(super) fn build_task(
        &self,
        env: &WorkspaceEnvironment,
        check: &WorkspaceCheck,
        instruction: &str,
        family: &str,
        reference: Option<&str>,
    ) -> Result<Task, WorkspaceError> {
        let record = ResolvedEnvironment::Workspace(env.clone()).record()?;
        let mut privileged = vec![check.as_check()?];
        privileged.extend(reference.map(|patch| Privileged {
            kind: PrivilegedKind::Reference,
            content: patch.to_string(),
            span: None,
        }));
        let task = Task::new(
            WORKSPACE_CHANGE,
            Vec::new(),
            record,
            instruction,
            privileged,
        )?;
        Ok(task.with_concepts([family])?)
    }

    /// Whether the workspace verifier passes `patch` on `task`: the proof
    /// that the reference change makes the check pass, run the way a solver's
    /// answer will be. `clock` stamps the throwaway experience.
    pub(super) fn passes(
        &self,
        env: &WorkspaceEnvironment,
        task: &Task,
        patch: &str,
        clock: &dyn Clock,
    ) -> Result<bool, WorkspaceError> {
        let verifier = WorkspaceVerifier::new(
            vec![env.clone()],
            Arc::new(self.store.clone()),
            self.scratch.join("verify"),
        );
        let provenance = Provenance::new("splinter/reference-patch", clock);
        let experience = Experience::answered_without_a_run(task.clone(), patch, provenance)?;
        Ok(verifier.verify(task, &experience)?.outcome == Outcome::Pass)
    }
}
