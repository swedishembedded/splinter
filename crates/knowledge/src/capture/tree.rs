// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reproducible snapshots of source trees that
// agents are trained and measured on, for its clients. If your team needs
// expertise in turning a codebase into executable training tasks, you can
// procure our services by sending an email to info@swedishembedded.com.

//! A git revision captured as a workspace snapshot: the tracked files at
//! that revision, archived by `git archive`, kept as an artifact, and
//! described by a source that records where it came from.
//!
//! The archive is made from the revision's tree object with a fixed
//! modification time, so the same tree is the same archive (and so the same
//! environment) whichever commit, clone or day it is taken from. Executable
//! bits are kept; no `.git` entry is. `git archive` honours the repository's
//! own `export-ignore` and `export-subst` attributes, which would make the
//! snapshot a different tree than the revision names, so the tree the
//! extracted archive makes is compared with the revision's own and a
//! difference is refused.
//!
//! A revision that is the one checked out while its tracked files differ
//! from it is refused unless the caller says [`DirtyPolicy::Allow`]; the
//! archive then still holds the committed files, and the source records the
//! tree as dirty.

use std::path::{Path, PathBuf};

use serde::Serialize;
use splinter_core::clock::Clock;
use splinter_core::digest::{canonical_json, Digest};
use splinter_core::source::{CapturedSource, Origin, PartContent, Revision, SourceError};
use splinter_core::terms::Terms;
use splinter_sandbox::{SandboxError, TreeSnapshot, WorkspaceHost};
use splinter_store::artifacts::{Artifact, ArtifactSpec, ArtifactStore};
use splinter_store::error::StoreError;

use super::repository::git;

/// The producer name the snapshot artifact carries.
pub const SNAPSHOT_PRODUCER: &str = "splinter-knowledge/tree@1";

/// The role of a snapshot's archive in the artifact store.
pub const SNAPSHOT_ROLE: &str = "workspace-snapshot";

/// The name of the part of a tree's source that describes the snapshot.
pub const SNAPSHOT_PART: &str = "workspace-snapshot.json";

/// The modification time every archived entry carries, so an archive is a
/// function of the tree alone.
const ARCHIVE_MTIME: &str = "2000-01-01T00:00:00Z";

/// What to do when the revision is checked out and its tracked files have
/// been changed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DirtyPolicy {
    /// Refuse: the person's uncommitted work is not what the snapshot holds.
    #[default]
    Refuse,
    /// Snapshot the committed files and record that the tree was dirty.
    Allow,
}

/// What to capture.
#[derive(Clone, Debug)]
pub struct TreeRequest<'a> {
    /// The git repository (a work tree or a bare repository).
    pub repository: &'a Path,
    /// The revision: anything git resolves to a commit.
    pub revision: &'a str,
    /// What to do about uncommitted changes to the checked-out revision.
    pub dirty: DirtyPolicy,
    /// The terms the repository is under (its licence as a person states
    /// it); absent when none were stated, which every consumer treats as
    /// unknown.
    pub terms: Option<Terms>,
}

/// A captured tree.
#[derive(Debug)]
pub struct CapturedTree {
    /// The archive's digest and the id of the git tree it makes: what a
    /// workspace environment is pinned to.
    pub snapshot: TreeSnapshot,
    /// The archive, as the artifact store keeps it.
    pub archive: Artifact,
    /// Where the archive's file is.
    pub archive_path: PathBuf,
    /// The commit that was archived.
    pub commit: String,
    /// The source describing the capture, to be stored with the source
    /// store: its origin is the repository path with the revision, and its
    /// one part names the archive, the commit and the tree id.
    pub source: CapturedSource,
}

/// Why a tree could not be captured.
#[derive(Debug, thiserror::Error)]
pub enum TreeError {
    /// A revision git cannot resolve to a commit, or one that could be read
    /// as an option.
    #[error("{revision:?} is not a revision of {repository}: {detail}")]
    Revision {
        /// The revision.
        revision: String,
        /// The repository.
        repository: PathBuf,
        /// What git said.
        detail: String,
    },
    /// The revision is checked out and its tracked files were changed.
    #[error(
        "{repository} has uncommitted changes to tracked files at {commit}; commit them or allow a dirty tree"
    )]
    Dirty {
        /// The repository.
        repository: PathBuf,
        /// The commit that is checked out.
        commit: String,
    },
    /// A git command failed.
    #[error("git {args:?} failed in {repository}: {detail}")]
    Git {
        /// The arguments.
        args: Vec<String>,
        /// The repository.
        repository: PathBuf,
        /// Its status and standard error.
        detail: String,
    },
    /// The archive does not make the revision's tree: an attribute of the
    /// repository (`export-ignore`, `export-subst`) or a submodule changed
    /// what `git archive` holds.
    #[error(
        "the archive of {revision:?} makes tree {found}, but the revision's tree is {expected}; \
         export-ignore, export-subst or submodules changed what git archive holds"
    )]
    NotTheTree {
        /// The revision.
        revision: String,
        /// The revision's tree id.
        expected: String,
        /// The tree id the archive makes.
        found: String,
    },
    /// A file operation failed.
    #[error("{path}: {source}")]
    Io {
        /// The file or directory.
        path: PathBuf,
        /// The underlying error.
        source: std::io::Error,
    },
    /// Laying the archive out to read its tree failed.
    #[error(transparent)]
    Sandbox(#[from] SandboxError),
    /// The artifact store failed.
    #[error(transparent)]
    Store(#[from] StoreError),
    /// The source is not valid.
    #[error(transparent)]
    Source(#[from] SourceError),
    /// The description of the snapshot cannot be serialized.
    #[error("cannot serialize the snapshot description: {0}")]
    Serialize(#[from] serde_json::Error),
}

/// What the part of a tree's source says.
#[derive(Serialize)]
struct SnapshotDescription<'a> {
    archive: &'a Digest,
    commit: &'a str,
    tree_id: &'a str,
}

/// Captures the tracked files of `request.revision` as a snapshot: the
/// archive is kept in `store`, laid out once in a scratch directory under
/// `scratch` by `host` to read the tree it makes, and described by a
/// source stamped by `clock` (and stated under `request.terms`, when given).
pub fn capture_tree(
    request: &TreeRequest<'_>,
    host: &WorkspaceHost,
    store: &ArtifactStore,
    scratch: &Path,
    clock: &dyn Clock,
) -> Result<CapturedTree, TreeError> {
    let repository = request
        .repository
        .canonicalize()
        .map_err(io(request.repository))?;
    let revision = request.revision;
    let unusable = |detail: String| TreeError::Revision {
        revision: revision.to_string(),
        repository: repository.clone(),
        detail,
    };
    if revision.is_empty() || revision.starts_with('-') {
        return Err(unusable(
            "a revision may not be empty or start with '-'".into(),
        ));
    }
    let commit = resolve(&repository, &format!("{revision}^{{commit}}")).map_err(unusable)?;
    let tree_id = resolve(&repository, &format!("{commit}^{{tree}}")).map_err(unusable)?;
    let dirty = is_dirty(&repository, &commit)?;
    if dirty && request.dirty == DirtyPolicy::Refuse {
        return Err(TreeError::Dirty { repository, commit });
    }

    std::fs::create_dir_all(scratch).map_err(io(scratch))?;
    let work = tempfile::Builder::new()
        .prefix("splinter-capture-")
        .tempdir_in(scratch)
        .map_err(io(scratch))?;
    let tar = work.path().join("tree.tar");
    let mtime = format!("--mtime={ARCHIVE_MTIME}");
    let out = format!("--output={}", tar.display());
    run_git(
        &repository,
        &["archive", "--format=tar", &mtime, &out, &tree_id],
    )?;
    let archive = store.put_file(&tar, &ArtifactSpec::new(SNAPSHOT_ROLE, SNAPSHOT_PRODUCER))?;
    let archive_path = store.path(&archive.digest)?;
    let snapshot = host.snapshot(&archive_path)?;
    if snapshot.tree_id != tree_id {
        return Err(TreeError::NotTheTree {
            revision: revision.to_string(),
            expected: tree_id,
            found: snapshot.tree_id,
        });
    }

    let description = canonical_json(&SnapshotDescription {
        archive: &snapshot.digest,
        commit: &commit,
        tree_id: &snapshot.tree_id,
    })?;
    let origin = Origin::Repository {
        path: repository.to_string_lossy().into_owned(),
        revision: Some(Revision {
            commit: Some(commit.clone()),
            dirty,
        }),
        skipped: Vec::new(),
    };
    let part = PartContent {
        name: SNAPSHOT_PART.into(),
        media_type: "application/json".into(),
        bytes: description,
    };
    let mut source = CapturedSource::new(origin, vec![part], clock)?;
    if let Some(terms) = &request.terms {
        source = source.with_terms(terms.clone())?;
    }
    Ok(CapturedTree {
        snapshot,
        archive,
        archive_path,
        commit,
        source,
    })
}

/// The object `spec` names, or git's complaint.
fn resolve(repository: &Path, spec: &str) -> Result<String, String> {
    let args = ["rev-parse", "--verify", "--quiet", spec];
    let output = git(repository, &args).map_err(|e| e.to_string())?;
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    } else {
        Err(format!("{spec} does not resolve"))
    }
}

/// Whether `commit` is checked out in `repository` with changes to its
/// tracked files. A commit that is not `HEAD`, and a repository with no work
/// tree, have none to speak of.
fn is_dirty(repository: &Path, commit: &str) -> Result<bool, TreeError> {
    let Ok(head) = resolve(repository, "HEAD^{commit}") else {
        return Ok(false);
    };
    if head != commit {
        return Ok(false);
    }
    let inside =
        git(repository, &["rev-parse", "--is-inside-work-tree"]).map_err(io(repository))?;
    if String::from_utf8_lossy(&inside.stdout).trim() != "true" {
        return Ok(false);
    }
    let status = run_git(
        repository,
        &["status", "--porcelain", "--untracked-files=no"],
    )?;
    Ok(!status.is_empty())
}

/// The standard output of `git args` in `repository`.
fn run_git(repository: &Path, args: &[&str]) -> Result<Vec<u8>, TreeError> {
    let output = git(repository, args).map_err(io(repository))?;
    if !output.status.success() {
        return Err(TreeError::Git {
            args: args.iter().map(|a| (*a).to_string()).collect(),
            repository: repository.to_path_buf(),
            detail: format!(
                "{}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            ),
        });
    }
    Ok(output.stdout)
}

fn io(path: &Path) -> impl FnOnce(std::io::Error) -> TreeError + '_ {
    move |source| TreeError::Io {
        path: path.to_path_buf(),
        source,
    }
}
