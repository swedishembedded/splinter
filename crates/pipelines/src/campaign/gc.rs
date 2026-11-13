// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning pipelines that keep a multi-day
// training campaign inside its disk quota, for its clients. If your team
// needs expertise in checkpoint retention for long training runs you can
// procure our services by sending an email to info@swedishembedded.com.

//! Checkpoint garbage collection: keep the digests, drop the bytes.
//!
//! A campaign trains one candidate adapter per arm, and a rolled-back or
//! superseded one is never used again; kept, they cost gigabytes and thousands
//! of files per candidate on a disk with a file quota. [`plan`] decides which
//! to drop under a [`Keep`] rule that never touches the champion or a
//! candidate still awaiting its gate; [`collect`] removes what it names, only
//! inside the directory the adapters live in, and returns a [`Dropped`]
//! record per removal so the campaign can still say which digest was trained,
//! when, and how big it was.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use splinter_core::digest::Digest;

/// One adapter on disk.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Its content digest.
    pub digest: Digest,
    /// Where it is.
    pub path: PathBuf,
    /// The arm that trained it.
    pub generation: u32,
    /// Bytes it occupies.
    pub bytes: u64,
}

/// What survives a collection, besides the champion and any candidate still
/// awaiting a gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Keep {
    /// The most recent checkpoints, by arm, kept for a post-mortem.
    pub recent: usize,
}

impl Default for Keep {
    fn default() -> Self {
        Self { recent: 1 }
    }
}

/// A checkpoint removed: the facts that outlive the files.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Dropped {
    /// Its content digest.
    pub digest: Digest,
    /// The arm that trained it.
    pub generation: u32,
    /// Bytes it occupied.
    pub bytes: u64,
}

/// Which of `checkpoints` to drop. `protected` digests (the champion, a
/// pending candidate) are never dropped.
#[must_use]
pub fn plan<'a>(
    checkpoints: &'a [Checkpoint],
    protected: &[&Digest],
    keep: &Keep,
) -> Vec<&'a Checkpoint> {
    let mut newest: Vec<&Checkpoint> = checkpoints.iter().collect();
    newest.sort_by_key(|c| std::cmp::Reverse(c.generation));
    let recent: Vec<&Digest> = newest.iter().take(keep.recent).map(|c| &c.digest).collect();
    checkpoints
        .iter()
        .filter(|c| !protected.contains(&&c.digest) && !recent.contains(&&c.digest))
        .collect()
}

/// Why a collection could not run.
#[derive(Debug, thiserror::Error)]
pub enum GcError {
    /// A checkpoint to drop is not inside the directory the campaign keeps
    /// its adapters in: nothing is removed.
    #[error("{path} is outside {root}; refusing to delete it")]
    Outside {
        /// The checkpoint's path.
        path: String,
        /// The adapters directory.
        root: String,
    },
    /// A removal failed.
    #[error("removing {path}: {source}")]
    Remove {
        /// The checkpoint's path.
        path: String,
        /// The operating system's error.
        #[source]
        source: std::io::Error,
    },
}

/// Removes `doomed` (from [`plan`]) from under `root`, and returns what was
/// dropped. Every path is checked before the first is removed.
///
/// # Errors
/// A path outside `root`, or a removal that failed; the checkpoints removed
/// before a failure stay removed.
pub fn collect(root: &Path, doomed: &[&Checkpoint]) -> Result<Vec<Dropped>, GcError> {
    let root_real = root.canonicalize().map_err(|source| GcError::Remove {
        path: root.display().to_string(),
        source,
    })?;
    let mut targets = Vec::new();
    for checkpoint in doomed {
        let outside = || GcError::Outside {
            path: checkpoint.path.display().to_string(),
            root: root.display().to_string(),
        };
        match checkpoint.path.canonicalize() {
            Ok(real) if real != root_real && real.starts_with(&root_real) => {
                targets.push((real, checkpoint));
            }
            Ok(_) => return Err(outside()),
            // Already gone: it counts as dropped, there is nothing to remove.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                if !checkpoint.path.starts_with(root) {
                    return Err(outside());
                }
            }
            Err(source) => {
                return Err(GcError::Remove {
                    path: checkpoint.path.display().to_string(),
                    source,
                })
            }
        }
    }
    for (real, _) in &targets {
        std::fs::remove_dir_all(real).map_err(|source| GcError::Remove {
            path: real.display().to_string(),
            source,
        })?;
    }
    Ok(doomed
        .iter()
        .map(|c| Dropped {
            digest: c.digest.clone(),
            generation: c.generation,
            bytes: c.bytes,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkpoint(root: &Path, generation: u32) -> Checkpoint {
        let path = root.join(format!("g{generation}"));
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(path.join("adapter.safetensors"), b"weights").unwrap();
        Checkpoint {
            digest: Digest::of(format!("g{generation}").as_bytes()),
            path,
            generation,
            bytes: 7,
        }
    }

    #[test]
    fn the_champion_a_pending_candidate_and_the_newest_survive() {
        let dir = tempfile::tempdir().unwrap();
        let all: Vec<Checkpoint> = (1..=5).map(|g| checkpoint(dir.path(), g)).collect();
        let champion = &all[1].digest;
        let pending = &all[3].digest;
        let doomed = plan(&all, &[champion, pending], &Keep { recent: 1 });
        let generations: Vec<u32> = doomed.iter().map(|c| c.generation).collect();
        assert_eq!(generations, [1, 3]);
    }

    #[test]
    fn collecting_removes_the_files_and_keeps_the_digests() {
        let dir = tempfile::tempdir().unwrap();
        let all: Vec<Checkpoint> = (1..=3).map(|g| checkpoint(dir.path(), g)).collect();
        let doomed = plan(&all, &[&all[2].digest], &Keep { recent: 0 });
        let dropped = collect(dir.path(), &doomed).unwrap();
        assert_eq!(dropped.len(), 2);
        assert_eq!(dropped[0].digest, all[0].digest);
        assert_eq!(dropped[0].bytes, 7);
        assert!(!all[0].path.exists() && !all[1].path.exists());
        assert!(
            all[2].path.exists(),
            "the protected checkpoint is untouched"
        );
    }

    #[test]
    fn a_path_outside_the_adapters_directory_is_never_deleted() {
        let adapters = tempfile::tempdir().unwrap();
        let elsewhere = tempfile::tempdir().unwrap();
        let stray = checkpoint(elsewhere.path(), 1);
        let inside = checkpoint(adapters.path(), 2);
        let refused = collect(adapters.path(), &[&inside, &stray]).unwrap_err();
        assert!(matches!(refused, GcError::Outside { .. }), "{refused}");
        assert!(stray.path.exists());
        assert!(
            inside.path.exists(),
            "nothing is removed when any path is refused"
        );
        let root_itself = Checkpoint {
            path: adapters.path().to_path_buf(),
            ..inside.clone()
        };
        assert!(matches!(
            collect(adapters.path(), &[&root_itself]),
            Err(GcError::Outside { .. })
        ));
    }

    #[test]
    fn a_checkpoint_already_gone_counts_as_dropped() {
        let dir = tempfile::tempdir().unwrap();
        let ghost = Checkpoint {
            digest: Digest::of(b"ghost"),
            path: dir.path().join("ghost"),
            generation: 1,
            bytes: 0,
        };
        assert_eq!(collect(dir.path(), &[&ghost]).unwrap().len(), 1);
    }
}
