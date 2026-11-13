// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning pipelines that refuse to start a
// long run on a disk that cannot finish it, for its clients. If your team
// needs expertise in making multi-day training campaigns fail early and
// clearly you can procure our services by sending an email to
// info@swedishembedded.com.

//! Storage preflight: whether the disks a long run writes to have the bytes
//! and the files it needs, answered before the run starts and again before
//! each arm of a campaign.
//!
//! The project disk of this deployment has a quota of 500 000 files; a
//! registered run was killed by it with gigabytes free, in every replicate, and
//! its state was lost. A byte check alone does not see that, so every
//! requirement names both and the first shortfall is an error that says which
//! one, where, and what is left ([`PreflightError`]).

use std::path::{Path, PathBuf};

use splinter_store::space::{headroom, Headroom};

/// What a run needs free on the filesystem holding `path`.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Requirement {
    /// A directory the run writes under (or a parent that exists).
    pub path: PathBuf,
    /// Bytes that must be free.
    pub min_free_bytes: u64,
    /// Files that must be free; not checked on a filesystem that does not
    /// count them.
    pub min_free_inodes: u64,
}

/// Why a run must not start.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum PreflightError {
    /// Too few bytes are free.
    #[error(
        "{path}: {free} bytes free, the run needs {needed}; free space or point the run at another disk"
    )]
    Bytes {
        /// The directory checked.
        path: String,
        /// Bytes free.
        free: u64,
        /// Bytes needed.
        needed: u64,
    },
    /// Too few files can be created: the disk has bytes but its file quota is
    /// nearly used up.
    #[error(
        "{path}: {free} of {total} files free, the run needs {needed}; the disk's file quota, not its size, is the limit: delete superseded checkpoints and scratch trees or use another disk"
    )]
    Inodes {
        /// The directory checked.
        path: String,
        /// Files free.
        free: u64,
        /// Files the disk allows in all.
        total: u64,
        /// Files needed.
        needed: u64,
    },
    /// The directory could not be inspected.
    #[error("{path}: the free space could not be read: {reason}")]
    Unreadable {
        /// The directory checked.
        path: String,
        /// What the operating system said.
        reason: String,
    },
}

/// Whether `room` meets `requirement`.
///
/// # Errors
/// The first shortfall: bytes, then files.
pub fn judge(
    path: &Path,
    requirement: &Requirement,
    room: &Headroom,
) -> Result<(), PreflightError> {
    if room.free_bytes < requirement.min_free_bytes {
        return Err(PreflightError::Bytes {
            path: path.display().to_string(),
            free: room.free_bytes,
            needed: requirement.min_free_bytes,
        });
    }
    if let (Some(free), Some(total)) = (room.free_inodes, room.total_inodes) {
        if free < requirement.min_free_inodes {
            return Err(PreflightError::Inodes {
                path: path.display().to_string(),
                free,
                total,
                needed: requirement.min_free_inodes,
            });
        }
    }
    Ok(())
}

/// Checks every requirement against the disk it names.
///
/// # Errors
/// The first requirement the disk cannot meet, or a directory that cannot be
/// inspected.
pub fn check(requirements: &[Requirement]) -> Result<Vec<Headroom>, PreflightError> {
    requirements
        .iter()
        .map(|requirement| {
            let room = headroom(&requirement.path).map_err(|e| PreflightError::Unreadable {
                path: requirement.path.display().to_string(),
                reason: e.to_string(),
            })?;
            judge(&requirement.path, requirement, &room)?;
            Ok(room)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn requirement(bytes: u64, inodes: u64) -> Requirement {
        Requirement {
            path: PathBuf::from("/disk/run"),
            min_free_bytes: bytes,
            min_free_inodes: inodes,
        }
    }

    fn room(bytes: u64, inodes: Option<u64>) -> Headroom {
        Headroom {
            free_bytes: bytes,
            free_inodes: inodes,
            total_inodes: inodes.map(|_| 500_000),
        }
    }

    #[test]
    fn a_disk_with_room_in_bytes_and_files_passes() {
        let r = requirement(1 << 30, 20_000);
        assert_eq!(judge(&r.path, &r, &room(2 << 30, Some(100_000))), Ok(()));
    }

    #[test]
    fn gigabytes_free_do_not_hide_an_exhausted_file_quota() {
        let r = requirement(1 << 30, 20_000);
        let refused = judge(&r.path, &r, &room(300 << 30, Some(3_000))).unwrap_err();
        assert_eq!(
            refused,
            PreflightError::Inodes {
                path: "/disk/run".into(),
                free: 3_000,
                total: 500_000,
                needed: 20_000
            }
        );
        assert!(refused.to_string().contains("file quota"));
    }

    #[test]
    fn too_few_bytes_is_refused_first() {
        let r = requirement(1 << 30, 20_000);
        assert!(matches!(
            judge(&r.path, &r, &room(1 << 20, Some(3))),
            Err(PreflightError::Bytes { .. })
        ));
    }

    #[test]
    fn a_filesystem_that_does_not_count_files_is_checked_on_bytes_alone() {
        let r = requirement(1 << 20, u64::MAX);
        assert_eq!(judge(&r.path, &r, &room(2 << 20, None)), Ok(()));
    }

    #[test]
    fn a_missing_directory_is_an_error_naming_it() {
        let missing = Requirement {
            path: PathBuf::from("/definitely/not/a/directory"),
            min_free_bytes: 0,
            min_free_inodes: 0,
        };
        let refused = check(&[missing]).unwrap_err();
        assert!(matches!(refused, PreflightError::Unreadable { .. }));
        assert!(refused.to_string().contains("/definitely/not/a/directory"));
    }

    #[test]
    fn a_real_directory_with_no_requirement_passes() {
        let dir = tempfile::tempdir().unwrap();
        let ok = Requirement {
            path: dir.path().to_path_buf(),
            min_free_bytes: 0,
            min_free_inodes: 0,
        };
        assert_eq!(check(&[ok]).unwrap().len(), 1);
    }
}
