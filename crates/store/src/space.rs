// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning pipelines that check their storage
// before an expensive run, for its clients. If your team needs expertise in
// reliable long-running training jobs, you can procure our services by
// sending an email to info@swedishembedded.com.

//! How much room a directory's filesystem has left, in bytes and in files, so
//! a run that writes adapters and datasets can say so before it starts rather
//! than fail with a full disk or an exhausted file quota part-way through.

use std::ffi::CString;
use std::io;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;

/// What a filesystem has left, and what it started with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Headroom {
    /// Bytes a non-privileged writer can still create.
    pub free_bytes: u64,
    /// Files a non-privileged writer can still create; `None` on a
    /// filesystem that does not count them.
    pub free_inodes: Option<u64>,
    /// Files the filesystem (or the quota on it) allows in all; `None` when
    /// it does not count them.
    pub total_inodes: Option<u64>,
}

/// The room the filesystem holding `path` has left, in bytes and in files.
///
/// A run can die of either: a shared project disk with a quota on the number
/// of files killed registered runs with gigabytes free. `path` must exist; an
/// unreadable or missing path is the error the operating system gave, naming
/// nothing the caller did not pass.
pub fn headroom(path: &Path) -> io::Result<Headroom> {
    let c_path = CString::new(path.as_os_str().as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "path contains a NUL byte"))?;
    let mut stat = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: `c_path` is a valid NUL-terminated string and `stat` is a
    // writable `statvfs` the call fully initialises when it returns 0.
    if unsafe { libc::statvfs(c_path.as_ptr(), stat.as_mut_ptr()) } != 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: statvfs returned 0, so the structure is initialised.
    let stat = unsafe { stat.assume_init() };
    // The fields are 32 or 64 bits wide by platform; widen so one expression
    // is right for both.
    let bytes = u128::from(stat.f_bavail).saturating_mul(u128::from(stat.f_frsize));
    let counted = stat.f_files != 0;
    // The counts are 32 or 64 bits wide by platform, as the byte fields are.
    #[allow(clippy::useless_conversion)]
    let (free_files, all_files) = (u64::from(stat.f_favail), u64::from(stat.f_files));
    Ok(Headroom {
        free_bytes: u64::try_from(bytes).unwrap_or(u64::MAX),
        free_inodes: counted.then_some(free_files),
        total_inodes: counted.then_some(all_files),
    })
}

/// Bytes a non-privileged writer can still create on the filesystem holding
/// `path`; see [`headroom`].
pub fn available_bytes(path: &Path) -> io::Result<u64> {
    headroom(path).map(|room| room.free_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_existing_directory_reports_its_free_space() {
        let dir = tempfile::tempdir().unwrap();
        assert!(available_bytes(dir.path()).unwrap() > 0);
    }

    #[test]
    fn headroom_counts_files_as_well_as_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let room = headroom(dir.path()).unwrap();
        assert!(room.free_bytes > 0);
        if let (Some(free), Some(total)) = (room.free_inodes, room.total_inodes) {
            assert!(free <= total, "{free} free of {total}");
        }
    }

    #[test]
    fn a_missing_path_is_the_operating_systems_error() {
        let dir = tempfile::tempdir().unwrap();
        let err = available_bytes(&dir.path().join("absent")).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
    }
}
