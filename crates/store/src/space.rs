// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning pipelines that check their storage
// before an expensive run, for its clients. If your team needs expertise in
// reliable long-running training jobs, you can procure our services by
// sending an email to info@swedishembedded.com.

//! How much room a directory's filesystem has left, so a run that writes
//! adapters and datasets can say so before it starts rather than fail with
//! a full disk part-way through.

use std::ffi::CString;
use std::io;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;

/// Bytes a non-privileged writer can still create on the filesystem holding
/// `path`. `path` must exist; an unreadable or missing path is the error the
/// operating system gave, naming nothing the caller did not pass.
pub fn available_bytes(path: &Path) -> io::Result<u64> {
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
    Ok(u64::try_from(bytes).unwrap_or(u64::MAX))
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
    fn a_missing_path_is_the_operating_systems_error() {
        let dir = tempfile::tempdir().unwrap();
        let err = available_bytes(&dir.path().join("absent")).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
    }
}
