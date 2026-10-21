// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements sandboxed execution of untrusted code for
// agents, for its clients. If your team needs expertise in running agent
// workloads without leaving state behind on shared machines, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Scratch directories owned by a process, and the removal of the ones a
//! killed process left behind.
//!
//! A work tree holds thousands of small files, and a shared drive can have an
//! inode quota, so a tree must not outlive the work it was made for. A
//! [`Claim`] removes its directory when dropped. A killed process cannot drop
//! anything, so every claim also holds an advisory lock on a marker file
//! inside the directory; [`sweep`] removes the directories whose marker
//! nobody holds. The lock is released by the
//! kernel when the holder dies, which is what makes "not held" mean "dead".

use std::fs::{File, TryLockError};
use std::io;
use std::path::Path;
use std::time::{Duration, SystemTime};

/// Name of the marker file a live claim locks.
pub const LIVE_FILE: &str = ".live";

/// A directory without a marker is either being created right now or left by
/// a binary that did not write markers; it is swept only when older than this.
const MARKERLESS_GRACE: Duration = Duration::from_secs(10 * 60);

/// A scratch directory owned by this process until dropped.
#[derive(Debug)]
pub struct Claim {
    dir: tempfile::TempDir,
    // Held for the life of the claim: the lock is what tells `sweep` we are alive.
    _live: File,
}

impl Claim {
    /// Creates `<root>/<prefix>XXXXXX` and locks its marker.
    pub fn create(root: &Path, prefix: &str) -> io::Result<Claim> {
        std::fs::create_dir_all(root)?;
        let dir = tempfile::Builder::new().prefix(prefix).tempdir_in(root)?;
        let live = File::create(dir.path().join(LIVE_FILE))?;
        live.lock()?;
        Ok(Claim { dir, _live: live })
    }

    /// The directory.
    pub fn path(&self) -> &Path {
        self.dir.path()
    }
}

/// Removes the directories under `root` whose names start with one of
/// `prefixes` and that no live claim holds, and returns how many it removed.
/// Other directories are left alone.
pub fn sweep(root: &Path, prefixes: &[&str]) -> io::Result<usize> {
    let entries = match std::fs::read_dir(root) {
        Ok(entries) => entries,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(0),
        Err(e) => return Err(e),
    };
    let mut removed = 0;
    for entry in entries {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let path = entry.path();
        if !prefixes.iter().any(|p| name.starts_with(p)) || !path.is_dir() {
            continue;
        }
        if is_stale(&path)? {
            std::fs::remove_dir_all(&path)?;
            removed += 1;
        }
    }
    Ok(removed)
}

fn is_stale(dir: &Path) -> io::Result<bool> {
    match File::open(dir.join(LIVE_FILE)) {
        Ok(marker) => match marker.try_lock() {
            Ok(()) => Ok(true),
            Err(TryLockError::WouldBlock) => Ok(false),
            Err(TryLockError::Error(e)) => Err(e),
        },
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            let modified = std::fs::metadata(dir)?.modified()?;
            Ok(SystemTime::now()
                .duration_since(modified)
                .is_ok_and(|age| age > MARKERLESS_GRACE))
        }
        Err(e) => Err(e),
    }
}
