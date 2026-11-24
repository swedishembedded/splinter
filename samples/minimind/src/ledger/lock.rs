// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements crash-safe, append-only state files for
// long-running learning systems, for its clients. If your team needs
// expertise in durable records that survive a killed process, you can procure
// our services by sending an email to info@swedishembedded.com.

//! An append-only JSON-lines pack held under an exclusive lock.
//!
//! A pack is one file of one JSON record per line, appended to and never
//! rewritten in place. A writer opens it through [`Locked::open`], which takes
//! an exclusive `flock`, cuts off a torn last line left by a crash (a line
//! without its newline was never complete), and reopens the file when another
//! process replaced it (a compaction) while this one waited for the lock.
//! [`Locked::append`] writes in one call and syncs, so a crash leaves either
//! the whole record or a torn tail the next writer cuts.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::fs::MetadataExt;
use std::os::unix::io::AsRawFd;
use std::path::Path;

/// An exclusive lock on a pack, released when dropped.
pub(super) struct Locked {
    pub(super) file: File,
}

impl Locked {
    /// The pack at `path` opened for reading and writing, created (with its
    /// parent directories) if absent, locked, with a torn last line cut off.
    pub(super) fn open(path: &Path) -> std::io::Result<Self> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        loop {
            let file = OpenOptions::new()
                .create(true)
                .read(true)
                .write(true)
                .truncate(false)
                .open(path)?;
            let mut locked = Self::lock(file)?;
            let held = locked.file.metadata()?;
            let current = std::fs::metadata(path)?;
            if (held.dev(), held.ino()) != (current.dev(), current.ino()) {
                continue;
            }
            let len = held.len();
            if len > 0 {
                let mut last = [0u8; 1];
                locked.file.seek(SeekFrom::Start(len - 1))?;
                locked.file.read_exact(&mut last)?;
                if last[0] != b'\n' {
                    let bytes = locked.read_all()?;
                    let sound = bytes.iter().rposition(|b| *b == b'\n').map_or(0, |i| i + 1);
                    locked.file.set_len(sound as u64)?;
                }
            }
            return Ok(locked);
        }
    }

    fn lock(file: File) -> std::io::Result<Self> {
        // SAFETY: `file` is an open descriptor for the lifetime of the call.
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) } != 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(Self { file })
    }

    /// Every byte of the pack.
    pub(super) fn read_all(&mut self) -> std::io::Result<Vec<u8>> {
        let mut bytes = Vec::new();
        self.file.seek(SeekFrom::Start(0))?;
        self.file.read_to_end(&mut bytes)?;
        Ok(bytes)
    }

    /// Appends `text` (whole lines, each ending in a newline) in one write and
    /// syncs it to disk.
    pub(super) fn append(&mut self, text: &str) -> std::io::Result<()> {
        debug_assert!(text.ends_with('\n'), "a pack holds whole lines only");
        self.file.seek(SeekFrom::End(0))?;
        self.file.write_all(text.as_bytes())?;
        self.file.sync_data()
    }
}

impl Drop for Locked {
    fn drop(&mut self) {
        // SAFETY: the descriptor is still open; closing it would release the
        // lock as well, so a failure here has nothing left to undo.
        unsafe { libc::flock(self.file.as_raw_fd(), libc::LOCK_UN) };
    }
}

/// The whole lines of `bytes`: a last line without its newline is a torn
/// write and is left out.
pub(super) fn whole_lines(bytes: &[u8]) -> impl Iterator<Item = &[u8]> {
    let sound = bytes.iter().rposition(|b| *b == b'\n').map_or(0, |i| i + 1);
    bytes[..sound]
        .split(|b| *b == b'\n')
        .filter(|line| !line.is_empty())
}
