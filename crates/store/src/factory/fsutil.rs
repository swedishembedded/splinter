// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements crash-safe file publication for
// long-running automation, for its clients. If your team needs expertise in
// durable file-system protocols, you can procure our services by sending an
// email to info@swedishembedded.com.

//! The write-then-rename protocol the factory's files share.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static UNIQUE: AtomicU64 = AtomicU64::new(0);

/// A sibling name for `path` that no other writer in any process uses: the
/// process id and a counter keep concurrent publishers apart.
pub(super) fn temp_sibling(path: &Path) -> PathBuf {
    let name = path
        .file_name()
        .map_or_else(|| "tmp".into(), |n| n.to_string_lossy().into_owned());
    let n = UNIQUE.fetch_add(1, Ordering::Relaxed);
    path.with_file_name(format!(".{name}.tmp-{}-{n}", std::process::id()))
}

/// Flushes the directory entry of `dir` so a rename inside it survives a
/// power loss.
pub(super) fn sync_dir(dir: &Path) -> io::Result<()> {
    File::open(dir)?.sync_all()
}

/// Whether two open files are one inode (a file and its hard link).
pub(super) fn same_file(a: &File, b: &File) -> io::Result<bool> {
    use std::os::unix::fs::MetadataExt;
    let (a, b) = (a.metadata()?, b.metadata()?);
    Ok((a.dev(), a.ino()) == (b.dev(), b.ino()))
}

/// Writes `bytes` to `path` only when nothing is there: the bytes go to a
/// temporary sibling, are synced, and are linked into place, which fails
/// when `path` exists, so a reader sees the whole file or none and a file
/// already in place is never replaced. Whether this call created it.
pub(super) fn write_new(path: &Path, bytes: &[u8]) -> io::Result<bool> {
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(dir)?;
    let temp = temp_sibling(path);
    let written = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        file.write_all(bytes)?;
        file.sync_all()
    })();
    if let Err(error) = written {
        let _ = fs::remove_file(&temp);
        return Err(error);
    }
    crate::faults::boundary("new write: temporary synced");
    let linked = fs::hard_link(&temp, path);
    let _ = fs::remove_file(&temp);
    match linked {
        Ok(()) => {
            crate::faults::boundary("new write: linked");
            sync_dir(dir)?;
            crate::faults::boundary("new write: directory synced");
            Ok(true)
        }
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => Ok(false),
        Err(error) => Err(error),
    }
}

/// Writes `bytes` to `path` atomically: a temporary sibling is written and
/// synced, renamed into place, and the directory synced. Readers see the old
/// file or the whole new one, never a torn one.
pub(super) fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(dir)?;
    let temp = temp_sibling(path);
    let written = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        file.write_all(bytes)?;
        file.sync_all()
    })();
    if let Err(error) = written {
        let _ = fs::remove_file(&temp);
        return Err(error);
    }
    crate::faults::boundary("atomic write: temporary synced");
    if let Err(error) = fs::rename(&temp, path) {
        let _ = fs::remove_file(&temp);
        return Err(error);
    }
    crate::faults::boundary("atomic write: renamed");
    sync_dir(dir)?;
    crate::faults::boundary("atomic write: directory synced");
    Ok(())
}
