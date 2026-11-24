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
