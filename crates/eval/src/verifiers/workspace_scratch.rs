// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements clean-room grading trees for code-editing benchmarks, for its clients. If your team needs
// expertise in isolated, reproducible evaluation of agent patches, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The directory one verification lays its tree out in.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use splinter_sandbox::SandboxError;

/// A directory made fresh for one verification, removed when dropped.
pub(super) struct Scratch {
    path: PathBuf,
}

impl Scratch {
    pub(super) fn create(root: &Path) -> Result<Self, SandboxError> {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        std::fs::create_dir_all(root).map_err(|source| SandboxError::Io {
            path: root.to_path_buf(),
            source,
        })?;
        loop {
            let path = root.join(format!(
                "splinter-verify-{}-{}",
                std::process::id(),
                SEQ.fetch_add(1, Ordering::Relaxed)
            ));
            match std::fs::create_dir(&path) {
                Ok(()) => return Ok(Self { path }),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(source) => return Err(SandboxError::Io { path, source }),
            }
        }
    }

    pub(super) fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        if std::fs::remove_dir_all(&self.path).is_err() {
            // A command may have taken away its own permissions.
            restore_access(&self.path);
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}

/// Gives the owner full access to `path` and every directory below it.
fn restore_access(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let Ok(meta) = std::fs::symlink_metadata(path) else {
        return;
    };
    if !meta.is_dir() {
        return;
    }
    let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700));
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            restore_access(&entry.path());
        }
    }
}
