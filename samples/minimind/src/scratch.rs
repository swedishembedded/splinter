// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training pipelines that keep their
// short-lived files off quota-bound shared storage, for its clients. If your
// team needs expertise in running large experiments on shared file systems,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! The run's scratch directory: where the files that live for one episode or
//! one stage are made (agent workspaces, an episode's files until its stage
//! packs them, the task folders before they are packed), so the run
//! directory on shared storage only ever holds a stage's single archive.
//!
//! It is one private directory (mode 700) per process, under a memory-backed
//! file system by default, created when the run opens and removed when it
//! ends, a panic's unwinding included. Nothing in it is needed to resume: a
//! run killed outright leaves its archives, which hold every finished
//! episode, and the next run unpacks what it needs into its own scratch.

use std::os::unix::fs::PermissionsExt as _;
use std::path::{Path, PathBuf};

use anyhow::Context as _;

/// The memory-backed file system scratch goes to when it exists.
const MEMORY: &str = "/dev/shm";

/// One run's scratch directory, removed when dropped.
#[derive(Debug)]
pub struct Scratch {
    dir: PathBuf,
}

impl Scratch {
    /// Where scratch goes when none is named: the memory-backed file system
    /// when the machine has one, else `scratch` in the run directory.
    #[must_use]
    pub fn default_root(run_dir: &Path) -> PathBuf {
        if Path::new(MEMORY).is_dir() {
            PathBuf::from(MEMORY)
        } else {
            run_dir.join("scratch")
        }
    }

    /// A new private scratch directory under `root` for the run in
    /// `run_dir`, named after the run and this process.
    ///
    /// # Errors
    /// The directory cannot be made.
    pub fn create(root: &Path, run_dir: &Path) -> anyhow::Result<Self> {
        let run = std::fs::canonicalize(run_dir).unwrap_or_else(|_| run_dir.to_path_buf());
        let id = splinter_sdk::vocabulary::digest::Digest::of(run.as_os_str().as_encoded_bytes());
        let dir = root.join(format!(
            "minimind-{}-{}",
            &id.hex()[..12],
            std::process::id()
        ));
        if dir.exists() {
            std::fs::remove_dir_all(&dir).with_context(|| format!("clearing {}", dir.display()))?;
        }
        std::fs::create_dir_all(&dir).with_context(|| format!("making {}", dir.display()))?;
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700))
            .with_context(|| format!("making {} private", dir.display()))?;
        Ok(Self { dir })
    }

    /// The scratch directory.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// `rel` inside the scratch directory.
    #[must_use]
    pub fn path(&self, rel: &str) -> PathBuf {
        self.dir.join(rel)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        // Best effort: what is left behind is in memory, and the next run
        // makes its own.
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scratch_is_private_and_gone_when_the_run_ends() {
        let root = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let scratch = Scratch::create(root.path(), root.path()).unwrap_or_else(|e| panic!("{e}"));
        let dir = scratch.dir().to_path_buf();
        std::fs::write(scratch.path("f"), "x").unwrap_or_else(|e| panic!("{e}"));
        let mode = std::fs::metadata(&dir)
            .unwrap_or_else(|e| panic!("{e}"))
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o700);
        drop(scratch);
        assert!(!dir.exists());
    }
}
