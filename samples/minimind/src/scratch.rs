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
//! file system by default (when that file system lets programs run: agent
//! workspaces run their helpers, and a container may mount it `noexec`), created when the run opens and removed when it
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
    /// when the machine has one that lets programs run, else `scratch` in
    /// the run directory.
    #[must_use]
    pub fn default_root(run_dir: &Path) -> PathBuf {
        memory_root().unwrap_or_else(|| run_dir.join("scratch"))
    }

    /// A new private scratch directory under `root` for the run in
    /// `run_dir`, named after the run and this process.
    ///
    /// # Errors
    /// The directory cannot be made.
    pub fn create(root: &Path, run_dir: &Path) -> anyhow::Result<Self> {
        let run = std::fs::canonicalize(run_dir).unwrap_or_else(|_| run_dir.to_path_buf());
        let id = splinter_sdk::vocabulary::digest::Digest::of(run.as_os_str().as_encoded_bytes());
        let prefix = format!("minimind-{}-", &id.hex()[..12]);
        // What a run of this directory that was killed outright left.
        if let Ok(entries) = std::fs::read_dir(root) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().into_owned();
                let dead = name
                    .strip_prefix(&prefix)
                    .and_then(|pid| pid.parse::<u32>().ok())
                    .is_some_and(|pid| !Path::new(&format!("/proc/{pid}")).exists());
                if dead {
                    let _ = std::fs::remove_dir_all(entry.path());
                }
            }
        }
        let dir = root.join(format!("{prefix}{}", std::process::id()));
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

/// The memory-backed file system, when the machine has one that lets
/// programs run.
#[must_use]
pub fn memory_root() -> Option<PathBuf> {
    let mounts = std::fs::read_to_string("/proc/mounts").unwrap_or_default();
    let memory = Path::new(MEMORY);
    (memory.is_dir() && runs_programs(&mounts, memory)).then(|| memory.to_path_buf())
}

/// Whether the file system `path` is on lets programs run, as `mounts` (the
/// text of `/proc/mounts`) says: the mount whose point is the longest
/// prefix of `path` is not `noexec`. Without a mount table, it is assumed
/// to.
fn runs_programs(mounts: &str, path: &Path) -> bool {
    mounts
        .lines()
        .filter_map(|line| {
            let mut fields = line.split_whitespace();
            let point = fields.nth(1)?;
            let options = fields.nth(1)?;
            path.starts_with(point).then_some((point.len(), options))
        })
        .max_by_key(|(len, _)| *len)
        .is_none_or(|(_, options)| !options.split(',').any(|o| o == "noexec"))
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
    fn scratch_goes_only_where_programs_may_run() {
        let mounts = "overlay / overlay rw,relatime 0 0\n\
                      shm /dev/shm tmpfs rw,nosuid,nodev,noexec,relatime 0 0\n";
        assert!(!runs_programs(mounts, Path::new("/dev/shm")));
        assert!(runs_programs(mounts, Path::new("/tmp")));
        let open = "shm /dev/shm tmpfs rw,nosuid,nodev,relatime 0 0\n";
        assert!(runs_programs(open, Path::new("/dev/shm")));
        assert!(runs_programs("", Path::new("/dev/shm")), "no mount table");
    }

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
        // A killed run's scratch (no process has its id) goes with the next.
        let left = dir.with_file_name(format!(
            "{}{}",
            dir.file_name()
                .unwrap()
                .to_string_lossy()
                .rsplit_once('-')
                .unwrap()
                .0,
            "-4294967295"
        ));
        std::fs::create_dir_all(&left).unwrap_or_else(|e| panic!("{e}"));
        let again = Scratch::create(root.path(), root.path()).unwrap_or_else(|e| panic!("{e}"));
        assert!(!left.exists(), "{}", left.display());
        drop(again);
    }
}
