// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Splinter's durable state: where everything lives, and how it is written.
//!
//! Everything Splinter records sits under one [`StateRoot`], passed in by
//! whoever owns the configuration - never read from the environment here, so
//! a test or a second campaign gets its own root without touching process
//! state. Layout:
//!
//! ```text
//! <root>/
//!   runs/<run_id>/            one command's run record and cancel request
//!   sources/                  the content-addressed source store
//!   tasks/                    the content-addressed task store
//!   experiences/              the content-addressed experience store
//!   datasets/<id>/            one dataset and its manifest
//!   calibrations/             each judge's latest calibration
//!   train/<candidate_id>/     one trained candidate: adapter, scores, record
//!   releases/<hex>/           one immutable release: its adapter and manifest
//!   releases/aliases/<name>   the release an alias (`default`) points at
//!   answers/<hex>.json        one answer `ask` gave, and the release it used
//!   suites/anchor/            the frozen anchor suite's versions
//!   curriculum/               pass@k measurements and concepts queued for generation
//!   sandbox/                  the process sandbox's per-call directories
//! ```
//!
//! Every file a reader acts on is written with [`write_atomic`]: a status
//! half-written by a crash must never read as a status. A content-addressed
//! object is written with [`write_once`], which never replaces a file.
//! [`runs`] holds a command's run record, readable and cancellable from any
//! process.
//!
//! [`source`] and [`sources`] are what Splinter learns from: every
//! document, repository and command run, its content stored once per
//! digest, so a span resolves to the exact bytes it names. [`tasks`] holds
//! the tasks generated from them and the named sets a stage hands the
//! next. [`experience`], [`annotation`] and [`experiences`] are the
//! experience store every training set is projected from: immutable
//! experiences under their content address, append-only annotations beside
//! them, and named sets of experience ids. [`error`] is the error every
//! store reports.

#![warn(missing_docs)]

pub mod annotation;
pub mod clock;
pub mod digest;
pub mod error;
pub mod experience;
pub mod experiences;
pub mod runs;
pub mod source;
pub mod sources;
pub mod tasks;

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

/// The directory all of Splinter's durable state lives under.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateRoot(PathBuf);

impl StateRoot {
    /// A state root at `path`.
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self(path.into())
    }

    /// The default root for a user whose home directory is `home`:
    /// `<home>/.sven/splinter`, a namespace of its own beside sven's files.
    #[must_use]
    pub fn under_home(home: &Path) -> Self {
        Self(home.join(".sven").join("splinter"))
    }

    /// The root directory itself.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.0
    }

    /// The directory holding every run.
    #[must_use]
    pub fn runs(&self) -> PathBuf {
        self.0.join("runs")
    }

    /// One run's directory.
    #[must_use]
    pub fn run_dir(&self, run_id: &str) -> PathBuf {
        self.runs().join(run_id)
    }

    /// The source store's directory.
    #[must_use]
    pub fn sources(&self) -> PathBuf {
        self.0.join("sources")
    }

    /// The task store's directory.
    #[must_use]
    pub fn tasks(&self) -> PathBuf {
        self.0.join("tasks")
    }

    /// The experience store's directory.
    #[must_use]
    pub fn experiences(&self) -> PathBuf {
        self.0.join("experiences")
    }

    /// Where datasets and their manifests live.
    #[must_use]
    pub fn datasets(&self) -> PathBuf {
        self.0.join("datasets")
    }

    /// Where each judge's latest calibration lives.
    #[must_use]
    pub fn calibrations(&self) -> PathBuf {
        self.0.join("calibrations")
    }

    /// Where trained candidates live.
    #[must_use]
    pub fn train(&self) -> PathBuf {
        self.0.join("train")
    }

    /// Where releases and the aliases pointing at them live.
    #[must_use]
    pub fn releases(&self) -> PathBuf {
        self.0.join("releases")
    }

    /// Where the answers `ask` gave are recorded.
    #[must_use]
    pub fn answers(&self) -> PathBuf {
        self.0.join("answers")
    }

    /// Where frozen evaluation suites live.
    #[must_use]
    pub fn suites(&self) -> PathBuf {
        self.0.join("suites")
    }

    /// Where the curriculum keeps its pass@k measurements and the concepts
    /// queued for new tasks.
    #[must_use]
    pub fn curriculum(&self) -> PathBuf {
        self.0.join("curriculum")
    }

    /// The process sandbox's scratch root: a directory per code call.
    #[must_use]
    pub fn sandbox(&self) -> PathBuf {
        self.0.join("sandbox")
    }
}

/// A new id led by `prefix`: time-ordered, so a directory listing reads as
/// a history, with a sub-second suffix so two ids made in the same
/// millisecond are unlikely to collide.
#[must_use]
pub fn new_id_with_prefix(prefix: &str) -> String {
    let t = clock::now();
    let rand = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    format!(
        "{}-{:04}{:02}{:02}T{:02}{:02}{:02}.{:03}-{:04x}",
        prefix,
        t.year,
        t.month,
        t.day,
        t.hour,
        t.min,
        t.sec,
        t.millis,
        rand & 0xffff
    )
}

/// Writes `text` to `path` atomically: a temporary file in the same
/// directory, fsync, rename. Creates the parent directory.
pub fn write_atomic(path: &Path, text: &str) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("tmp");
    {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(text.as_bytes())?;
        f.sync_all()?;
    }
    fs::rename(&tmp, path)?;
    Ok(())
}

/// Creates `path` holding `bytes`, unless it already exists; returns
/// whether it was created. Never replaces an existing file, and never
/// exposes a partial one: the bytes go to a uniquely named temporary file in
/// the same directory, are fsynced, and are hard-linked into place (which
/// fails rather than replaces when `path` exists), then the directory entry
/// is fsynced. Two writers racing on one path both succeed, and exactly one
/// of them reports the creation.
pub fn write_once(path: &Path, bytes: &[u8]) -> std::io::Result<bool> {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let parent = path.parent().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("{} has no parent directory", path.display()),
        )
    })?;
    fs::create_dir_all(parent)?;
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("object");
    let tmp = parent.join(format!(
        ".{name}.{}.{}.tmp",
        std::process::id(),
        SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let linked = (|| {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
        match fs::hard_link(&tmp, path) {
            Ok(()) => Ok(true),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Ok(false),
            Err(e) => Err(e),
        }
    })();
    let removed = fs::remove_file(&tmp);
    let created = linked?;
    removed?;
    if created {
        sync_dir(parent)?;
    }
    Ok(created)
}

/// Fsyncs a directory, so an entry just created in it survives a crash.
pub fn sync_dir(dir: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    fs::File::open(dir)?.sync_all()?;
    #[cfg(not(unix))]
    let _ = dir;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_ids_are_time_ordered_and_unique_within_a_second() {
        let a = new_id_with_prefix("run");
        std::thread::sleep(std::time::Duration::from_millis(1));
        let b = new_id_with_prefix("run");
        assert!(a < b, "ids must sort as a history: {a} vs {b}");
    }

    /// The default root is Splinter's own namespace under `~/.sven/`,
    /// beside - never inside - sven's files.
    #[test]
    fn the_default_root_is_namespaced_under_the_home_directory() {
        let home = Path::new("home-dir");
        let root = StateRoot::under_home(home);
        assert_eq!(root.path(), home.join(".sven").join("splinter"));
        assert_eq!(root.run_dir("r1"), root.path().join("runs").join("r1"));
    }

    #[test]
    fn write_atomic_leaves_no_tmp_file_behind() {
        let dir = std::env::temp_dir().join(format!("splinter-store-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("run.json");
        write_atomic(&path, "{\"status\":\"pending\"}").unwrap();
        assert_eq!(
            fs::read_to_string(&path).unwrap(),
            "{\"status\":\"pending\"}"
        );
        let leftovers: Vec<String> = fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.unwrap().file_name().into_string().ok())
            .filter(|n| n.ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
        fs::remove_dir_all(&dir).unwrap();
    }
}
