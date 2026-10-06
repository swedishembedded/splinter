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
//!   expdb/        the experience database: sources, tasks, experiences,
//!                 annotations, sets, runs, dataset and release manifests,
//!                 candidates, answers, suites, calibrations, the curriculum,
//!                 and the pointers (aliases, the anchor in force)
//!   artifacts/    the bulk files tools need, by content address: adapters,
//!                 the records brain trains on, replayed records
//!   work/         work in progress, removed after: training scratch, the
//!                 sandbox's call directories
//! ```
//!
//! Everything but the bulk files is in the experience database, through one
//! shared [`workspace::Workspace`], so a snapshot of it is the state, and one
//! commit makes a change official whole. A bulk file is a plain file at a
//! stable path a tool opens directly, written before the commit that makes it
//! official ([`artifacts`]). [`runs`] holds a command's run record, readable and
//! cancellable from any process; [`pointers`] the few names that move, with
//! their history; [`documents`] what is kept by its digest.
//!
//! [`source`] and [`sources`] are what Splinter learns from: every
//! document, repository and command run, its content stored once per
//! digest, so a span resolves to the exact bytes it names. [`tasks`] holds
//! the tasks generated from them and the named sets a stage hands the
//! next. [`experiences`] is the experience store every training set is
//! projected from: immutable experiences under their content address,
//! append-only annotations beside them, and named sets of experience ids;
//! [`decision`] is the rule that turns an experience's verdicts into one
//! decision. The sources, tasks, experiences and annotations themselves are
//! `splinter-core`'s vocabulary. [`error`] is the error every store reports.

#![warn(missing_docs)]

pub mod address;
pub mod artifacts;
pub mod decision;
pub mod documents;
pub mod error;
pub mod experiences;
pub mod lineage;
pub mod longitudinal;
pub mod maintenance;
pub mod pointers;
mod projection;
pub mod recovery;
pub mod runs;
pub mod sources;
pub mod tasks;
pub mod workspace;

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

    /// Where the bulk files tools need are kept, by content address.
    #[must_use]
    pub fn artifacts(&self) -> PathBuf {
        self.0.join("artifacts")
    }

    /// Where work in progress is done: files that exist only while a command
    /// runs, and are removed after.
    #[must_use]
    pub fn work(&self) -> PathBuf {
        self.0.join("work")
    }

    /// The experience database's directory.
    #[must_use]
    pub fn expdb(&self) -> PathBuf {
        self.0.join("expdb")
    }

    /// The process sandbox's scratch root: a directory per code call.
    #[must_use]
    pub fn sandbox(&self) -> PathBuf {
        self.work().join("sandbox")
    }
}

/// A new id led by `prefix`: time-ordered, so a directory listing reads as
/// a history, with a sub-second suffix so two ids made in the same
/// millisecond are unlikely to collide.
#[must_use]
pub fn new_id_with_prefix(prefix: &str) -> String {
    let t = splinter_core::clock::now();
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
    }
}
