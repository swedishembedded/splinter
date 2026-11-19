// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements durable learning state that fits on
// quota-limited shared filesystems, for its clients. If your team needs
// expertise in migrating stores in place without losing a byte, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Rewriting a state root's experience database into the packed layout, in
//! place (`splinter state pack`).
//!
//! The new database is built and verified beside the old one, under
//! `expdb.packing`; only then is the old one moved aside to `expdb.loose` and
//! the new one moved into place, and the old one deleted. A crash before the
//! first move leaves the old database in use and a staging directory the
//! next pack replaces; a crash between the moves leaves no `expdb`, which
//! [`settle`] (run before any database is opened) completes from the
//! verified staging copy; a crash after them leaves the old copy, which
//! `settle` deletes.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::Serialize;
use splinter_expdb::backend::{repack, Layout};

use crate::error::StoreError;
use crate::workspace::Workspace;
use crate::StateRoot;

/// What `state pack` reports.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct StatePacked {
    /// The state root.
    pub state: PathBuf,
    /// The layout the database was in: `loose` or `packed` (a packed one is
    /// compacted: only live objects are copied).
    pub from: &'static str,
    /// Objects copied and verified.
    pub objects: u64,
    /// Their bytes.
    pub bytes: u64,
    /// Files the database held before.
    pub files_before: u64,
    /// Files it holds now.
    pub files_after: u64,
}

fn staging(root: &StateRoot) -> PathBuf {
    root.path().join("expdb.packing")
}

fn retired(root: &StateRoot) -> PathBuf {
    root.path().join("expdb.loose")
}

fn io_error(path: &Path) -> impl FnOnce(io::Error) -> StoreError + '_ {
    move |source| StoreError::Io {
        path: path.to_path_buf(),
        source,
    }
}

fn sync_dir(dir: &Path) -> Result<(), StoreError> {
    fs::File::open(dir)
        .and_then(|d| d.sync_all())
        .map_err(io_error(dir))
}

fn remove_tree(path: &Path) -> Result<(), StoreError> {
    match fs::remove_dir_all(path) {
        Ok(()) => Ok(()),
        // Another opener's settle got there first.
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(io_error(path)(e)),
    }
}

/// Whether a `state pack` died before it finished, so the database to open
/// is still to be settled.
pub(crate) fn interrupted(root: &StateRoot) -> bool {
    retired(root).exists()
}

/// Finishes a `state pack` that died between moving the old database aside
/// and deleting it. Cheap when there is nothing to finish: two existence
/// checks.
pub(crate) fn settle(root: &StateRoot) -> Result<(), StoreError> {
    let (db, retired) = (root.expdb(), retired(root));
    if !retired.exists() {
        return Ok(());
    }
    if !db.exists() {
        // Died between the two moves: the staging copy was verified before
        // the first, so it goes into place. Without one, the old database
        // goes back.
        let staging = staging(root);
        let from = if staging.exists() {
            staging
        } else {
            retired.clone()
        };
        match fs::rename(&from, &db) {
            Ok(()) => sync_dir(root.path())?,
            // Another opener settled it first.
            Err(e) if e.kind() == io::ErrorKind::NotFound && db.exists() => {}
            Err(e) => return Err(io_error(&db)(e)),
        }
    }
    remove_tree(&retired)
}

impl Workspace {
    /// Rewrites this workspace's database into a new packed one and puts it
    /// in place; see the module documentation. Refused once this workspace
    /// has opened the database, whose handle would go on naming the old one.
    /// Every process using the state must be stopped first: a write to the
    /// old database while it is copied is detected and the pack refused, and
    /// the old database is kept.
    pub fn pack_state(&self) -> Result<StatePacked, StoreError> {
        if self.opened() {
            return Err(StoreError::Recovery(
                "this process already opened the database; pack the state from a fresh process"
                    .into(),
            ));
        }
        let root = self.root();
        settle(root)?;
        let db = root.expdb();
        let from = match Layout::of_root(&db)? {
            Some(Layout::Loose) => "loose",
            Some(Layout::Packed) => "packed",
            None => {
                return Err(StoreError::Recovery(format!(
                    "{} holds no experience database to pack",
                    root.path().display()
                )))
            }
        };
        let staging = staging(root);
        remove_tree(&staging)?;
        let packed = match repack(&db, &staging) {
            Ok(packed) => packed,
            Err(e) => {
                let _ = fs::remove_dir_all(&staging);
                return Err(e.into());
            }
        };
        sync_dir(root.path())?;
        let retired = retired(root);
        fs::rename(&db, &retired).map_err(io_error(&db))?;
        sync_dir(root.path())?;
        match fs::rename(&staging, &db) {
            Ok(()) => sync_dir(root.path())?,
            // A process that opened the state just now settled it first.
            Err(e) if e.kind() == io::ErrorKind::NotFound && db.exists() => {}
            Err(e) => return Err(io_error(&db)(e)),
        }
        remove_tree(&retired)?;
        Ok(StatePacked {
            state: root.path().to_path_buf(),
            from,
            objects: packed.objects,
            bytes: packed.bytes,
            files_before: packed.files_before,
            files_after: packed.files_after,
        })
    }
}
