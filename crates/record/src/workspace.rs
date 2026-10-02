// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The workspace: the one open experience database of a process, shared by
//! every store that reads and writes it.
//!
//! Each store used to own its directory. They now share a [`Workspace`], so
//! one process is one writer however many stores it opens, and what one
//! store wrote another reads at once.
//!
//! # Durability
//!
//! A write is made durable and visible to other processes before the call
//! returns, unless the workspace is in a [`Batch`]: bulk ingest groups its
//! writes into commits of at most [`GROUP_COMMIT_WRITES`] writes or
//! [`GROUP_COMMIT_INTERVAL`] of waiting, and commits what is left at the end
//! or on [`Batch::commit`]. Every commit costs several file syncs, so one per
//! record is the price of a one-off write and too dear for a stage that writes
//! thousands. A crash inside a batch loses at most the uncommitted group and
//! nothing else; a record is never half written.

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use splinter_expdb::{Config, Database, Session, WriterIdentity};

use crate::digest::Digest;
use crate::error::StoreError;
use crate::StateRoot;

/// Writes a batch groups into one commit.
pub const GROUP_COMMIT_WRITES: usize = 256;
/// The longest a batch holds a write before committing it.
pub const GROUP_COMMIT_INTERVAL: Duration = Duration::from_secs(2);

/// What a batch has written and not yet committed.
struct Group {
    open_batches: usize,
    writes: usize,
    since: Instant,
}

struct Shared {
    root: StateRoot,
    session: Mutex<Option<Session>>,
    group: Mutex<Group>,
}

impl Drop for Shared {
    fn drop(&mut self) {
        // Best effort: a writer that is going away must not lose what it
        // was asked to keep, and a destructor has nowhere to report a failure.
        if let Some(session) = self
            .session
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner)
            .as_mut()
        {
            let _ = session.flush();
        }
    }
}

/// An open experience database. Cheap to clone; clones are the same writer.
#[derive(Clone)]
pub struct Workspace {
    shared: Arc<Shared>,
}

impl std::fmt::Debug for Workspace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Workspace")
    }
}

/// Writes made while it lives are committed together when it ends.
#[must_use = "a batch commits when it is dropped; hold it for the writes it should group"]
pub struct Batch {
    workspace: Workspace,
    done: bool,
}

impl Batch {
    /// Commits the batch now and reports a failure, which dropping cannot.
    pub fn commit(mut self) -> Result<(), StoreError> {
        self.done = true;
        self.workspace.leave_batch()
    }
}

impl Drop for Batch {
    fn drop(&mut self) {
        if !self.done {
            let _ = self.workspace.leave_batch();
        }
    }
}

fn locked<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The database id of the object `address` names.
pub(crate) fn content_id(address: &Digest) -> Result<splinter_expdb::ContentId, StoreError> {
    address.content_id().ok_or_else(|| StoreError::Rejected {
        what: "address",
        reason: format!("{address} is not a content address"),
    })
}

impl Workspace {
    /// The database under `root`. Nothing is opened or created until the
    /// first read or write, so a command that only plans leaves the root
    /// untouched.
    #[must_use]
    pub fn at(root: &StateRoot) -> Self {
        Self {
            shared: Arc::new(Shared {
                root: root.clone(),
                session: Mutex::new(None),
                group: Mutex::new(Group {
                    open_batches: 0,
                    writes: 0,
                    since: Instant::now(),
                }),
            }),
        }
    }

    /// Runs `f` against the session, opening the database first if this is
    /// the first use.
    fn with_session<R>(
        &self,
        f: impl FnOnce(&mut Session) -> splinter_expdb::Result<R>,
    ) -> Result<R, StoreError> {
        let mut slot = locked(&self.shared.session);
        if slot.is_none() {
            let db = Database::open(self.shared.root.expdb(), Config::default())?;
            let identity = WriterIdentity::new("splinter", "state", "local", 0);
            *slot = Some(Session::open(&db, &identity)?);
        }
        match slot.as_mut() {
            Some(session) => Ok(f(session)?),
            None => unreachable!("the session was opened above"),
        }
    }

    /// The entity of `class` whose address is `id`.
    pub(crate) fn find(
        &self,
        class: &str,
        id: &Digest,
    ) -> Result<Option<splinter_expdb::model::Entity>, StoreError> {
        let cid = content_id(id)?;
        let found = self.read(|s| s.entity(&cid))?;
        Ok(found.filter(|e| e.class == class))
    }

    /// The addresses of every entity of `class`, in address order.
    pub(crate) fn ids_of(&self, class: &str) -> Result<Vec<Digest>, StoreError> {
        let mut ids: Vec<Digest> = self
            .read(|s| s.entities(class))?
            .into_iter()
            .map(|stored| Digest::from(stored.id))
            .collect();
        ids.sort();
        Ok(ids)
    }

    /// Groups the writes made until the returned guard is dropped (or
    /// committed) into one commit.
    pub fn batch(&self) -> Batch {
        locked(&self.shared.group).open_batches += 1;
        Batch {
            workspace: self.clone(),
            done: false,
        }
    }

    fn leave_batch(&self) -> Result<(), StoreError> {
        let mut group = locked(&self.shared.group);
        group.open_batches = group.open_batches.saturating_sub(1);
        if group.open_batches == 0 {
            group.writes = 0;
            if let Some(session) = locked(&self.shared.session).as_mut() {
                session.flush()?;
            }
        }
        Ok(())
    }

    /// Makes everything written so far durable and visible to other
    /// processes.
    pub fn commit(&self) -> Result<(), StoreError> {
        if let Some(session) = locked(&self.shared.session).as_mut() {
            session.flush()?;
        }
        Ok(())
    }

    /// Reads other processes' commits from now on.
    pub fn refresh(&self) -> Result<(), StoreError> {
        self.with_session(Session::refresh)
    }

    /// Runs `f` against the session, for a read.
    pub(crate) fn read<R>(
        &self,
        f: impl FnOnce(&mut Session) -> splinter_expdb::Result<R>,
    ) -> Result<R, StoreError> {
        self.with_session(f)
    }

    /// Runs `f` against the session, for a write: committed before it
    /// returns, or, inside a batch, with the group it belongs to.
    pub(crate) fn write<R>(
        &self,
        f: impl FnOnce(&mut Session) -> splinter_expdb::Result<R>,
    ) -> Result<R, StoreError> {
        let mut group = locked(&self.shared.group);
        let batching = group.open_batches > 0;
        if !batching || group.writes == 0 {
            group.since = Instant::now();
        }
        let out = self.with_session(|session| {
            let out = f(session)?;
            let due = !batching
                || group.writes + 1 >= GROUP_COMMIT_WRITES
                || group.since.elapsed() >= GROUP_COMMIT_INTERVAL;
            if due {
                session.flush()?;
            }
            Ok((out, due))
        })?;
        group.writes = if out.1 { 0 } else { group.writes + 1 };
        Ok(out.0)
    }
}
