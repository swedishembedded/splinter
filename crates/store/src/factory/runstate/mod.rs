// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements restart-safe controllers for long-running
// automation, with fenced leases and an auditable event trail, for its
// clients. If your team needs expertise in durable workflow state, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The authoritative run state of the learning factory.
//!
//! One SQLite database in WAL mode, `<run>/state/controller.db`, holds what a
//! controller has decided: the lease that makes it the only controller
//! ([`RunState::acquire`]), the `stages` of the run, and the logical
//! `rollouts` slots with their physical attempts, and the `groups` of an
//! online run. Every mutation is one
//! `BEGIN IMMEDIATE` transaction that first checks the caller's [`Fence`], so
//! a controller that lost its lease cannot change anything even if it still
//! holds a connection. After the commit, one JSON line describing the change
//! is appended to `<run>/state/events.jsonl`; the log is a trail for people
//! and tools, never a source of truth, and a crash between commit and append
//! loses at most that line.
//!
//! Time is an argument: `now` is whole seconds since the Unix epoch
//! (`splinter_core::clock::unix_now()` in production), so a test fixes it.
//!
//! The lease is a fencing token. [`RunState::acquire`] grants it when no live
//! lease exists or the same owner renews; every grant after an expiry raises
//! the token, and a mutation presenting an older token is refused with
//! [`RunStateError::StaleFence`].

use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};

use rusqlite::{params, Connection, Transaction, TransactionBehavior};
use serde::Serialize;

pub mod groups;
mod model;
mod rollouts;
mod stages;

pub use groups::{GroupEntry, GroupRecord, GroupRolloutRecord, NewGroup};
pub use model::{
    AttemptId, AttemptRecord, AttemptState, SlotKey, SlotRecord, SlotState, StageOutcome,
    StageRecord, StageState,
};
use splinter_core::digest::Digest;

/// The database's path below the run directory.
pub const DB_FILE: &str = "state/controller.db";

/// The event log's path below the run directory.
pub const EVENTS_FILE: &str = "state/events.jsonl";

/// The schema version this code reads and writes (`PRAGMA user_version`).
const SCHEMA_VERSION: i64 = 1;

/// How long a statement waits for another connection's write lock.
const BUSY_TIMEOUT_MS: u64 = 5_000;

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS lease (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    owner TEXT NOT NULL,
    token INTEGER NOT NULL,
    expires_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS stages (
    stage_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('pending','running','completed','failed')),
    version INTEGER NOT NULL,
    product TEXT,
    error TEXT,
    updated_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS stage_slots (
    stage_id TEXT NOT NULL,
    pass TEXT NOT NULL,
    PRIMARY KEY (stage_id, pass)
);
CREATE TABLE IF NOT EXISTS slots (
    pass TEXT NOT NULL,
    task TEXT NOT NULL,
    idx INTEGER NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('pending','running','finished','abandoned')),
    trial TEXT,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (pass, task, idx)
);
CREATE TABLE IF NOT EXISTS attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pass TEXT NOT NULL,
    task TEXT NOT NULL,
    idx INTEGER NOT NULL,
    backend TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('launched','running','succeeded','failed','lost')),
    started_at INTEGER NOT NULL,
    finished_at INTEGER,
    result TEXT,
    FOREIGN KEY (pass, task, idx) REFERENCES slots (pass, task, idx)
);
CREATE INDEX IF NOT EXISTS attempts_unfinished ON attempts (state);
";

/// Why a run-state operation failed.
#[derive(Debug, thiserror::Error)]
pub enum RunStateError {
    /// The database refused a statement.
    #[error("run-state database: {0}")]
    Sqlite(#[from] rusqlite::Error),
    /// A file-system operation failed.
    #[error("{context}: {source}")]
    Io {
        /// What was being done, with the path.
        context: String,
        /// The underlying error.
        #[source]
        source: io::Error,
    },
    /// Another controller holds a live lease.
    #[error("the controller lease is held by {owner:?} until {expires_at}")]
    LeaseHeld {
        /// The current holder.
        owner: String,
        /// When its lease lapses, in Unix seconds.
        expires_at: u64,
    },
    /// The caller's fence is no longer the current one.
    #[error("fence {presented} of {owner:?} is stale; the current fence is {current}")]
    StaleFence {
        /// The token the caller presented.
        presented: u64,
        /// Who presented it.
        owner: String,
        /// The token in force, or 0 when no lease was ever granted.
        current: u64,
    },
    /// The stage changed since the caller read it.
    #[error("stage {stage} is at version {actual}, not the expected {expected}")]
    Conflict {
        /// The stage.
        stage: Digest,
        /// The version the caller expected.
        expected: u64,
        /// The version stored.
        actual: u64,
    },
    /// No stage with this id.
    #[error("no stage {0} in the run state")]
    UnknownStage(Digest),
    /// No slot with this key.
    #[error("no slot {0} in the run state")]
    UnknownSlot(SlotKey),
    /// No group (or no rollout of it) with this key.
    #[error("no group {0} (or no such rollout of it) in the run state")]
    UnknownGroup(splinter_core::factory::group::GroupKey),
    /// No attempt with this id.
    #[error("no attempt {0} in the run state")]
    UnknownAttempt(AttemptId),
    /// The requested state change is not allowed from the current state.
    #[error("{subject}: cannot go from {from} to {to}")]
    IllegalTransition {
        /// What was changed.
        subject: String,
        /// The current state.
        from: String,
        /// The requested state.
        to: String,
    },
    /// A product or error was supplied that does not fit the new state.
    #[error("invalid outcome for state {state}: {reason}")]
    InvalidOutcome {
        /// The new state.
        state: String,
        /// What is wrong.
        reason: String,
    },
    /// The slot already holds a different trial result.
    #[error("slot {slot} is already finished with {existing}; refusing {attempted}")]
    SlotFinished {
        /// The slot.
        slot: SlotKey,
        /// The committed trial.
        existing: Digest,
        /// The refused trial.
        attempted: Digest,
    },
    /// A backend handle that is not JSON.
    #[error("the backend handle is not valid JSON: {0}")]
    InvalidBackend(serde_json::Error),
    /// The database holds a value this code cannot read.
    #[error("the run-state database is inconsistent: {0}")]
    Corrupt(String),
}

/// Proof of holding the controller lease, required by every mutation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fence {
    /// Who holds the lease.
    pub owner: String,
    /// The fencing token; it only ever rises.
    pub token: u64,
}

/// A run directory's run state; shared by reference between the
/// controller's threads.
pub struct RunState {
    /// One connection, shared by the controller's threads (concurrent
    /// stages): SQLite serialises writers anyway, and a statement holds it
    /// only while it runs.
    conn: Mutex<Connection>,
    events: PathBuf,
}

/// One line of `events.jsonl`.
#[derive(Serialize)]
struct Event<'a> {
    at: u64,
    kind: &'a str,
    subject: String,
    detail: serde_json::Value,
}

impl RunState {
    /// Opens (creating when needed) the run state under the run directory
    /// `root`, in WAL mode with full synchronisation.
    pub fn open(root: &Path) -> Result<Self, RunStateError> {
        let dir = root.join("state");
        fs::create_dir_all(&dir).map_err(|source| RunStateError::Io {
            context: format!("create {}", dir.display()),
            source,
        })?;
        let conn = Connection::open(root.join(DB_FILE))?;
        conn.busy_timeout(std::time::Duration::from_millis(BUSY_TIMEOUT_MS))?;
        // Returns the mode, so it is a query.
        conn.query_row("PRAGMA journal_mode = WAL", [], |_| Ok(()))?;
        conn.pragma_update(None, "synchronous", "FULL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        let version: i64 = conn.query_row("PRAGMA user_version", [], |row| row.get(0))?;
        if version > SCHEMA_VERSION {
            return Err(RunStateError::Corrupt(format!(
                "schema version {version} is newer than the {SCHEMA_VERSION} this code reads"
            )));
        }
        conn.execute_batch(SCHEMA)?;
        conn.execute_batch(groups::SCHEMA)?;
        groups::migrate(&conn)?;
        conn.pragma_update(None, "user_version", SCHEMA_VERSION)?;
        Ok(Self {
            conn: Mutex::new(conn),
            events: root.join(EVENTS_FILE),
        })
    }

    /// Takes or renews the controller lease for `ttl_secs` seconds from `now`.
    ///
    /// Succeeds when no lease exists, the lease has expired (the token is
    /// raised), or `owner` already holds it (the token is kept and the expiry
    /// extended). Otherwise [`RunStateError::LeaseHeld`].
    pub fn acquire(&self, owner: &str, ttl_secs: u64, now: u64) -> Result<Fence, RunStateError> {
        let conn = self.conn();
        let tx = begin(&conn)?;
        let current: Option<(String, u64, u64)> = optional(tx.query_row(
            "SELECT owner, token, expires_at FROM lease WHERE id = 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        ))?;
        let expires_at = now.saturating_add(ttl_secs);
        let (token, kind) = match current {
            Some((holder, token, until)) if until > now && holder == owner => {
                (token, "lease.renewed")
            }
            Some((holder, _, until)) if until > now => {
                return Err(RunStateError::LeaseHeld {
                    owner: holder,
                    expires_at: until,
                });
            }
            Some((_, token, _)) => (token + 1, "lease.acquired"),
            None => (1, "lease.acquired"),
        };
        tx.execute(
            "INSERT INTO lease (id, owner, token, expires_at) VALUES (1, ?1, ?2, ?3)
             ON CONFLICT(id) DO UPDATE SET owner = ?1, token = ?2, expires_at = ?3",
            params![owner, token, expires_at],
        )?;
        tx.commit()?;
        self.emit(
            now,
            kind,
            owner.to_string(),
            serde_json::json!({ "token": token, "expires_at": expires_at }),
        )?;
        Ok(Fence {
            owner: owner.to_string(),
            token,
        })
    }

    /// Extends `fence`'s lease to `ttl_secs` seconds from `now`, keeping its
    /// token: the heartbeat of a controller whose stage outlasts one lease.
    /// A lease that lapsed while no one took it is still the holder's; one
    /// another controller took is not given back.
    ///
    /// # Errors
    /// The fence is stale (someone else holds the lease).
    pub fn renew(&self, fence: &Fence, ttl_secs: u64, now: u64) -> Result<(), RunStateError> {
        let conn = self.conn();
        let tx = begin_fenced(&conn, fence)?;
        tx.execute(
            "UPDATE lease SET expires_at = ?1 WHERE id = 1",
            params![now.saturating_add(ttl_secs)],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Gives up `fence`'s lease at `now`: any controller may acquire it
    /// after, with a higher token.
    ///
    /// # Errors
    /// The fence is stale (someone else holds the lease).
    pub fn release(&self, fence: &Fence, now: u64) -> Result<(), RunStateError> {
        let conn = self.conn();
        let tx = begin_fenced(&conn, fence)?;
        tx.execute(
            "UPDATE lease SET expires_at = ?1 WHERE id = 1",
            params![now],
        )?;
        tx.commit()?;
        self.emit(
            now,
            "lease.released",
            fence.owner.clone(),
            serde_json::json!({ "token": fence.token }),
        )?;
        Ok(())
    }

    /// Takes the lease from `holder`, which the caller has established no
    /// longer runs (a process of this host that has exited), as if it had
    /// expired: the token rises. Refused when the lease is no longer
    /// `holder`'s.
    ///
    /// # Errors
    /// Another controller holds the lease.
    pub fn take_over(
        &self,
        owner: &str,
        holder: &str,
        ttl_secs: u64,
        now: u64,
    ) -> Result<Fence, RunStateError> {
        let conn = self.conn();
        let tx = begin(&conn)?;
        let current: Option<(String, u64, u64)> = optional(tx.query_row(
            "SELECT owner, token, expires_at FROM lease WHERE id = 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        ))?;
        let token = match current {
            Some((current, token, _)) if current == holder => token + 1,
            Some((current, _, until)) if until > now => {
                return Err(RunStateError::LeaseHeld {
                    owner: current,
                    expires_at: until,
                });
            }
            Some((_, token, _)) => token + 1,
            None => 1,
        };
        let expires_at = now.saturating_add(ttl_secs);
        tx.execute(
            "INSERT INTO lease (id, owner, token, expires_at) VALUES (1, ?1, ?2, ?3)
             ON CONFLICT(id) DO UPDATE SET owner = ?1, token = ?2, expires_at = ?3",
            params![owner, token, expires_at],
        )?;
        tx.commit()?;
        self.emit(
            now,
            "lease.taken_over",
            owner.to_string(),
            serde_json::json!({ "token": token, "from": holder }),
        )?;
        Ok(Fence {
            owner: owner.to_string(),
            token,
        })
    }

    /// The connection, held by this thread until the guard drops: one
    /// statement or transaction at a time, whichever thread asks.
    fn conn(&self) -> MutexGuard<'_, Connection> {
        // A panic while the guard was held leaves no half-applied change: an
        // open transaction rolls back when it drops.
        self.conn.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Appends one event line after a commit. A failure is reported: the
    /// change is durable in the database, only its trail line is missing.
    fn emit(
        &self,
        at: u64,
        kind: &str,
        subject: String,
        detail: serde_json::Value,
    ) -> Result<(), RunStateError> {
        let mut line = serde_json::to_vec(&Event {
            at,
            kind,
            subject,
            detail,
        })
        .map_err(|e| RunStateError::Corrupt(format!("event line: {e}")))?;
        line.push(b'\n');
        let io_error = |source| RunStateError::Io {
            context: format!("append to {}", self.events.display()),
            source,
        };
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.events)
            .map_err(io_error)?;
        // One write call keeps a line whole against a concurrent appender.
        file.write_all(&line).map_err(io_error)?;
        file.flush().map_err(io_error)
    }
}

/// A write transaction that holds the database's write lock from the start.
fn begin(conn: &Connection) -> Result<Transaction<'_>, RunStateError> {
    Ok(Transaction::new_unchecked(
        conn,
        TransactionBehavior::Immediate,
    )?)
}

/// A write transaction in which `fence` has been checked against the
/// lease, so a stale controller never reaches its statements.
fn begin_fenced<'c>(conn: &'c Connection, fence: &Fence) -> Result<Transaction<'c>, RunStateError> {
    let tx = begin(conn)?;
    let current: Option<(String, u64)> = optional(tx.query_row(
        "SELECT owner, token FROM lease WHERE id = 1",
        [],
        |row| Ok((row.get(0)?, row.get(1)?)),
    ))?;
    match current {
        Some((owner, token)) if token == fence.token && owner == fence.owner => Ok(tx),
        other => Err(RunStateError::StaleFence {
            presented: fence.token,
            owner: fence.owner.clone(),
            current: other.map_or(0, |(_, token)| token),
        }),
    }
}

/// Turns "no row" into `None`.
fn optional<T>(result: rusqlite::Result<T>) -> rusqlite::Result<Option<T>> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(error) => Err(error),
    }
}

fn parse_digest(text: String) -> Result<Digest, RunStateError> {
    Digest::parse(&text).map_err(|e| RunStateError::Corrupt(e.to_string()))
}
