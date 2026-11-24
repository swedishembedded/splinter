// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements online reinforcement-learning controllers
// that never train a group twice or lose one they produced, for its clients.
// If your team needs expertise in asynchronous policy optimisation, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The group ledger of an online run: each group of rollouts of one task,
//! its rollouts' outcomes, and the records the learner is sent, in the
//! run's one database under the controller's fence.
//!
//! A rollout's records (JSON lines from its model requests) are kept in its
//! row until the group is terminal (`filtered`, `stale`, `committed`): until
//! then a resumed controller may need to send them again. The outcomes stay,
//! so the run's reward statistics can be rebuilt from the ledger.

use rusqlite::{params, Connection, Row};
use splinter_core::factory::group::{GroupKey, GroupState, RolloutOutcome, RolloutState};

use super::{optional, Fence, RunState, RunStateError};

/// The ledger's tables; created with the rest of the schema.
pub(super) const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS groups (
    key TEXT PRIMARY KEY,
    seq INTEGER NOT NULL UNIQUE,
    task TEXT NOT NULL,
    size INTEGER NOT NULL,
    state TEXT NOT NULL,
    reason TEXT,
    staleness INTEGER,
    updated_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS group_rollouts (
    key TEXT NOT NULL REFERENCES groups (key),
    idx INTEGER NOT NULL,
    state TEXT NOT NULL,
    failures INTEGER NOT NULL,
    outcome TEXT,
    records TEXT,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (key, idx)
);
";

/// A group as stored.
#[derive(Clone, Debug, PartialEq)]
pub struct GroupRecord {
    /// Its key.
    pub key: GroupKey,
    /// Its place in the run's order of groups, from 0.
    pub seq: u64,
    /// The task its rollouts attempt.
    pub task: String,
    /// How many rollouts it has.
    pub size: u32,
    /// Its state.
    pub state: GroupState,
    /// Why it was filtered or dropped, when it was.
    pub reason: Option<String>,
    /// How many policy versions behind the learner it was when last sent.
    pub staleness: Option<u64>,
    /// When it last changed, in Unix seconds.
    pub updated_at: u64,
}

/// One rollout of a group as stored.
#[derive(Clone, Debug, PartialEq)]
pub struct GroupRolloutRecord {
    /// Its index in the group, from 0.
    pub index: u32,
    /// Its state.
    pub state: RolloutState,
    /// How many times its infrastructure failed.
    pub failures: u32,
    /// How it ended, once finished.
    pub outcome: Option<RolloutOutcome>,
    /// When it last changed, in Unix seconds.
    pub updated_at: u64,
}

/// A group with its rollouts, by index.
#[derive(Clone, Debug, PartialEq)]
pub struct GroupEntry {
    /// The group.
    pub group: GroupRecord,
    /// Its rollouts.
    pub rollouts: Vec<GroupRolloutRecord>,
}

fn corrupt(error: impl std::fmt::Display) -> RunStateError {
    RunStateError::Corrupt(error.to_string())
}

type RawGroup = (
    String,
    u64,
    String,
    u32,
    String,
    Option<String>,
    Option<u64>,
    u64,
);

fn group_row(row: &Row<'_>) -> rusqlite::Result<RawGroup> {
    Ok((
        row.get(0)?,
        row.get(1)?,
        row.get(2)?,
        row.get(3)?,
        row.get(4)?,
        row.get(5)?,
        row.get(6)?,
        row.get(7)?,
    ))
}

fn decode_group(raw: RawGroup) -> Result<GroupRecord, RunStateError> {
    let (key, seq, task, size, state, reason, staleness, updated_at) = raw;
    Ok(GroupRecord {
        key: GroupKey::new(key).map_err(corrupt)?,
        seq,
        task,
        size,
        state: state.parse().map_err(corrupt)?,
        reason,
        staleness,
        updated_at,
    })
}

const GROUP_COLUMNS: &str = "key, seq, task, size, state, reason, staleness, updated_at";

fn fetch_group(conn: &Connection, key: &GroupKey) -> Result<GroupRecord, RunStateError> {
    optional(conn.query_row(
        &format!("SELECT {GROUP_COLUMNS} FROM groups WHERE key = ?1"),
        [key.as_str()],
        group_row,
    ))?
    .map(decode_group)
    .transpose()?
    .ok_or_else(|| RunStateError::UnknownGroup(key.clone()))
}

type RawRollout = (u32, String, u32, Option<String>, u64);

fn decode_rollout(raw: RawRollout) -> Result<GroupRolloutRecord, RunStateError> {
    let (index, state, failures, outcome, updated_at) = raw;
    Ok(GroupRolloutRecord {
        index,
        state: state.parse().map_err(corrupt)?,
        failures,
        outcome: outcome
            .map(|text| serde_json::from_str(&text))
            .transpose()
            .map_err(corrupt)?,
        updated_at,
    })
}

fn fetch_rollout(
    conn: &Connection,
    key: &GroupKey,
    index: u32,
) -> Result<GroupRolloutRecord, RunStateError> {
    optional(conn.query_row(
        "SELECT idx, state, failures, outcome, updated_at FROM group_rollouts
         WHERE key = ?1 AND idx = ?2",
        params![key.as_str(), index],
        |row| {
            Ok((
                row.get(0)?,
                row.get(1)?,
                row.get(2)?,
                row.get(3)?,
                row.get(4)?,
            ))
        },
    ))?
    .map(decode_rollout)
    .transpose()?
    .ok_or_else(|| RunStateError::UnknownGroup(key.clone()))
}

fn illegal(subject: String, from: impl ToString, to: impl ToString) -> RunStateError {
    RunStateError::IllegalTransition {
        subject,
        from: from.to_string(),
        to: to.to_string(),
    }
}

impl RunState {
    /// Records the group `key`, the `seq`-th of the run, as `size` pending
    /// rollouts of `task`. True when it was created; recording the same
    /// group again is a no-op (false), and a different group under the same
    /// key or seq is refused.
    ///
    /// # Errors
    /// The fence is stale, or the key or seq is another group's.
    pub fn create_group(
        &self,
        fence: &Fence,
        key: &GroupKey,
        seq: u64,
        task: &str,
        size: u32,
        now: u64,
    ) -> Result<bool, RunStateError> {
        let tx = self.begin_fenced(fence)?;
        let existing: Option<(String, u64, String, u32)> = optional(tx.query_row(
            "SELECT key, seq, task, size FROM groups WHERE key = ?1 OR seq = ?2",
            params![key.as_str(), seq],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        ))?;
        if let Some(found) = existing {
            if found == (key.as_str().to_string(), seq, task.to_string(), size) {
                return Ok(false);
            }
            return Err(RunStateError::InvalidOutcome {
                state: GroupState::Pending.to_string(),
                reason: format!(
                    "group {key} (seq {seq}, task {task}, size {size}) collides with group {} (seq {}, task {}, size {})",
                    found.0, found.1, found.2, found.3
                ),
            });
        }
        tx.execute(
            "INSERT INTO groups (key, seq, task, size, state, updated_at)
             VALUES (?1, ?2, ?3, ?4, 'pending', ?5)",
            params![key.as_str(), seq, task, size, now],
        )?;
        for index in 0..size {
            tx.execute(
                "INSERT INTO group_rollouts (key, idx, state, failures, updated_at)
                 VALUES (?1, ?2, 'pending', 0, ?3)",
                params![key.as_str(), index, now],
            )?;
        }
        tx.commit()?;
        self.emit(
            now,
            "group.created",
            key.to_string(),
            serde_json::json!({ "seq": seq, "task": task, "size": size }),
        )?;
        Ok(true)
    }

    /// Marks rollout `index` of `key` running, and the group running with
    /// it. A running rollout may be started again (a resumed controller
    /// relaunches it); a finished or abandoned one never is.
    ///
    /// # Errors
    /// The fence is stale, the rollout is unknown or terminal.
    pub fn start_rollout(
        &self,
        fence: &Fence,
        key: &GroupKey,
        index: u32,
        now: u64,
    ) -> Result<(), RunStateError> {
        let tx = self.begin_fenced(fence)?;
        let rollout = fetch_rollout(&tx, key, index)?;
        if rollout.state.is_terminal() {
            return Err(illegal(
                format!("rollout {key}/{index}"),
                rollout.state,
                RolloutState::Running,
            ));
        }
        tx.execute(
            "UPDATE group_rollouts SET state = 'running', updated_at = ?3 WHERE key = ?1 AND idx = ?2",
            params![key.as_str(), index, now],
        )?;
        tx.execute(
            "UPDATE groups SET state = 'running', updated_at = ?2 WHERE key = ?1 AND state = 'pending'",
            params![key.as_str(), now],
        )?;
        tx.commit()?;
        self.emit(
            now,
            "group.rollout.started",
            key.to_string(),
            serde_json::json!({ "rollout": index }),
        )
    }

    /// Records that running rollout `index` of `key` ended with `outcome`,
    /// keeping `records` (its JSON lines) until the group is terminal.
    ///
    /// # Errors
    /// The fence is stale, the rollout is not running, or the outcome is
    /// invalid.
    pub fn finish_rollout(
        &self,
        fence: &Fence,
        key: &GroupKey,
        index: u32,
        outcome: &RolloutOutcome,
        records: &str,
        now: u64,
    ) -> Result<(), RunStateError> {
        outcome
            .validate()
            .map_err(|e| RunStateError::InvalidOutcome {
                state: RolloutState::Finished.to_string(),
                reason: e.to_string(),
            })?;
        let text = serde_json::to_string(outcome).map_err(corrupt)?;
        let tx = self.begin_fenced(fence)?;
        let rollout = fetch_rollout(&tx, key, index)?;
        if rollout.state != RolloutState::Running {
            return Err(illegal(
                format!("rollout {key}/{index}"),
                rollout.state,
                RolloutState::Finished,
            ));
        }
        tx.execute(
            "UPDATE group_rollouts SET state = 'finished', outcome = ?3, records = ?4, updated_at = ?5
             WHERE key = ?1 AND idx = ?2",
            params![key.as_str(), index, text, records, now],
        )?;
        tx.commit()?;
        self.emit(
            now,
            "group.rollout.finished",
            key.to_string(),
            serde_json::json!({ "rollout": index, "reward": outcome.reward, "tokens": outcome.tokens }),
        )
    }

    /// Records that running rollout `index` of `key` failed in its
    /// infrastructure (`reason`): it goes back to pending, or is abandoned
    /// when `abandon` (its retry budget is spent).
    ///
    /// # Errors
    /// The fence is stale or the rollout is not running.
    pub fn fail_rollout(
        &self,
        fence: &Fence,
        key: &GroupKey,
        index: u32,
        abandon: bool,
        reason: &str,
        now: u64,
    ) -> Result<(), RunStateError> {
        let to = if abandon {
            RolloutState::Abandoned
        } else {
            RolloutState::Pending
        };
        let tx = self.begin_fenced(fence)?;
        let rollout = fetch_rollout(&tx, key, index)?;
        if rollout.state != RolloutState::Running {
            return Err(illegal(format!("rollout {key}/{index}"), rollout.state, to));
        }
        tx.execute(
            "UPDATE group_rollouts SET state = ?3, failures = failures + 1, updated_at = ?4
             WHERE key = ?1 AND idx = ?2",
            params![key.as_str(), index, to.as_str(), now],
        )?;
        tx.commit()?;
        self.emit(
            now,
            "group.rollout.failed",
            key.to_string(),
            serde_json::json!({ "rollout": index, "to": to.as_str(), "reason": reason }),
        )
    }

    /// Moves group `key` to `to` along its lifecycle, recording why
    /// (`reason`) and its staleness when sent. A terminal group's records
    /// are dropped.
    ///
    /// # Errors
    /// The fence is stale, the group is unknown, or the transition is not in
    /// the lifecycle.
    pub fn set_group_state(
        &self,
        fence: &Fence,
        key: &GroupKey,
        to: GroupState,
        reason: Option<&str>,
        staleness: Option<u64>,
        now: u64,
    ) -> Result<(), RunStateError> {
        let tx = self.begin_fenced(fence)?;
        let group = fetch_group(&tx, key)?;
        if !group.state.may_become(to) {
            return Err(illegal(format!("group {key}"), group.state, to));
        }
        tx.execute(
            "UPDATE groups SET state = ?2, reason = COALESCE(?3, reason),
             staleness = COALESCE(?4, staleness), updated_at = ?5 WHERE key = ?1",
            params![key.as_str(), to.as_str(), reason, staleness, now],
        )?;
        if to.is_terminal() {
            tx.execute(
                "UPDATE group_rollouts SET records = NULL WHERE key = ?1",
                [key.as_str()],
            )?;
        }
        tx.commit()?;
        self.emit(
            now,
            "group.transition",
            key.to_string(),
            serde_json::json!({ "from": group.state.as_str(), "to": to.as_str(), "reason": reason, "staleness": staleness }),
        )
    }

    /// Every group with its rollouts, by seq: what a resumed controller
    /// rebuilds itself from.
    ///
    /// # Errors
    /// The database cannot be read or holds an unreadable value.
    pub fn group_entries(&self) -> Result<Vec<GroupEntry>, RunStateError> {
        let mut groups = self
            .conn
            .prepare(&format!("SELECT {GROUP_COLUMNS} FROM groups ORDER BY seq"))?;
        let mut entries: Vec<GroupEntry> = groups
            .query_map([], group_row)?
            .map(|row| {
                Ok(GroupEntry {
                    group: decode_group(row?)?,
                    rollouts: Vec::new(),
                })
            })
            .collect::<Result<_, RunStateError>>()?;
        let position: std::collections::HashMap<String, usize> = entries
            .iter()
            .enumerate()
            .map(|(i, e)| (e.group.key.as_str().to_string(), i))
            .collect();
        let mut rollouts = self.conn.prepare(
            "SELECT key, idx, state, failures, outcome, updated_at FROM group_rollouts ORDER BY key, idx",
        )?;
        let rows = rollouts.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                (
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                ),
            ))
        })?;
        for row in rows {
            let (key, raw) = row?;
            let at = *position
                .get(&key)
                .ok_or_else(|| corrupt(format!("rollout of unknown group {key}")))?;
            entries[at].rollouts.push(decode_rollout(raw)?);
        }
        Ok(entries)
    }

    /// The records of group `key`'s finished rollouts, by index; empty once
    /// the group is terminal.
    ///
    /// # Errors
    /// The database cannot be read.
    pub fn group_records(&self, key: &GroupKey) -> Result<Vec<(u32, String)>, RunStateError> {
        let mut statement = self.conn.prepare(
            "SELECT idx, records FROM group_rollouts
             WHERE key = ?1 AND records IS NOT NULL ORDER BY idx",
        )?;
        let rows = statement.query_map([key.as_str()], |row| Ok((row.get(0)?, row.get(1)?)))?;
        rows.map(|row| row.map_err(RunStateError::from)).collect()
    }
}
