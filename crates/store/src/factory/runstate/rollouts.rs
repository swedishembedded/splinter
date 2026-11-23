// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements restart-safe controllers for long-running
// automation, with fenced leases and an auditable event trail, for its
// clients. If your team needs expertise in durable workflow state, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Logical rollout slots and the physical attempts made at them.
//!
//! A slot is the unit the plan asks for (pass, task, repeat index) and ends
//! with exactly one trial result. An attempt is one try at a slot on some
//! backend; a slot may need several, and a restarted controller reconciles
//! the attempts it finds unfinished ([`RunState::unfinished_attempts`])
//! against what the backends still run before it claims more work.

use rusqlite::{params, Connection, Row};
use splinter_core::digest::Digest;

use super::{
    optional, parse_digest, AttemptId, AttemptRecord, AttemptState, Fence, RunState, RunStateError,
    SlotKey, SlotRecord, SlotState,
};

fn slot_row(row: &Row<'_>) -> rusqlite::Result<RawSlot> {
    Ok((
        row.get(0)?,
        row.get(1)?,
        row.get(2)?,
        row.get(3)?,
        row.get(4)?,
        row.get(5)?,
    ))
}

type RawSlot = (String, String, u32, String, Option<String>, u64);

fn decode_slot(raw: RawSlot) -> Result<SlotRecord, RunStateError> {
    let (pass, task, index, state, trial, updated_at) = raw;
    Ok(SlotRecord {
        key: SlotKey { pass, task, index },
        state: SlotState::parse(&state)?,
        trial: trial.map(parse_digest).transpose()?,
        updated_at,
    })
}

const SLOT_COLUMNS: &str = "pass, task, idx, state, trial, updated_at";

fn fetch_slot(conn: &Connection, key: &SlotKey) -> Result<Option<SlotRecord>, RunStateError> {
    optional(conn.query_row(
        &format!("SELECT {SLOT_COLUMNS} FROM slots WHERE pass = ?1 AND task = ?2 AND idx = ?3"),
        params![key.pass, key.task, key.index],
        slot_row,
    ))?
    .map(decode_slot)
    .transpose()
}

type RawAttempt = (
    u64,
    String,
    String,
    u32,
    String,
    String,
    u64,
    Option<u64>,
    Option<String>,
);

const ATTEMPT_COLUMNS: &str =
    "id, pass, task, idx, backend, state, started_at, finished_at, result";

fn attempt_row(row: &Row<'_>) -> rusqlite::Result<RawAttempt> {
    Ok((
        row.get(0)?,
        row.get(1)?,
        row.get(2)?,
        row.get(3)?,
        row.get(4)?,
        row.get(5)?,
        row.get(6)?,
        row.get(7)?,
        row.get(8)?,
    ))
}

fn decode_attempt(raw: RawAttempt) -> Result<AttemptRecord, RunStateError> {
    let (id, pass, task, index, backend, state, started_at, finished_at, result) = raw;
    Ok(AttemptRecord {
        id: AttemptId(id),
        slot: SlotKey { pass, task, index },
        backend,
        state: AttemptState::parse(&state)?,
        started_at,
        finished_at,
        result: result.map(parse_digest).transpose()?,
    })
}

fn fetch_attempt(conn: &Connection, id: AttemptId) -> Result<Option<AttemptRecord>, RunStateError> {
    optional(conn.query_row(
        &format!("SELECT {ATTEMPT_COLUMNS} FROM attempts WHERE id = ?1"),
        [id.0],
        attempt_row,
    ))?
    .map(decode_attempt)
    .transpose()
}

fn illegal(subject: String, from: impl ToString, to: impl ToString) -> RunStateError {
    RunStateError::IllegalTransition {
        subject,
        from: from.to_string(),
        to: to.to_string(),
    }
}

impl RunState {
    /// Makes sure `pass` has `slots_per_task` slots for each of `tasks`,
    /// creating the missing ones as pending, and returns how many it created.
    /// Existing slots are never touched, so replaying the plan after a
    /// restart is safe.
    pub fn ensure_slots(
        &self,
        fence: &Fence,
        pass: &str,
        tasks: &[String],
        slots_per_task: u32,
        now: u64,
    ) -> Result<usize, RunStateError> {
        let tx = self.begin_fenced(fence)?;
        let mut created = 0;
        for task in tasks {
            for index in 0..slots_per_task {
                created += tx.execute(
                    "INSERT OR IGNORE INTO slots (pass, task, idx, state, updated_at)
                     VALUES (?1, ?2, ?3, 'pending', ?4)",
                    params![pass, task, index, now],
                )?;
            }
        }
        tx.commit()?;
        if created > 0 {
            self.emit(
                now,
                "slots.ensured",
                pass.to_string(),
                serde_json::json!({ "created": created, "tasks": tasks.len(), "per_task": slots_per_task }),
            )?;
        }
        Ok(created)
    }

    /// Claims the first pending slot of `pass` (by task, then index): marks
    /// it running and opens a `Launched` attempt carrying `backend`, the
    /// backend's handle as JSON text. `None` when no slot is pending.
    pub fn claim_next(
        &self,
        fence: &Fence,
        pass: &str,
        backend: &str,
        now: u64,
    ) -> Result<Option<AttemptRecord>, RunStateError> {
        serde_json::from_str::<serde_json::Value>(backend)
            .map_err(RunStateError::InvalidBackend)?;
        let tx = self.begin_fenced(fence)?;
        let next = optional(tx.query_row(
            &format!(
                "SELECT {SLOT_COLUMNS} FROM slots WHERE pass = ?1 AND state = 'pending'
                 ORDER BY task, idx LIMIT 1"
            ),
            [pass],
            slot_row,
        ))?
        .map(decode_slot)
        .transpose()?;
        let Some(slot) = next else {
            return Ok(None);
        };
        let key = &slot.key;
        tx.execute(
            "UPDATE slots SET state = 'running', updated_at = ?4
             WHERE pass = ?1 AND task = ?2 AND idx = ?3",
            params![key.pass, key.task, key.index, now],
        )?;
        tx.execute(
            "INSERT INTO attempts (pass, task, idx, backend, state, started_at)
             VALUES (?1, ?2, ?3, ?4, 'launched', ?5)",
            params![key.pass, key.task, key.index, backend, now],
        )?;
        let id = AttemptId(
            u64::try_from(tx.last_insert_rowid())
                .map_err(|_| RunStateError::Corrupt("negative attempt id".to_string()))?,
        );
        let attempt = fetch_attempt(&tx, id)?.ok_or(RunStateError::UnknownAttempt(id))?;
        tx.commit()?;
        self.emit(
            now,
            "attempt.claimed",
            key.to_string(),
            serde_json::json!({ "attempt": id.0, "backend": serde_json::from_str::<serde_json::Value>(backend).ok() }),
        )?;
        Ok(Some(attempt))
    }

    /// Records how an attempt stands. States only move forward (launched,
    /// running, then one terminal state); `result` is the trial digest and is
    /// required exactly for `Succeeded`. A failed or lost attempt returns its
    /// slot to pending so another attempt can be claimed.
    pub fn record_attempt(
        &self,
        fence: &Fence,
        attempt: AttemptId,
        state: AttemptState,
        result: Option<Digest>,
        now: u64,
    ) -> Result<AttemptRecord, RunStateError> {
        let succeeded = state == AttemptState::Succeeded;
        if succeeded != result.is_some() {
            return Err(RunStateError::InvalidOutcome {
                state: state.to_string(),
                reason: "a result digest goes with exactly the succeeded state".to_string(),
            });
        }
        let tx = self.begin_fenced(fence)?;
        let current = fetch_attempt(&tx, attempt)?.ok_or(RunStateError::UnknownAttempt(attempt))?;
        let forward = match current.state {
            AttemptState::Launched => state != AttemptState::Launched,
            AttemptState::Running => state.is_terminal(),
            _ => false,
        };
        if !forward {
            return Err(illegal(format!("attempt {attempt}"), current.state, state));
        }
        let finished_at = state.is_terminal().then_some(now);
        tx.execute(
            "UPDATE attempts SET state = ?2, finished_at = ?3, result = ?4 WHERE id = ?1",
            params![
                attempt.0,
                state.as_str(),
                finished_at,
                result.as_ref().map(Digest::as_str)
            ],
        )?;
        if matches!(state, AttemptState::Failed | AttemptState::Lost) {
            let key = &current.slot;
            tx.execute(
                "UPDATE slots SET state = 'pending', updated_at = ?4
                 WHERE pass = ?1 AND task = ?2 AND idx = ?3 AND state = 'running'",
                params![key.pass, key.task, key.index, now],
            )?;
        }
        let stored = fetch_attempt(&tx, attempt)?.ok_or(RunStateError::UnknownAttempt(attempt))?;
        tx.commit()?;
        self.emit(
            now,
            "attempt.transition",
            current.slot.to_string(),
            serde_json::json!({
                "attempt": attempt.0,
                "from": current.state.as_str(),
                "to": state.as_str(),
                "result": result.map(|d| d.to_string()),
            }),
        )?;
        Ok(stored)
    }

    /// Commits `trial` as the slot's result. Finishing a finished slot with
    /// the same digest is a no-op; with a different digest it is refused, so
    /// a slot's trial never changes once committed.
    pub fn finish_slot(
        &self,
        fence: &Fence,
        key: &SlotKey,
        trial: &Digest,
        now: u64,
    ) -> Result<SlotRecord, RunStateError> {
        let tx = self.begin_fenced(fence)?;
        let current =
            fetch_slot(&tx, key)?.ok_or_else(|| RunStateError::UnknownSlot(key.clone()))?;
        match (current.state, &current.trial) {
            (SlotState::Finished, Some(existing)) if existing == trial => return Ok(current),
            (SlotState::Finished, Some(existing)) => {
                return Err(RunStateError::SlotFinished {
                    slot: key.clone(),
                    existing: existing.clone(),
                    attempted: trial.clone(),
                });
            }
            (SlotState::Abandoned, _) => {
                return Err(illegal(
                    format!("slot {key}"),
                    current.state,
                    SlotState::Finished,
                ));
            }
            _ => {}
        }
        tx.execute(
            "UPDATE slots SET state = 'finished', trial = ?4, updated_at = ?5
             WHERE pass = ?1 AND task = ?2 AND idx = ?3",
            params![key.pass, key.task, key.index, trial.as_str(), now],
        )?;
        let stored =
            fetch_slot(&tx, key)?.ok_or_else(|| RunStateError::UnknownSlot(key.clone()))?;
        tx.commit()?;
        self.emit(
            now,
            "slot.finished",
            key.to_string(),
            serde_json::json!({ "trial": trial.as_str() }),
        )?;
        Ok(stored)
    }

    /// Gives up on a slot that is not finished. Abandoning an abandoned slot
    /// is a no-op.
    pub fn abandon_slot(
        &self,
        fence: &Fence,
        key: &SlotKey,
        now: u64,
    ) -> Result<SlotRecord, RunStateError> {
        let tx = self.begin_fenced(fence)?;
        let current =
            fetch_slot(&tx, key)?.ok_or_else(|| RunStateError::UnknownSlot(key.clone()))?;
        match current.state {
            SlotState::Abandoned => return Ok(current),
            SlotState::Finished => {
                return Err(illegal(
                    format!("slot {key}"),
                    current.state,
                    SlotState::Abandoned,
                ));
            }
            SlotState::Pending | SlotState::Running => {}
        }
        tx.execute(
            "UPDATE slots SET state = 'abandoned', updated_at = ?4
             WHERE pass = ?1 AND task = ?2 AND idx = ?3",
            params![key.pass, key.task, key.index, now],
        )?;
        let stored =
            fetch_slot(&tx, key)?.ok_or_else(|| RunStateError::UnknownSlot(key.clone()))?;
        tx.commit()?;
        self.emit(
            now,
            "slot.abandoned",
            key.to_string(),
            serde_json::json!({ "from": current.state.as_str() }),
        )?;
        Ok(stored)
    }

    /// The slot `key`, if known.
    pub fn slot(&self, key: &SlotKey) -> Result<Option<SlotRecord>, RunStateError> {
        fetch_slot(&self.conn, key)
    }

    /// Every slot of `pass`, by task then index.
    pub fn list_slots(&self, pass: &str) -> Result<Vec<SlotRecord>, RunStateError> {
        let mut statement = self.conn.prepare(&format!(
            "SELECT {SLOT_COLUMNS} FROM slots WHERE pass = ?1 ORDER BY task, idx"
        ))?;
        let rows = statement.query_map([pass], slot_row)?;
        rows.map(|row| decode_slot(row?)).collect()
    }

    /// The attempts that were launched or running and have no outcome, oldest
    /// first: what a restarted controller reconciles with its backends.
    pub fn unfinished_attempts(&self) -> Result<Vec<AttemptRecord>, RunStateError> {
        let mut statement = self.conn.prepare(&format!(
            "SELECT {ATTEMPT_COLUMNS} FROM attempts
             WHERE state IN ('launched', 'running') ORDER BY id"
        ))?;
        let rows = statement.query_map([], attempt_row)?;
        rows.map(|row| decode_attempt(row?)).collect()
    }
}
