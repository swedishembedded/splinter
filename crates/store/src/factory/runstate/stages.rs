// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements restart-safe controllers for long-running
// automation, with fenced leases and an auditable event trail, for its
// clients. If your team needs expertise in durable workflow state, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Stage records: what the controller has decided about each stage.

use rusqlite::{params, Connection, Row};
use splinter_core::digest::Digest;

use super::{
    optional, parse_digest, Fence, RunState, RunStateError, StageOutcome, StageRecord, StageState,
};

type RawStage = (
    String,
    String,
    String,
    String,
    u64,
    Option<String>,
    Option<String>,
    u64,
);

const COLUMNS: &str = "stage_id, name, kind, state, version, product, error, updated_at";

fn record(row: &Row<'_>) -> rusqlite::Result<RawStage> {
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

fn decode(raw: RawStage) -> Result<StageRecord, RunStateError> {
    let (stage_id, name, kind, state, version, product, error, updated_at) = raw;
    Ok(StageRecord {
        stage_id: parse_digest(stage_id)?,
        name,
        kind,
        state: StageState::parse(&state)?,
        version,
        product: product.map(parse_digest).transpose()?,
        error,
        updated_at,
    })
}

fn fetch(conn: &Connection, stage_id: &Digest) -> Result<Option<StageRecord>, RunStateError> {
    optional(conn.query_row(
        &format!("SELECT {COLUMNS} FROM stages WHERE stage_id = ?1"),
        [stage_id.as_str()],
        record,
    ))?
    .map(decode)
    .transpose()
}

/// Checks the outcome carries exactly what `state` needs.
fn check_outcome(state: StageState, outcome: &StageOutcome) -> Result<(), RunStateError> {
    let reason = match (state, outcome) {
        (StageState::Completed, StageOutcome::Product(_))
        | (StageState::Failed, StageOutcome::Error(_))
        | (StageState::Pending | StageState::Running, StageOutcome::None) => return Ok(()),
        (StageState::Completed, _) => "a completed stage needs the product it committed",
        (StageState::Failed, _) => "a failed stage needs an error text",
        (StageState::Pending | StageState::Running, _) => "this state carries no product or error",
    };
    Err(RunStateError::InvalidOutcome {
        state: state.to_string(),
        reason: reason.to_string(),
    })
}

impl RunState {
    /// Records `stage_id` as pending unless it is already known; either way
    /// returns the stored record. Idempotent, so a restarted controller can
    /// replay its plan.
    pub fn upsert_pending(
        &self,
        fence: &Fence,
        stage_id: &Digest,
        name: &str,
        kind: &str,
        now: u64,
    ) -> Result<StageRecord, RunStateError> {
        let tx = self.begin_fenced(fence)?;
        let inserted = tx.execute(
            "INSERT OR IGNORE INTO stages (stage_id, name, kind, state, version, updated_at)
             VALUES (?1, ?2, ?3, 'pending', 0, ?4)",
            params![stage_id.as_str(), name, kind, now],
        )?;
        let stored =
            fetch(&tx, stage_id)?.ok_or_else(|| RunStateError::UnknownStage(stage_id.clone()))?;
        tx.commit()?;
        if inserted > 0 {
            self.emit(
                now,
                "stage.pending",
                stage_id.to_string(),
                serde_json::json!({ "name": name, "kind": kind }),
            )?;
        }
        Ok(stored)
    }

    /// Moves a stage to `new_state`, provided it is still at
    /// `expected_version` ([`RunStateError::Conflict`] otherwise), and bumps
    /// its version. A completed stage never changes again; a failed one may
    /// be retried.
    pub fn transition(
        &self,
        fence: &Fence,
        stage_id: &Digest,
        expected_version: u64,
        new_state: StageState,
        outcome: StageOutcome,
        now: u64,
    ) -> Result<StageRecord, RunStateError> {
        check_outcome(new_state, &outcome)?;
        let tx = self.begin_fenced(fence)?;
        let current =
            fetch(&tx, stage_id)?.ok_or_else(|| RunStateError::UnknownStage(stage_id.clone()))?;
        if current.version != expected_version {
            return Err(RunStateError::Conflict {
                stage: stage_id.clone(),
                expected: expected_version,
                actual: current.version,
            });
        }
        if current.state == StageState::Completed {
            return Err(RunStateError::IllegalTransition {
                subject: format!("stage {stage_id}"),
                from: current.state.to_string(),
                to: new_state.to_string(),
            });
        }
        let (product, error) = match &outcome {
            StageOutcome::Product(digest) => (Some(digest.as_str()), None),
            StageOutcome::Error(text) => (None, Some(text.as_str())),
            StageOutcome::None => (None, None),
        };
        tx.execute(
            "UPDATE stages SET state = ?2, version = version + 1, product = ?3, error = ?4,
             updated_at = ?5 WHERE stage_id = ?1",
            params![stage_id.as_str(), new_state.as_str(), product, error, now],
        )?;
        let stored =
            fetch(&tx, stage_id)?.ok_or_else(|| RunStateError::UnknownStage(stage_id.clone()))?;
        tx.commit()?;
        self.emit(
            now,
            "stage.transition",
            stage_id.to_string(),
            serde_json::json!({
                "from": current.state.as_str(),
                "to": new_state.as_str(),
                "version": stored.version,
                "product": product,
                "error": error,
            }),
        )?;
        Ok(stored)
    }

    /// The stage `stage_id`, if known.
    pub fn get(&self, stage_id: &Digest) -> Result<Option<StageRecord>, RunStateError> {
        fetch(&self.conn, stage_id)
    }

    /// Every stage, ordered by name then id.
    pub fn list(&self) -> Result<Vec<StageRecord>, RunStateError> {
        let mut statement = self.conn.prepare(&format!(
            "SELECT {COLUMNS} FROM stages ORDER BY name, stage_id"
        ))?;
        let rows = statement.query_map([], record)?;
        rows.map(|row| decode(row?)).collect()
    }
}
