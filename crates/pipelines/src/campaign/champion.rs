// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that run model learning as one
// resumable, multi-generation campaign with a single promotion gate, for its
// clients. If your team needs expertise in self-improving training loops you
// can procure our services by sending an email to info@swedishembedded.com.

//! The champion pointer and its history: never a value overwritten, only
//! records appended to the campaign's ledger. Each generation ends in one
//! record of what became of the champion (`gNNNN/promote`: promoted or
//! kept, with the gate's reasons), and each rollback is one more record
//! (`rollback/N`) naming the champion it moves back to and why. The champion
//! is the base until the first promotion, and then whatever the newest of
//! these records moved it to.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use splinter_core::digest::digest_of;
use splinter_store::stage_ledger::{Commit, KeyParts, OnCorrupt, StageLedger, StageWork};

use super::controller::{record, unit};
use super::work::Model;
use crate::served::ServedError;

/// The unit-name prefix of a rollback record.
const ROLLBACK: &str = "rollback/";

/// What a record did to the champion.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MoveKind {
    /// The generation's candidate passed the gate and became the champion.
    Promote,
    /// The generation's candidate was rejected, or there was none: the
    /// champion stays.
    Keep,
    /// The champion was moved back to an earlier one by hand.
    Rollback,
}

/// One entry of the champion's history.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Move {
    /// What it did.
    pub kind: MoveKind,
    /// The generation that ended in it; `None` for a rollback.
    pub generation: Option<u32>,
    /// The champion before.
    pub from: Model,
    /// The champion after.
    pub to: Model,
    /// Why: the gate's reasons, or the reason a rollback was given.
    pub reason: String,
}

fn moves(commits: &[Commit]) -> Result<Vec<Move>, ServedError> {
    commits
        .iter()
        .filter(|c| c.unit.starts_with(ROLLBACK) || c.unit.ends_with("/promote"))
        .map(record::<Move>)
        .collect()
}

/// The champion and its history, oldest move first, from the ledger's
/// `commits`; `base` is the champion before any move.
///
/// # Errors
/// A record of another build.
pub fn history(base: &Model, commits: &[Commit]) -> Result<(Model, Vec<Move>), ServedError> {
    let moves = moves(commits)?;
    let champion = moves
        .iter()
        .rev()
        .find(|m| m.kind != MoveKind::Keep)
        .map_or_else(|| base.clone(), |m| m.to.clone());
    Ok((champion, moves))
}

fn append(
    ledger: &StageLedger,
    name: &str,
    key: &KeyParts,
    moved: &Move,
) -> Result<Move, ServedError> {
    let summary = serde_json::to_value(moved)
        .map_err(|e| ServedError::Config(format!("a champion move: {e}")))?;
    let ran = ledger.run(name, &key.key(), OnCorrupt::Refuse, |_| {
        Ok::<_, String>(StageWork::new(summary))
    })?;
    record(&ran.commit)
}

fn ended(
    ledger: &StageLedger,
    generation: u32,
    moved: &Move,
    gate: &Value,
) -> Result<Move, ServedError> {
    let key = KeyParts::new("promote")
        .config(&json!({"kind": moved.kind}))
        .input("gate", &digest_of(gate))
        .input("from", &digest_of(&moved.from));
    append(ledger, &unit(generation, "promote"), &key, moved)
}

/// Records that `generation`'s `candidate` became the champion in place of
/// `from` on the gate's decision `gate`.
///
/// # Errors
/// The ledger.
pub fn promote(
    ledger: &StageLedger,
    generation: u32,
    from: &Model,
    candidate: &Model,
    gate: &Value,
) -> Result<Move, ServedError> {
    let moved = Move {
        kind: MoveKind::Promote,
        generation: Some(generation),
        from: from.clone(),
        to: candidate.clone(),
        reason: "the gate's rules hold: dev gain shown, retention non-inferior, fresh sealed slice verified".into(),
    };
    ended(ledger, generation, &moved, gate)
}

/// Records that `generation` ended with `champion` staying, for `reason`.
///
/// # Errors
/// The ledger.
pub fn keep(
    ledger: &StageLedger,
    generation: u32,
    champion: &Model,
    reason: &str,
    gate: &Value,
) -> Result<Move, ServedError> {
    let moved = Move {
        kind: MoveKind::Keep,
        generation: Some(generation),
        from: champion.clone(),
        to: champion.clone(),
        reason: reason.to_string(),
    };
    ended(ledger, generation, &moved, gate)
}

/// Moves the champion back to the model generation `to` made (0: `base`),
/// for `reason`. Asked again after it was recorded, the same rollback finds
/// its record and records nothing more.
///
/// # Errors
/// An empty reason, a generation whose model was never a champion, a
/// target that is the champion now (by another move), or the ledger.
pub fn rollback(
    ledger: &StageLedger,
    base: &Model,
    to: u32,
    reason: &str,
) -> Result<Move, ServedError> {
    let reason = reason.trim();
    if reason.is_empty() {
        return Err(ServedError::Config(
            "a rollback is recorded with its reason: give one".into(),
        ));
    }
    let commits = ledger.commits()?;
    let (champion, history) = history(base, &commits)?;
    if let Some(last) = history.last() {
        if last.kind == MoveKind::Rollback && last.to.generation == to && last.reason == reason {
            return Ok(last.clone());
        }
    }
    let target = if to == 0 {
        base.clone()
    } else {
        history
            .iter()
            .find(|m| m.kind == MoveKind::Promote && m.to.generation == to)
            .map(|m| m.to.clone())
            .ok_or_else(|| {
                ServedError::Config(format!(
                    "generation {to}'s candidate was never the champion: a rollback moves only to an earlier champion"
                ))
            })?
    };
    if target == champion {
        return Err(ServedError::Config(format!(
            "generation {to}'s model is the champion now: there is nothing to roll back"
        )));
    }
    let number = commits
        .iter()
        .filter(|c| c.unit.starts_with(ROLLBACK))
        .count()
        + 1;
    let moved = Move {
        kind: MoveKind::Rollback,
        generation: None,
        from: champion.clone(),
        to: target,
        reason: reason.to_string(),
    };
    let key = KeyParts::new("rollback")
        .config(&json!({"to": to, "reason": reason, "number": number}))
        .input("from", &digest_of(&champion));
    append(ledger, &format!("{ROLLBACK}{number}"), &key, &moved)
}
