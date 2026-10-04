// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements retrieval-augmented fine-tuning of models
// on a person's writing, for its clients. If your team needs expertise in
// training a model to use retrieved passages and ignore the wrong ones, you
// can procure our services by sending an email to info@swedishembedded.com.

//! Retrieval-augmented fine-tuning: training records with passages in the
//! prompt.
//!
//! A model trained only closed-book has never seen a prompt with passages in
//! front of the question. Asked with retrieved ones it cannot tell the
//! passage that holds the answer from one that only resembles the question:
//! measured, the passages it was given helped where retrieval found the
//! evidence and hurt exactly as much where it did not. So a share of the
//! training records is given passages as `ask --retrieve` gives them, and
//! of those some carry the passage the task was written from among retrieved
//! distractors, and the rest the distractors alone; the answer never changes.
//! The model learns to use the context where it holds the answer and to answer
//! from itself where it does not.
//!
//! Which records, and which of them hold the evidence, is a stable function of
//! the experience, so the same dataset is built every time.

use serde::Serialize;
use splinter_core::digest::Digest;
use splinter_data::{Projection, RecordBody};
use splinter_knowledge::retrieve::Passage;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

use crate::retrieval::{overlaps_evidence, Retrieval, Retrieved};

/// More passages than are shown are retrieved, so that dropping those that
/// hold the evidence still leaves enough distractors.
const SPARE: usize = 3;

/// Of the records given passages, the share whose passages include the one
/// the task was written from: most, so context is learned to be worth using,
/// and enough without it that the model does not lean on it.
pub const DEFAULT_EVIDENCE_SHARE: f64 = 0.8;

/// How much of a dataset is given passages, and how many of those hold the
/// evidence.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct PassageShare {
    /// The share of records given passages at all, in `[0, 1]`.
    pub records: f64,
    /// Of those, the share whose passages include the one the task was written
    /// from, in `[0, 1]`; the rest carry distractors only.
    pub with_evidence: f64,
}

/// Whether `subject` falls within the first `share` of a stable draw under
/// `salt`.
fn chosen(subject: &Digest, salt: &str, share: f64) -> bool {
    let draw = Digest::of(format!("splinter-raft/{salt}/{subject}").as_bytes());
    let top = u64::from_str_radix(&draw.hex()[..16], 16).unwrap_or(0);
    (top as f64) / (u64::MAX as f64) < share
}

/// `projection` with passages in the first user turn of the records `share`
/// picks; see the module documentation. A record with no experience to find a
/// task by is left as it was.
pub fn with_passages(
    ctx: &Context,
    mut projection: Projection,
    share: &PassageShare,
    retrieval: &Retrieval<'_>,
) -> Result<Projection, OrchestratorError> {
    let experiences = ctx.experiences();
    for record in &mut projection.records {
        let Some(id) = record.metadata.experiences.first() else {
            continue;
        };
        if !chosen(&id.0, "records", share.records) {
            continue;
        }
        let task = experiences.get(id)?.to_task();
        let shown = retrieval.passages.max(1);
        let found = retrieval
            .library
            .find(&task.instruction, retrieval.embedder, shown + SPARE)
            .map_err(|e| OrchestratorError::Refused(format!("retrieval: {e}")))?;
        let oracle: Option<&Passage> = chosen(&id.0, "evidence", share.with_evidence)
            .then(|| {
                retrieval
                    .library
                    .passages()
                    .iter()
                    .find(|p| overlaps_evidence(p, &task))
            })
            .flatten();
        let mut context: Vec<&Passage> = found
            .into_iter()
            .filter(|p| !overlaps_evidence(p, &task))
            .take(shown - usize::from(oracle.is_some()))
            .collect();
        if let Some(oracle) = oracle {
            // Where among the others it sits is not a tell.
            let draw = Digest::of(format!("splinter-raft/place/{}", id.0).as_bytes());
            let place =
                usize::from_str_radix(&draw.hex()[..8], 16).unwrap_or(0) % (context.len() + 1);
            context.insert(place, oracle);
        }
        let prompt = Retrieved { passages: context }.prompt(&task.instruction);
        let messages = match &mut record.body {
            RecordBody::Chat { messages } | RecordBody::Rewarded { messages, .. } => messages,
            RecordBody::Preference { prompt, .. } => prompt,
            _ => continue,
        };
        if let Some(user) = messages.iter_mut().find(|m| m.role == "user") {
            user.content = prompt;
        }
    }
    Ok(projection)
}
