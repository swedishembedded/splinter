// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements curricula that spend a learner's solver and
// training budget where it still learns, for its clients. If your team needs
// expertise in continual learning or retention, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Concepts queued for new tasks: what a release gate saw forgotten.
//!
//! When the gate's retention check finds an earlier release's suite
//! dropped beyond its bound, every concept of a task in that suite the
//! champion got right and the candidate got wrong is queued
//! ([`enqueue_retention`]), with the source sections new tasks for it can
//! be generated from: the sections the forgotten tasks' evidence falls in.
//! The next `learn` generates tasks from every queued concept's sections
//! beside its own sources ([`pending`]), and takes the concepts off the
//! queue once its task set is stored ([`complete`]); a run that fails
//! before that leaves them queued.
//!
//! Each queued concept is a pointer, `queue-<digest of the concept>`, to the
//! document that holds its entry, or to `done` once it was taken off the queue:
//! queuing a concept again merges the new sections and reasons into its entry
//! and moves the pointer, so two processes queuing at once cannot lose one
//! another's merge, and the history of a concept's queuing stays on record.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use splinter_core::digest::{canonical_json, Digest};
use splinter_knowledge::concepts::{Concept, ConceptResolver, SectionRef};
use splinter_store::experiences::StoreError;

use crate::context::Context;
use crate::error::CampaignError;
use splinter_eval::gate::GateReport;

const ENTRY: &str = "queue_entry";
const PREFIX: &str = "queue-";
const DONE: &str = "done";

/// How many times a merge retries when another process moved the same
/// pointer first.
const MERGE_ATTEMPTS: usize = 16;

/// A concept waiting for new tasks.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QueuedConcept {
    /// The concept.
    pub concept: Concept,
    /// The source sections new tasks for it are generated from; empty when
    /// the forgotten tasks named none, and there is nothing to generate
    /// from.
    pub sections: Vec<SectionRef>,
    /// Why it was queued, one line per gate that queued it.
    pub reasons: Vec<String>,
    /// When it was first queued, from the injected clock.
    pub queued_at: String,
}

fn pointer_of(concept: &Concept) -> Result<String, CampaignError> {
    let bytes = canonical_json(concept).map_err(|source| CampaignError::Json {
        what: "concept".into(),
        source,
    })?;
    Ok(format!("{PREFIX}{}", &Digest::of(&bytes).hex()[..32]))
}

/// The entry `concept`'s pointer holds, and the value to move it from.
fn entry_of(
    ctx: &Context,
    pointer: &str,
) -> Result<(Option<String>, Option<QueuedConcept>), CampaignError> {
    let Some((_, value)) = ctx.workspace().pointer(pointer)? else {
        return Ok((None, None));
    };
    if value == DONE {
        return Ok((Some(value), None));
    }
    let digest = Digest::parse(&value)
        .map_err(|e| CampaignError::Refused(format!("queue pointer {pointer} is corrupt: {e}")))?;
    Ok((Some(value), ctx.workspace().get_document(ENTRY, &digest)?))
}

/// Queues the concepts of every task a failed retention suite of `gate`
/// lost; returns them, each once, in order.
pub fn enqueue_retention(ctx: &Context, gate: &GateReport) -> Result<Vec<Concept>, CampaignError> {
    let Some(retention) = &gate.retention.measured else {
        return Ok(Vec::new());
    };
    let mut resolver = ConceptResolver::new(ctx.sources());
    let mut found: BTreeMap<Concept, (BTreeSet<SectionRef>, BTreeSet<String>)> = BTreeMap::new();
    for suite in retention.suites.iter().filter(|s| !s.passed) {
        let reason = format!(
            "release {}'s suite dropped {} (bound {})",
            suite.release,
            suite
                .drop
                .map_or_else(|| "unmeasured".to_string(), |d| format!("{d:.4}")),
            retention.bound
        );
        for lost in &suite.lost {
            let Ok(task_id) = Digest::parse(lost) else {
                continue;
            };
            if !ctx.tasks().contains(&task_id)? {
                continue;
            }
            let task = ctx.tasks().get(&task_id)?;
            let sections = resolver.evidence_sections(&task)?;
            for concept in resolver.concepts(&task)? {
                let (into, reasons) = found.entry(concept).or_default();
                into.extend(sections.iter().cloned());
                reasons.insert(reason.clone());
            }
        }
    }
    for (concept, (sections, reasons)) in &found {
        let pointer = pointer_of(concept)?;
        for attempt in 0..MERGE_ATTEMPTS {
            ctx.workspace().refresh()?;
            let (from, existing) = entry_of(ctx, &pointer)?;
            let mut entry = existing.unwrap_or_else(|| QueuedConcept {
                concept: concept.clone(),
                sections: Vec::new(),
                reasons: Vec::new(),
                queued_at: ctx.clock().utc_now(),
            });
            for section in sections {
                if !entry.sections.contains(section) {
                    entry.sections.push(section.clone());
                }
            }
            for reason in reasons {
                if !entry.reasons.contains(reason) {
                    entry.reasons.push(reason.clone());
                }
            }
            let stored = ctx.workspace().put_document(ENTRY, &entry)?;
            if from.as_deref() == Some(stored.as_str()) {
                break;
            }
            match ctx.workspace().move_pointer(
                &pointer,
                from.as_deref(),
                stored.as_str(),
                &ctx.clock().utc_now(),
            ) {
                Ok(_) => break,
                // Another process moved it first: merge into what it left.
                Err(StoreError::PointerConflict { .. }) if attempt + 1 < MERGE_ATTEMPTS => {}
                Err(other) => return Err(other.into()),
            }
        }
    }
    Ok(found.into_keys().collect())
}

/// Every queued concept, in concept order.
pub fn pending(ctx: &Context) -> Result<Vec<QueuedConcept>, CampaignError> {
    ctx.workspace().refresh()?;
    let mut queued: Vec<QueuedConcept> = Vec::new();
    for (_, value) in ctx.workspace().pointers(PREFIX)? {
        if value == DONE {
            continue;
        }
        let digest = Digest::parse(&value)
            .map_err(|e| CampaignError::Refused(format!("a queue pointer is corrupt: {e}")))?;
        if let Some(entry) = ctx.workspace().get_document(ENTRY, &digest)? {
            queued.push(entry);
        }
    }
    queued.sort_by(|a, b| a.concept.cmp(&b.concept));
    Ok(queued)
}

/// Takes `done` off the queue.
pub fn complete(ctx: &Context, done: &[QueuedConcept]) -> Result<(), CampaignError> {
    for entry in done {
        let pointer = pointer_of(&entry.concept)?;
        let (from, _) = entry_of(ctx, &pointer)?;
        if from.as_deref().is_some_and(|value| value != DONE) {
            match ctx.workspace().move_pointer(
                &pointer,
                from.as_deref(),
                DONE,
                &ctx.clock().utc_now(),
            ) {
                // Queued again meanwhile: it stays queued.
                Ok(_) | Err(StoreError::PointerConflict { .. }) => {}
                Err(other) => return Err(other.into()),
            }
        }
    }
    Ok(())
}
