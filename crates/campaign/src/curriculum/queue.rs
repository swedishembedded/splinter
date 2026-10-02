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
//! Each queued concept is one file, `<root>/curriculum/queue/<hex>.json`,
//! named by the digest of the concept: queuing a concept again merges the
//! new sections and reasons into the entry already there.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use splinter_knowledge::concepts::{Concept, ConceptResolver, SectionRef};
use splinter_record::digest::{canonical_json, Digest};
use splinter_record::write_atomic;

use crate::context::Context;
use crate::error::{io, CampaignError};
use crate::release::gate::GateReport;

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

fn queue_dir(ctx: &Context) -> PathBuf {
    ctx.root().curriculum().join("queue")
}

fn entry_path(ctx: &Context, concept: &Concept) -> Result<PathBuf, CampaignError> {
    let bytes = canonical_json(concept).map_err(|source| CampaignError::Json {
        what: "concept".into(),
        source,
    })?;
    Ok(queue_dir(ctx).join(format!("{}.json", Digest::of(&bytes).hex())))
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
            if !ctx.tasks().contains(&task_id) {
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
        let path = entry_path(ctx, concept)?;
        let mut entry = read(&path)?.unwrap_or_else(|| QueuedConcept {
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
        write(&path, &entry)?;
    }
    Ok(found.into_keys().collect())
}

/// Every queued concept, in concept order.
pub fn pending(ctx: &Context) -> Result<Vec<QueuedConcept>, CampaignError> {
    let dir = queue_dir(ctx);
    let listing = match std::fs::read_dir(&dir) {
        Ok(listing) => listing,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(io(&dir)(e)),
    };
    let mut queued = Vec::new();
    for item in listing {
        let path = item.map_err(io(&dir))?.path();
        if path.extension().is_some_and(|e| e == "json") {
            if let Some(entry) = read(&path)? {
                queued.push(entry);
            }
        }
    }
    queued.sort_by(|a, b| a.concept.cmp(&b.concept));
    Ok(queued)
}

/// Takes `done` off the queue.
pub fn complete(ctx: &Context, done: &[QueuedConcept]) -> Result<(), CampaignError> {
    for entry in done {
        let path = entry_path(ctx, &entry.concept)?;
        match std::fs::remove_file(&path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(io(&path)(e)),
        }
    }
    Ok(())
}

fn read(path: &std::path::Path) -> Result<Option<QueuedConcept>, CampaignError> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(io(path)(e)),
    };
    serde_json::from_str(&text)
        .map(Some)
        .map_err(|source| CampaignError::Json {
            what: path.display().to_string(),
            source,
        })
}

fn write(path: &std::path::Path, entry: &QueuedConcept) -> Result<(), CampaignError> {
    let text = serde_json::to_string_pretty(entry).map_err(|source| CampaignError::Json {
        what: "queued concept".into(),
        source,
    })?;
    write_atomic(path, &text).map_err(io(path))
}
