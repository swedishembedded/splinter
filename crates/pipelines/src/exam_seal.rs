// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements leakage-free held-out examinations of
// code-editing agents, for its clients. If your team needs expertise in
// keeping an exam's files out of the data a model is trained on, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The seal between an exam and the data trained on: which families the
//! store keeps reserved and which it has trained on, and the check that a
//! dataset built from workspace attempts crosses neither way.
//!
//! * An experience of a reserved family is refused, naming the family: the
//!   exam's attempts are never training data.
//! * A supervised action whose tool-call arguments name a file of a reserved
//!   family is refused: the teacher works in the whole snapshot and could
//!   write a qualification file's code.
//! * A workspace record that names no family cannot be shown to be outside
//!   the exam and is refused.
//! * The families a clean build trains on are remembered, so no later run
//!   reserves one of them (see [`crate::practice`]).
//!
//! * An experience of a benchmark task that is not in the benchmark's `train`
//!   split is refused: it is examination material ([`crate::bench::split`]).
//!
//! A family is the file a task's fault was seeded in (its first concept).

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use splinter_core::chat::WireMessage;
use splinter_core::digest::Digest;
use splinter_core::experience::{Experience, ExperienceId};
use splinter_core::kinds::WORKSPACE_CHANGE;
use splinter_core::rng::Seed;
use splinter_data::{Corpus, Projection, RecordBody};

use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

/// The class of the stored record of the reserved families.
const RESERVED: &str = "practice_exam_families";

/// The pointer that names the latest record of the reserved families.
const RESERVED_POINTER: &str = "practice-exam-families";

/// The class of the stored record of the families trained on.
const TRAINED: &str = "practice_trained_families";

/// The pointer that names the latest record of the families trained on.
const TRAINED_POINTER: &str = "practice-trained-families";

/// A set of families, as the store keeps them.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
struct Families {
    families: BTreeSet<String>,
    /// The seed the families were chosen by. Records of reserved families
    /// made before it was kept have none and were chosen by seed 0; the
    /// record of the families trained on never has one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    seed: Option<u64>,
}

/// Every family an earlier run reserved for an exam.
pub fn reserved(ctx: &Context) -> Result<BTreeSet<String>, OrchestratorError> {
    recorded(ctx, RESERVED, RESERVED_POINTER)
}

/// Every family a dataset was built from.
pub fn trained(ctx: &Context) -> Result<BTreeSet<String>, OrchestratorError> {
    recorded(ctx, TRAINED, TRAINED_POINTER)
}

/// The seed the reserved families were chosen by; `None` when no run has
/// reserved any yet.
pub fn reserved_seed(ctx: &Context) -> Result<Option<Seed>, OrchestratorError> {
    Ok(stored(ctx, RESERVED, RESERVED_POINTER)?.map(|record| Seed::new(record.seed.unwrap_or(0))))
}

/// Remembers `added` as reserved, chosen by `seed`, on top of those already.
pub fn remember_reserved(
    ctx: &Context,
    added: &BTreeSet<String>,
    seed: Seed,
) -> Result<(), OrchestratorError> {
    let before = reserved(ctx)?;
    remember(
        ctx,
        (RESERVED, RESERVED_POINTER),
        &before,
        added,
        Some(seed),
    )
}

/// Remembers `families` as trained on, so that no later run reserves one of
/// them.
pub fn remember_trained(
    ctx: &Context,
    families: &BTreeSet<String>,
) -> Result<(), OrchestratorError> {
    let before = trained(ctx)?;
    remember(ctx, (TRAINED, TRAINED_POINTER), &before, families, None)
}

/// An experience that would cross the seal, and why.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Leak {
    /// The experience.
    pub experience: ExperienceId,
    /// What crosses the seal.
    pub reason: String,
}

/// Every experience of `corpus` that `projection` of it would carry across
/// the seal (see the module), once each, with the first reason found: an
/// attempt at a reserved family first, then a record that names no family or
/// supervises an action naming a reserved family's file. Empty for a corpus
/// with no workspace attempt, and when no family is reserved.
pub fn leaks(
    ctx: &Context,
    corpus: &Corpus,
    projection: &Projection,
) -> Result<Vec<Leak>, OrchestratorError> {
    let mut found: Vec<Leak> = Vec::new();
    let register = crate::bench::split::trials(ctx)?;
    for entry in corpus.entries() {
        if let Some(reason) = crate::bench::split::exam_only(&register, &entry.experience) {
            found.push(Leak {
                experience: entry.id.clone(),
                reason: format!("experience {}: {reason}", entry.id),
            });
        }
    }
    let exam = reserved(ctx)?;
    if exam.is_empty() {
        return Ok(found);
    }
    let workspace = workspace_entries(corpus);
    let mut note = |experience: &ExperienceId, reason: String| {
        if !found.iter().any(|l| &l.experience == experience) {
            found.push(Leak {
                experience: experience.clone(),
                reason,
            });
        }
    };
    for entry in &workspace {
        if let Some(family) = family_of(&entry.experience).filter(|f| exam.contains(*f)) {
            note(
                &entry.id,
                format!(
                    "experience {} is an attempt at a task of the exam family {family}",
                    entry.id
                ),
            );
        }
    }
    for (n, record) in projection.records.iter().enumerate() {
        let Some(entry) = record
            .metadata
            .experiences
            .first()
            .and_then(|id| workspace.iter().find(|entry| &entry.id == id))
        else {
            continue;
        };
        if family_of(&entry.experience).is_none() {
            note(
                &entry.id,
                format!(
                    "record {} of experience {} names no family, so it cannot be shown to be \
                     outside the exam",
                    n + 1,
                    entry.id
                ),
            );
        }
        if let RecordBody::Chat { messages, .. } = &record.body {
            if let Some(family) = named_exam_file(messages, &exam) {
                note(
                    &entry.id,
                    format!(
                        "record {} of experience {} supervises an action naming the file of the \
                         exam family {family}",
                        n + 1,
                        entry.id
                    ),
                );
            }
        }
    }
    Ok(found)
}

/// The gate for a projection of `corpus`: refuses it when it crosses the
/// seal (see the module), else remembers the families its workspace records
/// train on. Nothing is checked, or remembered, for a corpus with no
/// workspace attempt.
pub fn seal(
    ctx: &Context,
    corpus: &Corpus,
    projection: &Projection,
) -> Result<(), OrchestratorError> {
    if let Some(first) = leaks(ctx, corpus, projection)?.first() {
        return Err(leak(first.reason.clone()));
    }
    let workspace = workspace_entries(corpus);
    let trained: BTreeSet<String> = projection
        .records
        .iter()
        .filter_map(|record| record.metadata.experiences.first())
        .filter_map(|id| workspace.iter().find(|entry| &entry.id == id))
        .filter_map(|entry| family_of(&entry.experience).map(str::to_string))
        .collect();
    if workspace.is_empty() {
        return Ok(());
    }
    remember_trained(ctx, &trained)
}

fn workspace_entries(corpus: &Corpus) -> Vec<&splinter_data::Entry> {
    corpus
        .entries()
        .iter()
        .filter(|entry| entry.experience.task.kind == WORKSPACE_CHANGE)
        .collect()
}

/// The family the task of `experience` declares: its first concept.
fn family_of(experience: &Experience) -> Option<&str> {
    experience.concepts.first().map(String::as_str)
}

fn leak(what: String) -> OrchestratorError {
    OrchestratorError::Refused(format!("leakage: {what}"))
}

/// The exam family, a file, that a supervised tool call in `messages` names
/// in its arguments.
fn named_exam_file<'a>(messages: &[WireMessage], exam: &'a BTreeSet<String>) -> Option<&'a str> {
    messages
        .iter()
        .filter(|message| message.train)
        .flat_map(|message| &message.tool_calls)
        .find_map(|call| {
            exam.iter()
                .find(|file| names_file(&call.function.arguments, file))
                .map(String::as_str)
        })
}

/// Whether `text` names `file` as a path: the file's name, not preceded by
/// a character that would make it the tail of a longer name (`data.py` is
/// not `a_data.py`; `src/data.py` and `./data.py` are `data.py`).
fn names_file(text: &str, file: &str) -> bool {
    text.match_indices(file).any(|(at, _)| {
        text[..at]
            .chars()
            .next_back()
            .is_none_or(|c| !(c.is_alphanumeric() || c == '_' || c == '-' || c == '.'))
    })
}

/// The families kept under `pointer` as documents of `class`.
fn recorded(
    ctx: &Context,
    class: &str,
    pointer: &str,
) -> Result<BTreeSet<String>, OrchestratorError> {
    Ok(stored(ctx, class, pointer)?
        .map(|r| r.families)
        .unwrap_or_default())
}

/// The record kept under `pointer` as a document of `class`, if there is one.
fn stored(
    ctx: &Context,
    class: &str,
    pointer: &str,
) -> Result<Option<Families>, OrchestratorError> {
    let Some((_, value)) = ctx.workspace().pointer(pointer)? else {
        return Ok(None);
    };
    let digest = Digest::parse(&value).map_err(|e| {
        OrchestratorError::Refused(format!("the pointer {pointer} is corrupt: {e}"))
    })?;
    Ok(ctx.workspace().get_document(class, &digest)?)
}

/// Records `before` and `added` as the families kept under `(class,
/// pointer)`, chosen by `seed` when they are an exam's.
fn remember(
    ctx: &Context,
    (class, pointer): (&str, &str),
    before: &BTreeSet<String>,
    added: &BTreeSet<String>,
    seed: Option<Seed>,
) -> Result<(), OrchestratorError> {
    if added.is_subset(before) {
        return Ok(());
    }
    let all = Families {
        families: before.union(added).cloned().collect(),
        seed: seed.map(Seed::value),
    };
    let digest = ctx.workspace().put_document(class, &all)?;
    let current = ctx.workspace().pointer(pointer)?.map(|(_, value)| value);
    ctx.workspace().move_pointer(
        pointer,
        current.as_deref(),
        digest.as_str(),
        &ctx.clock().utc_now(),
    )?;
    Ok(())
}
