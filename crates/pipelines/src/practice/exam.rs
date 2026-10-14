// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements leakage-free held-out examinations of
// code-editing agents, for its clients. If your team needs expertise in
// reserving an exam from a repository before any training data is built,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! The exam's families, decided before any rollout and never trained on.
//!
//! A task's family is the file its fault was seeded in (its first declared
//! concept). Which families are the exam is a stable hash of the family's
//! name and [`EXAM_SEED`], so the choice does not depend on the order tasks
//! were made in. It is also remembered: the families an earlier run reserved
//! are kept under a pointer in the store, are the exam in every later run
//! (the tasks of a later revision in them are kept apart as well), and count
//! toward the number asked for. A model trained by one run is therefore
//! never examined on what a later run would have trained it on.
//!
//! The reserved tasks are not solved here, so no attempt at them exists to
//! be trained on; [`assert_no_exam_family`] additionally reads the dataset
//! that was written and refuses a run whose training records carry one.

use std::collections::BTreeSet;
use std::path::Path;

use serde::{Deserialize, Serialize};
use splinter_core::digest::Digest;
use splinter_store::tasks::{TaskEntry, TaskSet, TaskSetId};

use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::{io, OrchestratorError};

/// Varies which families a run reserves; the same seed over the same
/// families reserves the same ones whatever order the tasks were made in.
pub const EXAM_SEED: u64 = 0;

/// The class of the stored record of the reserved families.
const RESERVED: &str = "practice_exam_families";

/// The pointer that names the latest record of the reserved families.
const RESERVED_POINTER: &str = "practice-exam-families";

/// The families reserved so far, as the store keeps them.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
struct Reserved {
    families: BTreeSet<String>,
}

/// What the reserve stage decided.
#[derive(Clone, Debug, Serialize)]
pub struct ExamReserved {
    /// The families of the exam among this run's tasks, by name.
    pub families: Vec<String>,
    /// Of those, the families an earlier run had already reserved.
    pub reserved_before: usize,
    /// The tasks of the exam families, set apart and not solved.
    pub exam_tasks: usize,
    /// The set the exam tasks are in; `None` when there are none.
    pub exam_task_set: Option<TaskSetId>,
    /// The tasks left to solve and train on.
    pub training_tasks: usize,
    /// The families those are in.
    pub training_families: usize,
}

/// Where a run's tasks went.
pub(super) struct Divided {
    pub(super) reserved: ExamReserved,
    /// The tasks to solve: those of no exam family.
    pub(super) training: Vec<TaskEntry>,
    /// Every exam family, this run's and earlier runs'.
    pub(super) exam_families: BTreeSet<String>,
}

/// The family `entry`'s task declares; `None` for a task that declares none.
fn family_of(ctx: &Context, entry: &TaskEntry) -> Result<Option<String>, OrchestratorError> {
    Ok(ctx.tasks().get(&entry.task)?.concepts.into_iter().next())
}

/// How a family ranks for the exam: the hash of its name under the seed.
fn rank(family: &str) -> String {
    let mut input = EXAM_SEED.to_le_bytes().to_vec();
    input.extend_from_slice(family.as_bytes());
    Digest::of(&input).to_string()
}

/// Divides `members`, the tasks a run made, into the exam's and the ones to
/// train on, reserving `asked` families (a quarter of the families, rounded
/// down, when `None`) on top of those earlier runs reserved, and remembers
/// the reservation. Refused when the reservation would leave nothing to
/// train on.
pub(super) fn reserve(
    ctx: &Context,
    members: Vec<TaskEntry>,
    asked: Option<usize>,
) -> Result<Divided, OrchestratorError> {
    let before = reserved_families(ctx)?;
    let mut by_family: Vec<(Option<String>, TaskEntry)> = Vec::with_capacity(members.len());
    for entry in members {
        by_family.push((family_of(ctx, &entry)?, entry));
    }
    let present: BTreeSet<String> = by_family
        .iter()
        .filter_map(|(family, _)| family.clone())
        .collect();
    let already: BTreeSet<String> = present.intersection(&before).cloned().collect();
    let wanted = asked.unwrap_or(present.len() / 4);
    let mut fresh: Vec<&String> = present.difference(&already).collect();
    fresh.sort_by_cached_key(|family| rank(family));
    fresh.truncate(wanted.saturating_sub(already.len()));
    let chosen: BTreeSet<String> = fresh.into_iter().cloned().collect();
    if already.len() + chosen.len() >= present.len() && !present.is_empty() {
        return Err(OrchestratorError::Refused(format!(
            "reserving {} of {} families would leave nothing to train on: name fewer exam \
             families, or practice on a repository with more files under test",
            already.len() + chosen.len(),
            present.len()
        )));
    }
    let exam: BTreeSet<String> = already.union(&chosen).cloned().collect();
    remember(ctx, &before, &chosen)?;

    let (exam_entries, training): (Vec<_>, Vec<_>) = by_family
        .into_iter()
        .partition(|(family, _)| family.as_ref().is_some_and(|f| exam.contains(f)));
    let exam_entries: Vec<TaskEntry> = exam_entries.into_iter().map(|(_, e)| e).collect();
    let training: Vec<TaskEntry> = training.into_iter().map(|(_, e)| e).collect();
    let exam_task_set = if exam_entries.is_empty() {
        None
    } else {
        Some(ctx.tasks().put_set(&TaskSet {
            name: "exam tasks reserved before any attempt".into(),
            members: exam_entries.clone(),
        })?)
    };
    let training_families = present.len() - already.len() - chosen.len();
    Ok(Divided {
        reserved: ExamReserved {
            families: exam.iter().cloned().collect(),
            reserved_before: already.len(),
            exam_tasks: exam_entries.len(),
            exam_task_set,
            training_tasks: training.len(),
            training_families,
        },
        training,
        exam_families: before.union(&exam).cloned().collect(),
    })
}

/// The families every earlier run reserved.
fn reserved_families(ctx: &Context) -> Result<BTreeSet<String>, OrchestratorError> {
    let Some((_, value)) = ctx.workspace().pointer(RESERVED_POINTER)? else {
        return Ok(BTreeSet::new());
    };
    let digest = Digest::parse(&value).map_err(|e| {
        OrchestratorError::Refused(format!(
            "the pointer to the reserved families is corrupt: {e}"
        ))
    })?;
    let stored: Option<Reserved> = ctx.workspace().get_document(RESERVED, &digest)?;
    Ok(stored.map(|r| r.families).unwrap_or_default())
}

/// Records `before` and `chosen` as the families reserved so far.
fn remember(
    ctx: &Context,
    before: &BTreeSet<String>,
    chosen: &BTreeSet<String>,
) -> Result<(), OrchestratorError> {
    if chosen.is_empty() {
        return Ok(());
    }
    let all = Reserved {
        families: before.union(chosen).cloned().collect(),
    };
    let digest = ctx.workspace().put_document(RESERVED, &all)?;
    let current = ctx
        .workspace()
        .pointer(RESERVED_POINTER)?
        .map(|(_, value)| value);
    ctx.workspace().move_pointer(
        RESERVED_POINTER,
        current.as_deref(),
        digest.as_str(),
        &ctx.clock().utc_now(),
    )?;
    Ok(())
}

/// The gate: refuses a dataset one of whose records is of a family of
/// `exam`, or of no family at all (a record whose family is not known
/// cannot be shown to be outside the exam).
pub(super) fn assert_no_exam_family(
    dataset: &Path,
    exam: &BTreeSet<String>,
) -> Result<(), OrchestratorError> {
    #[derive(Deserialize)]
    struct Line {
        metadata: Metadata,
    }
    #[derive(Deserialize)]
    struct Metadata {
        #[serde(default)]
        group: Option<String>,
    }
    let text = std::fs::read_to_string(dataset).map_err(io(dataset))?;
    for (n, line) in text.lines().filter(|l| !l.trim().is_empty()).enumerate() {
        let record: Line = serde_json::from_str(line).map_err(|e| {
            OrchestratorError::Refused(format!(
                "record {} of {} cannot be read for the leakage check: {e}",
                n + 1,
                dataset.display()
            ))
        })?;
        match record.metadata.group {
            Some(group) if exam.contains(&group) => {
                return Err(OrchestratorError::Refused(format!(
                    "leakage: record {} of {} is of the exam family {group}",
                    n + 1,
                    dataset.display()
                )));
            }
            None => {
                return Err(OrchestratorError::Refused(format!(
                    "leakage check: record {} of {} names no family",
                    n + 1,
                    dataset.display()
                )));
            }
            Some(_) => {}
        }
    }
    Ok(())
}
