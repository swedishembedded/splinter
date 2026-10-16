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
//! name and the exam's seed, so the choice does not depend on the order tasks
//! were made in. It is also remembered: the families an earlier run reserved
//! are kept under a pointer in the store with the seed they were chosen by,
//! are the exam in every later run (the tasks of a later revision in them are
//! kept apart as well), and count toward the number asked for. The first run
//! that reserves draws the seed; every later run reads it back and never
//! draws another for the same exam. A model trained by one run is therefore
//! never examined on what a later run would have trained it on.
//!
//! Two exams over the same repository are comparable only when they were
//! chosen by the same recorded seed.
//!
//! The reserved tasks are not solved here, so no attempt at them exists to
//! be trained on; [`crate::exam_seal`] additionally refuses a dataset build
//! whose records carry one. The other direction is kept as well: the
//! families a build trained on are remembered, and a family a model was
//! trained on is never reserved later, since a later exam would then be
//! examining what an earlier release learned. A run may therefore reserve
//! fewer families than it was asked to.

use std::collections::BTreeSet;

use serde::Serialize;
use splinter_core::experience::Task;
use splinter_core::rng::Seed;
use splinter_store::tasks::{TaskEntry, TaskSet, TaskSetId};

use crate::canaries;
use crate::exam_seal;
use crate::family::{family_of_task, rank};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

/// What the reserve stage decided.
#[derive(Clone, Debug, Serialize)]
pub struct ExamReserved {
    /// The seed the exam families were chosen by: the one the first run to
    /// reserve drew and every later run read back. Reserving again with it
    /// over the same tasks chooses the same families.
    pub seed: Seed,
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
    let task = ctx.tasks().get(&entry.task)?;
    Ok(family_of_task(&task).map(str::to_string))
}

/// The seed the remembered exam families were chosen by; `None` when no run
/// has reserved any yet.
pub(super) fn remembered_seed(ctx: &Context) -> Result<Option<Seed>, OrchestratorError> {
    exam_seal::reserved_seed(ctx)
}

/// Divides `members`, the tasks a run made, into the exam's and the ones to
/// train on, reserving `asked` families (a quarter of the families, rounded
/// down, when `None`) on top of those earlier runs reserved, and remembers
/// the reservation. A family an earlier run trained on is not reserved, so
/// fewer than asked may be. Refused when the reservation would leave nothing
/// to train on. `seed` is [`remembered_seed`] when there is one, else the
/// seed this run's exam is chosen by and remembered with.
pub(super) fn reserve(
    ctx: &Context,
    members: Vec<TaskEntry>,
    asked: Option<usize>,
    seed: Seed,
) -> Result<Divided, OrchestratorError> {
    let before = exam_seal::reserved(ctx)?;
    let trained = exam_seal::trained(ctx)?;
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
    let mut fresh: Vec<&String> = present
        .difference(&already)
        .filter(|family| !trained.contains(*family))
        .collect();
    fresh.sort_by_cached_key(|family| rank(family, seed));
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
    exam_seal::remember_reserved(ctx, &chosen, seed)?;

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
            seed,
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

/// The canaries of the exam this run reserved: its workspace tasks with the
/// hidden assertions also required to fail ([`crate::canaries`]), at most
/// `limit` ([`crate::canaries::CANARY_COUNT`] when `None`). Empty when the
/// run reserved no exam, or `limit` is 0. Every release a practice run makes
/// carries them, so the gate always asks whether the exam can be passed by
/// tampering with what reports its result.
pub fn exam_canaries(
    ctx: &Context,
    reserved: Option<&ExamReserved>,
    limit: Option<usize>,
) -> Result<Vec<Task>, OrchestratorError> {
    let Some(exam) = reserved.and_then(|r| r.exam_task_set.as_ref()) else {
        return Ok(Vec::new());
    };
    let made = canaries::make(ctx, exam, limit)?;
    match made.canary_set {
        Some(set) => canaries::load(ctx, &set),
        None => Ok(Vec::new()),
    }
}
