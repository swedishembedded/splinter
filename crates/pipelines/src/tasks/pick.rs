// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements leakage-free held-out splits of
// code-editing task sets, for its clients. If your team needs expertise in
// holding out whole task families before any training data is built, you
// can procure our services by sending an email to info@swedishembedded.com.

//! `tasks pick`: the union of task sets divided into a picked set and the
//! rest, by a stable hash of a seed and the task's family.
//!
//! The division is a function of the seed and the tasks alone: naming the
//! sets in another order, or listing a task in two of them, gives the same
//! two sets. Both are written with their members in task-id order, so the
//! same division is the same two set ids. A task that declares no family is
//! a family of its own.
//!
//! The same [`rank`] chooses the practice exam's families, so a development
//! holdout drawn here is chosen the way the exam is.

use std::collections::BTreeMap;

use serde::Serialize;
use splinter_core::rng::Seed;
use splinter_store::tasks::{TaskEntry, TaskSet, TaskSetId};

use crate::family::{family_of_task, rank};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

/// How much of the union is picked.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Quota {
    /// This fraction of the families, rounded to the nearest whole number of
    /// families, with every task of each.
    FamilyShare(f64),
    /// At most this many tasks of every family.
    PerFamily(usize),
}

/// What `tasks pick` did.
#[derive(Clone, Debug, Serialize)]
pub struct Picked {
    /// The seed the division was made by.
    pub seed: Seed,
    /// The picked set.
    pub picked: TaskSetId,
    /// Tasks in it.
    pub picked_tasks: usize,
    /// Families it holds a task of.
    pub picked_families: usize,
    /// Everything else in the union.
    pub rest: TaskSetId,
    /// Tasks in it.
    pub rest_tasks: usize,
    /// Families it holds a task of.
    pub rest_families: usize,
}

/// Divides the union of the task sets `sets` as `quota` says, by `seed`.
pub fn pick(
    ctx: &Context,
    sets: &[TaskSetId],
    quota: Quota,
    seed: Seed,
) -> Result<Picked, OrchestratorError> {
    match quota {
        Quota::FamilyShare(share) if !(0.0..=1.0).contains(&share) => {
            return Err(OrchestratorError::Refused(format!(
                "a family share is between 0 and 1, not {share}"
            )));
        }
        Quota::PerFamily(0) => {
            return Err(OrchestratorError::Refused(
                "a cap of 0 tasks per family picks nothing".into(),
            ));
        }
        _ => {}
    }
    if sets.is_empty() {
        return Err(OrchestratorError::Refused(
            "name at least one task set".into(),
        ));
    }
    let mut by_family: BTreeMap<String, BTreeMap<String, TaskEntry>> = BTreeMap::new();
    for set in sets {
        for entry in ctx.tasks().get_set(set)?.members {
            let task = ctx.tasks().get(&entry.task)?;
            let id = entry.task.to_string();
            let family = family_of_task(&task).map_or_else(|| id.clone(), str::to_string);
            by_family
                .entry(family)
                .or_default()
                .entry(id)
                .or_insert(entry);
        }
    }

    let mut picked: Vec<TaskEntry> = Vec::new();
    let mut rest: Vec<TaskEntry> = Vec::new();
    match quota {
        Quota::FamilyShare(share) => {
            let mut ranked: Vec<&String> = by_family.keys().collect();
            ranked.sort_by_cached_key(|family| rank(family, seed));
            let count = (share * ranked.len() as f64).round() as usize;
            let chosen: Vec<&String> = ranked.into_iter().take(count).collect();
            for (family, tasks) in &by_family {
                let side = if chosen.contains(&family) {
                    &mut picked
                } else {
                    &mut rest
                };
                side.extend(tasks.values().cloned());
            }
        }
        Quota::PerFamily(cap) => {
            for tasks in by_family.values() {
                let mut ranked: Vec<(&String, &TaskEntry)> = tasks.iter().collect();
                ranked.sort_by_cached_key(|(id, _)| rank(id, seed));
                for (at, (_, entry)) in ranked.into_iter().enumerate() {
                    if at < cap { &mut picked } else { &mut rest }.push(entry.clone());
                }
            }
        }
    }
    let families_in = |entries: &[TaskEntry]| {
        by_family
            .values()
            .filter(|tasks| {
                entries
                    .iter()
                    .any(|e| tasks.contains_key(&e.task.to_string()))
            })
            .count()
    };
    let (picked_families, rest_families) = (families_in(&picked), families_in(&rest));
    let (picked_tasks, rest_tasks) = (picked.len(), rest.len());
    let put = |name: &str, mut members: Vec<TaskEntry>| {
        members.sort_by_key(|e| e.task.to_string());
        ctx.tasks().put_set(&TaskSet {
            name: format!("{name} (seed {})", seed.value()),
            members,
        })
    };
    Ok(Picked {
        seed,
        picked: put("tasks picked", picked)?,
        picked_tasks,
        picked_families,
        rest: put("tasks left after the pick", rest)?,
        rest_tasks,
        rest_families,
    })
}
