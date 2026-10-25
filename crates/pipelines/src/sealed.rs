// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements examinations that stay sealed across every
// run of a learning pipeline, for its clients. If your team needs expertise
// in protecting a held-out final exam from the stages that read a repository,
// you can procure our services by sending an email to info@swedishembedded.com.

//! The sealed final exam, kept in the state root rather than in the command
//! that happens to be running.
//!
//! A validated [`ExperimentPlan`] writes its sealed families - their ids,
//! their git tree ids and the snapshot digests when the catalog states one -
//! under the state root ([`record`]). Every stage that reads a repository or a
//! task then holds the recorded seal against it, whether or not the stage was
//! given a plan: a repository whose tree is sealed is refused before it is
//! taken in as a source, and a task whose environment is a sealed tree is
//! refused before it is admitted, solved, verified, taught or trained on.
//!
//! The seal is matched by tree id (and by the family id the plan stored), not
//! by the name a later catalog gives the family, so renaming a family does not
//! open it. A recorded seal is never removed by a later plan: the plan check
//! says no sealed family may be scheduled, and a seal that could be lifted by
//! editing a plan would not be a seal.
//!
//! [`ExperimentPlan`]: crate::experiment::ExperimentPlan

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use splinter_core::digest::Digest;
use splinter_core::experience::Task;
use splinter_data::Corpus;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_store::tasks::TaskSetId;

use crate::experiment::ExperimentPlan;

/// The class of the stored record of the sealed families.
const SEALED: &str = "experiment_sealed_families";

/// The pointer that names the latest record of the sealed families.
const SEALED_POINTER: &str = "experiment-sealed-families";

/// One sealed final-exam family, as the state root keeps it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SealedFamily {
    /// The family's id in the catalog that sealed it.
    pub id: String,
    /// The git tree id of its files: what a snapshot of it records.
    pub tree: String,
    /// The digest of the workspace snapshot the catalog states for it, when
    /// it states one; a task in that environment is refused too.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub environment: Option<String>,
}

/// The record kept under [`SEALED_POINTER`].
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct Sealed {
    families: Vec<SealedFamily>,
}

/// The sealed families kept under the state root, in the order they were
/// recorded and without repeats.
pub fn families(ctx: &Context) -> Result<Vec<SealedFamily>, OrchestratorError> {
    Ok(stored(ctx)?
        .map(|sealed| sealed.families)
        .unwrap_or_default())
}

/// The tree ids of the sealed families.
pub fn trees(ctx: &Context) -> Result<BTreeSet<String>, OrchestratorError> {
    Ok(families(ctx)?.into_iter().map(|f| f.tree).collect())
}

/// Records the sealed families of a validated `plan` in the state root, on top
/// of those recorded before. Recording the same families again changes
/// nothing; a later plan cannot remove a seal.
pub fn record(ctx: &Context, plan: &ExperimentPlan) -> Result<(), OrchestratorError> {
    let added: Vec<SealedFamily> = plan
        .sealed()
        .iter()
        .filter_map(|id| plan.family(id))
        .map(|family| SealedFamily {
            id: family.id.clone(),
            tree: family.tree.clone(),
            environment: family.environment.clone(),
        })
        .collect();
    if added.is_empty() {
        return Ok(());
    }
    let mut sealed = stored(ctx)?.unwrap_or_default();
    let mut changed = false;
    for family in added {
        match sealed
            .families
            .iter_mut()
            .find(|f| f.id == family.id && f.tree == family.tree)
        {
            Some(kept) => {
                if kept.environment.is_none() && family.environment.is_some() {
                    kept.environment = family.environment;
                    changed = true;
                }
            }
            None => {
                sealed.families.push(family);
                changed = true;
            }
        }
    }
    if !changed {
        return Ok(());
    }
    let digest = ctx.workspace().put_document(SEALED, &sealed)?;
    let current = ctx
        .workspace()
        .pointer(SEALED_POINTER)?
        .map(|(_, value)| value);
    ctx.workspace().move_pointer(
        SEALED_POINTER,
        current.as_deref(),
        digest.as_str(),
        &ctx.clock().utc_now(),
    )?;
    Ok(())
}

/// Refuses `tree` when it is a sealed family's tree.
pub fn admit_tree(ctx: &Context, tree: &str) -> Result<(), OrchestratorError> {
    match families(ctx)?.into_iter().find(|f| f.tree == tree) {
        Some(family) => Err(OrchestratorError::Refused(refusal_of(&family))),
        None => Ok(()),
    }
}

/// Refuses `task` when its environment is a sealed family's tree or snapshot,
/// or when it declares a sealed family's id.
pub fn admit_task(ctx: &Context, task: &Task) -> Result<(), OrchestratorError> {
    match refuses_task(&families(ctx)?, task) {
        Some(why) => Err(OrchestratorError::Refused(why)),
        None => Ok(()),
    }
}

/// Refuses the task set `set` when any of its tasks is of a sealed family.
pub fn admit_set(ctx: &Context, set: &TaskSetId) -> Result<(), OrchestratorError> {
    let store = ctx.tasks();
    let members = store.get_set(set)?.members;
    let sealed = families(ctx)?;
    for member in members {
        if let Some(why) = refuses_task(&sealed, &store.get(&member.task)?) {
            return Err(OrchestratorError::Refused(why));
        }
    }
    Ok(())
}

/// Refuses a corpus with any attempt at a task of a sealed family.
pub fn admit_corpus(ctx: &Context, corpus: &Corpus) -> Result<(), OrchestratorError> {
    let sealed = families(ctx)?;
    for entry in corpus.entries() {
        if let Some(why) = refuses_task(&sealed, &entry.experience.to_task()) {
            return Err(OrchestratorError::Refused(why));
        }
    }
    Ok(())
}

/// Why `task` may not be read under `sealed`, if it may not.
pub fn refuses_task(sealed: &[SealedFamily], task: &Task) -> Option<String> {
    if task.environment.kind != splinter_core::experience::Environment::WORKSPACE {
        // A sealed family is files; a closed-book or runtime task is not one
        // of them unless it declares the family's id below.
        return task
            .concepts
            .iter()
            .find_map(|concept| sealed.iter().find(|f| &f.id == concept).map(refusal_of));
    }
    let spec = &task.environment.spec;
    let tree = spec.get("tree_id").and_then(serde_json::Value::as_str);
    let snapshot = spec.get("snapshot").and_then(serde_json::Value::as_str);
    for family in sealed {
        let by_tree = tree.is_some_and(|t| t == family.tree);
        let by_snapshot = snapshot
            .zip(family.environment.as_deref())
            .is_some_and(|(s, e)| s == e);
        let by_id = task.concepts.iter().any(|concept| concept == &family.id);
        if by_tree || by_snapshot || by_id {
            return Some(refusal_of(family));
        }
    }
    None
}

/// The refusal for reading a sealed family, naming it and the tree that
/// identifies it.
fn refusal_of(family: &SealedFamily) -> String {
    format!(
        "sealed: family {:?} (tree {}) is the sealed final exam; no stage may read it",
        family.id, family.tree
    )
}

/// The record under [`SEALED_POINTER`], when one exists.
fn stored(ctx: &Context) -> Result<Option<Sealed>, OrchestratorError> {
    let Some((_, value)) = ctx.workspace().pointer(SEALED_POINTER)? else {
        return Ok(None);
    };
    let digest = Digest::parse(&value).map_err(|e| {
        OrchestratorError::Refused(format!("the pointer {SEALED_POINTER} is corrupt: {e}"))
    })?;
    Ok(ctx.workspace().get_document(SEALED, &digest)?)
}
