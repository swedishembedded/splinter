// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training-data lineage that survives moving
// experience between machines and stores, for its clients. If your team needs
// expertise in curating agent experience across runs, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Experience sets moved from another state root into this one.
//!
//! A learning run can need experience another run's state holds: a teacher's
//! demonstrations admitted there from exports that no longer exist, beside
//! repairs admitted here. [`import_sets`] reads each named set from the other
//! root through the store's own API and writes it here through the same API,
//! whichever layout (loose or packed) either database has:
//!
//! - every member experience, read and checked against its content address,
//!   and written under the same address (a write that lands under another
//!   address is refused: the two builds disagree about the record);
//! - every annotation of it, in the order written (the store keeps one copy of
//!   an annotation said twice), and, before them, every experience a relation
//!   names, with its own annotations, so no relation points nowhere;
//! - the examiner's trial register for the imported experiences, so an
//!   experience admitted as examination material there stays out of every
//!   dataset here;
//! - the set itself, under the same address.
//!
//! Importing again writes nothing. The other root is only read; to be sure
//! of that, import from a copy of a root another process writes.
//!
//! [`copy`] is the same between any two databases, for experiences that are
//! in no set as well: a bundle is staged with it and ingested with it.

use std::collections::{BTreeSet, VecDeque};

use serde::Serialize;
use splinter_core::annotation::AnnotationBody;
use splinter_core::experience::ExperienceId;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_orchestrator::ids;
use splinter_store::experiences::{ExperienceSet, ExperienceStore, SetId};
use splinter_store::workspace::Workspace;
use splinter_store::StateRoot;

use crate::bench::split;

/// What an import did.
#[derive(Clone, Debug, Default, Serialize)]
pub struct SetsImported {
    /// The sets, as this root now holds them (the same ids).
    pub sets: Vec<SetId>,
    /// Experiences written here.
    pub experiences_new: usize,
    /// Experiences this root held already.
    pub experiences_held: usize,
    /// Of the experiences, the ones a relation named outside the sets.
    pub related: usize,
    /// Annotations written here.
    pub annotations_new: usize,
    /// Trial register records written here.
    pub trial_records_new: usize,
}

/// Imports the sets `sets` (ids or unique prefixes) of the state root
/// `source` into `ctx`'s; see the module documentation.
///
/// # Errors
/// [`OrchestratorError`] when a set is not in the source (or a prefix names
/// several), an object does not match its content address, or a store fails.
/// Nothing of a set is half-imported in a way a rerun cannot finish: every
/// write is idempotent.
pub fn import_sets(
    ctx: &Context,
    source: &StateRoot,
    sets: &[String],
) -> Result<SetsImported, OrchestratorError> {
    let workspace = Workspace::at(source);
    let stored: Vec<_> = ExperienceStore::new(&workspace)
        .list_sets()?
        .into_iter()
        .map(|s| s.0)
        .collect();
    let ids = sets
        .iter()
        .map(|name| {
            Ok(SetId(ids::resolve(
                "experience set",
                name,
                stored.iter().cloned(),
            )?))
        })
        .collect::<Result<Vec<_>, OrchestratorError>>()?;
    copy(ctx.workspace(), &workspace, &ids, &[])
}

/// Copies the sets `sets` and the experiences `loose` (each with every
/// experience its relations name) of the database `source` into `dest`,
/// with their annotations and the trial register for what was copied; see
/// the module documentation. Copying again writes nothing.
///
/// # Errors
/// As [`import_sets`], a set or an experience `source` does not hold.
pub fn copy(
    dest: &Workspace,
    source: &Workspace,
    sets: &[SetId],
    loose: &[ExperienceId],
) -> Result<SetsImported, OrchestratorError> {
    let theirs = ExperienceStore::new(source);
    let ours = ExperienceStore::new(dest);
    let mut done = SetsImported::default();
    let mut imported: BTreeSet<ExperienceId> = BTreeSet::new();
    let mut members: BTreeSet<ExperienceId> = loose.iter().cloned().collect();
    // Each set with its members, then the loose experiences with no set.
    type Group<'a> = (Option<(&'a SetId, ExperienceSet)>, Vec<ExperienceId>);
    let mut groups: Vec<Group<'_>> = Vec::new();
    for id in sets {
        let set = theirs.get_set(id)?;
        members.extend(set.members.iter().cloned());
        let start = set.members.clone();
        groups.push((Some((id, set)), start));
    }
    groups.push((None, loose.to_vec()));
    for (set, start) in groups {
        let needed = closure(&theirs, start)?;
        for experience in &needed {
            if imported.contains(experience) {
                continue;
            }
            if ours.contains(experience)? {
                ours.get(experience)?;
                done.experiences_held += 1;
            } else {
                let record = theirs.get(experience)?;
                let written = ours.put(&record)?;
                if &written != experience {
                    return Err(OrchestratorError::Refused(format!(
                        "experience {experience} of the source was written here as {written}: \
                         the two builds do not agree about its content address"
                    )));
                }
                done.experiences_new += 1;
            }
            if !members.contains(experience) {
                done.related += 1;
            }
        }
        // Annotations once every experience a relation names is here.
        for experience in &needed {
            if !imported.insert(experience.clone()) {
                continue;
            }
            let before = ours.annotations(experience)?.annotations.len();
            for note in theirs.annotations(experience)?.annotations {
                ours.annotate(&note)?;
            }
            done.annotations_new += ours.annotations(experience)?.annotations.len() - before;
        }
        if let Some((id, set)) = set {
            let written = ours.put_set(&set)?;
            if &written != id {
                return Err(OrchestratorError::Refused(format!(
                    "experience set {id} of the source was written here as {written}: the two \
                     builds do not agree about its content address"
                )));
            }
            done.sets.push(written);
        }
    }
    let held = split::trials_in(dest)?;
    for record in split::trials_in(source)? {
        if imported.contains(&record.experience) && !held.contains(&record) {
            split::register_in(dest, &record)?;
            done.trial_records_new += 1;
        }
    }
    Ok(done)
}

/// `start` and, after them, every experience their relations name, until
/// nothing new is named; each once, in the order first reached.
fn closure(
    store: &ExperienceStore,
    start: Vec<ExperienceId>,
) -> Result<Vec<ExperienceId>, OrchestratorError> {
    let mut needed = Vec::new();
    let mut queue: VecDeque<ExperienceId> = start.into();
    let mut seen: BTreeSet<ExperienceId> = BTreeSet::new();
    while let Some(next) = queue.pop_front() {
        if !seen.insert(next.clone()) {
            continue;
        }
        for note in store.annotations(&next)?.annotations {
            if let AnnotationBody::Relation { other, .. } = &note.body {
                queue.push_back(other.clone());
            }
        }
        needed.push(next);
    }
    Ok(needed)
}
