// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements experience databases that index what many
// learning experiments did, for its clients. If your team needs expertise in
// learning from agent experience across experiments, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The experience database as an optional index over learning-factory runs.
//!
//! [`index_run`] reads a run's committed rollout products and files every
//! trial that left a trajectory as an experience
//! ([`splinter_factory::run::corpus`], the conversion a dataset stage reads
//! its corpus by), with the factory's grade as an executable verdict, and
//! one experience set per pass. A trial whose grade is not a
//! verdict (the infrastructure decided it) is filed with no verdict, never a
//! fail. Indexing is idempotent: experiences and verdicts are content
//! addressed, so indexing a run again files nothing new. The index never
//! decides anything about the run; the run's own products stay the record.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Serialize;
use splinter_core::clock::SystemClock;
use splinter_factory::run::corpus::{experience_of, outcome, verdict_of};
use splinter_factory::run::rollout::trials_of;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_store::experiences::{ExperienceSet, SetId};
use splinter_store::factory::products::ProductStore;

/// The producer of the verdicts an index files.
pub use splinter_factory::run::corpus::PRODUCER;

/// What indexing one pass filed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct IndexedPass {
    /// The pass's product view (`rollouts/<pass>`, `eval/_arms/<arm>`).
    pub pass: String,
    /// Its experience set; `None` when no trial left a trajectory.
    pub experience_set: Option<SetId>,
    /// Experiences filed, by verdict (`pass`, `fail`, `abstain`).
    pub verdicts: BTreeMap<String, usize>,
    /// Trials left out, each with why.
    pub skipped: Vec<(String, String)>,
}

fn refused(why: impl ToString) -> OrchestratorError {
    OrchestratorError::Refused(why.to_string())
}

/// Index every committed rollout product of the factory run in `run`.
///
/// # Errors
/// The run's products cannot be read, or the store refuses a write.
pub fn index_run(ctx: &Context, run: &Path) -> Result<Vec<IndexedPass>, OrchestratorError> {
    splinter_factory::run::require_run(run).map_err(refused)?;
    let products = ProductStore::new(run, Box::new(SystemClock));
    let store = ctx.experiences();
    let mut out = Vec::new();
    for manifest in products.list().map_err(refused)? {
        let is_rollout =
            manifest.name.starts_with("rollouts/") || manifest.name.starts_with("eval/_arms/");
        if !is_rollout {
            continue;
        }
        let view = run.join(&manifest.name);
        let trials = trials_of(&view).map_err(refused)?;
        let (mut members, mut verdicts, mut skipped) = (Vec::new(), BTreeMap::new(), Vec::new());
        for trial in &trials {
            let experience = match experience_of(&view, trial, None) {
                Ok(experience) => experience,
                Err(why) => {
                    skipped.push((trial.slot.to_string(), why));
                    continue;
                }
            };
            let id = store.put(&experience)?;
            let verdict = outcome(trial);
            store.annotate(&verdict_of(&id, trial))?;
            *verdicts
                .entry(format!("{verdict:?}").to_lowercase())
                .or_default() += 1;
            if !members.contains(&id) {
                members.push(id);
            }
        }
        let experience_set = if members.is_empty() {
            None
        } else {
            Some(store.put_set(&ExperienceSet {
                name: format!("factory trials of {}", manifest.name),
                members,
            })?)
        };
        out.push(IndexedPass {
            pass: manifest.name.clone(),
            experience_set,
            verdicts,
            skipped,
        });
    }
    Ok(out)
}
