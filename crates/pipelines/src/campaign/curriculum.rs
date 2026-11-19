// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements curricula that spend a learner's practice
// where its measured capability gaps are, for its clients. If your team needs
// expertise in adaptive curricula for agent training, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Which practice tasks a generation rolls out.
//!
//! Under `curriculum = "all"` (the default) it is every practice task the
//! champion has not mastered. Under `curriculum = "capability"` the rollouts
//! of the generations before it are counted per capability of the
//! examiner's taxonomy ([`crate::curriculum::capability`]): the tasks are
//! drawn by their capabilities' weight, from a seed of the campaign's, never
//! a task the infrastructure stopped every time, and (with `transfer`)
//! fresh exercises are requested for the weakest capabilities. What was
//! decided is recorded with the rollout.

use std::collections::BTreeMap;

use splinter_core::rng::Seed;
use splinter_eval::campaign::pass_rate::{Band, PassRates};
use splinter_store::stage_ledger::Commit;

use super::account::{latest_per_generation, RolloutRecord};
use super::controller::record;
use super::spec::{CampaignSpec, RolloutCurriculum};
use super::types::CampaignError;
use super::work::Rollouts;
use crate::curriculum::capability::{competence, select, Attempt, SelectRequest, Selection};

/// What a generation rolls out.
pub(super) struct Planned {
    /// The practice tasks, by name.
    pub(super) tasks: Vec<String>,
    /// The capability curriculum's decision; `None` under `all`.
    pub(super) selection: Option<Selection>,
}

/// Every attempt of the rollouts before `generation`, each labelled by its
/// trial's capabilities, else by the material's labels of its task.
fn history(
    commits: &[Commit],
    generation: u32,
    labels: &BTreeMap<String, Vec<String>>,
) -> Result<Vec<Attempt>, CampaignError> {
    let mut attempts = Vec::new();
    for commit in latest_per_generation(commits, "rollout", generation.saturating_sub(1)) {
        let rolled: RolloutRecord = record(commit)?;
        for trial in rolled.trials {
            let capabilities = if trial.capabilities.is_empty() {
                labels.get(&trial.task).cloned().unwrap_or_default()
            } else {
                trial.capabilities
            };
            attempts.push(Attempt {
                task: trial.task,
                capabilities,
                passed: trial.passed,
                class: trial.class,
            });
        }
    }
    Ok(attempts)
}

/// The tasks generation `generation` rolls out, given the pass `rates` of
/// the rollouts before it; `seed` draws them under the capability
/// curriculum.
pub(super) fn plan(
    spec: &CampaignSpec,
    commits: &[Commit],
    rates: &PassRates,
    generation: u32,
    rollouts: &dyn Rollouts,
    seed: u64,
) -> Result<Planned, CampaignError> {
    let pool = rollouts.tasks()?;
    if spec.rollout.curriculum == RolloutCurriculum::All {
        return Ok(Planned {
            tasks: pool
                .into_iter()
                .filter(|task| rates.band(task) != Band::Retired)
                .collect(),
            selection: None,
        });
    }
    let labels = rollouts.capabilities()?;
    let attempts = history(commits, generation, &labels)?;
    let measured = competence(&attempts, rates);
    let selection = select(&SelectRequest {
        pool: &pool,
        labels: &labels,
        rates,
        competence: &measured,
        budget: spec.rollout.tasks,
        transfer: spec.rollout.transfer.map_or(0, |t| t.capabilities),
        seed: Seed::new(seed),
    });
    Ok(Planned {
        tasks: selection.tasks.clone(),
        selection: Some(selection),
    })
}
