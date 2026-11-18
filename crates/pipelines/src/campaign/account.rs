// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that run model learning as one
// resumable, multi-generation campaign with a single promotion gate and
// rollback, for its clients. If your team needs expertise in self-improving
// training loops you can procure our services by sending an email to
// info@swedishembedded.com.

//! What a campaign's ledger accounts for: the record each stage commits,
//! and what they come to read together (the generations done, the champion
//! and its history, what was spent, the pass rates), with the entry points
//! that read a campaign or roll it back without running it.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use splinter_eval::campaign::pass_rate::PassRates;
use splinter_eval::promotion::Promotion;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_store::experiences::SetId;
use splinter_store::stage_ledger::{Commit, StageLedger};

use super::champion::{self, Move};
use super::controller::{campaign_dir, parse_unit, record, unit, Stopped, CAMPAIGN_UNIT};
use super::spec::CampaignSpec;
use super::types::CampaignError;
use super::work::{Model, Trial};

/// What one generation came to.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GenerationReport {
    /// The generation.
    pub generation: u32,
    /// The candidate it trained; `None` when there was nothing to train on.
    pub candidate: Option<Model>,
    /// The gate's decision; `None` without a candidate.
    pub promotion: Option<Promotion>,
    /// What became of the champion.
    pub moved: Move,
}

/// What a campaign has done, read from its ledger.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CampaignReport {
    /// The campaign.
    pub campaign: String,
    /// Its seed.
    pub seed: u64,
    /// Every generation done.
    pub generations: Vec<GenerationReport>,
    /// The champion now.
    pub champion: Model,
    /// The champion's history, oldest first.
    pub history: Vec<Move>,
    /// What was spent.
    pub spent: Spent,
    /// Why it stopped; `None` for a report read without running.
    pub stopped: Option<Stopped>,
}

/// What the spending stages recorded, summed over every attempt that
/// committed.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Spent {
    /// Rollouts.
    pub rollouts: u64,
    /// Tokens the policy generated, where counted.
    pub tokens: u64,
    /// GPU-hours of training.
    pub gpu_hours: f64,
}

/// The rollout stage's record.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct RolloutRecord {
    pub(super) champion: Model,
    pub(super) tasks: Vec<String>,
    pub(super) trials: Vec<Trial>,
    pub(super) rollouts: u64,
    pub(super) tokens: Option<u64>,
}

/// The ingest stage's record.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct IngestRecord {
    pub(super) train_set: Option<SetId>,
    pub(super) experiences: usize,
    pub(super) refused: usize,
}

/// The curate stage's record.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct CurateRecord {
    /// The dataset's id: the digest of its manifest.
    pub(super) dataset: Option<String>,
    pub(super) records: usize,
    pub(super) sets: Vec<SetId>,
}

/// The train stage's record.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct TrainRecord {
    pub(super) candidate: Model,
    pub(super) resume_key: String,
    pub(super) steps: u32,
    pub(super) gpu_hours: f64,
}

/// The latest commit of `stage` of every generation up to `last`.
pub(super) fn latest_per_generation<'c>(
    commits: &'c [Commit],
    stage: &str,
    last: u32,
) -> Vec<&'c Commit> {
    let mut latest: BTreeMap<u32, &Commit> = BTreeMap::new();
    for commit in commits {
        if let Some((generation, s)) = parse_unit(&commit.unit) {
            if s == stage && generation <= last {
                latest.insert(generation, commit);
            }
        }
    }
    latest.into_values().collect()
}

/// The generation a campaign is at: one past the last whose champion move
/// is recorded.
pub(super) fn next_generation(commits: &[Commit]) -> u32 {
    commits
        .iter()
        .filter_map(|c| parse_unit(&c.unit))
        .filter(|(_, stage)| *stage == "promote")
        .map(|(generation, _)| generation)
        .max()
        .unwrap_or(0)
        + 1
}

/// The pass rate of every task over the rollouts of the generations before
/// `before`, each counted
/// under the generation that ran it.
pub(super) fn rates(
    spec: &CampaignSpec,
    commits: &[Commit],
    before: u32,
) -> Result<PassRates, CampaignError> {
    let mut rates = PassRates::new(spec.rollout.retire_at, spec.rollout.retire_after)
        .map_err(|e| CampaignError::Config(e.to_string()))?;
    for commit in latest_per_generation(commits, "rollout", before.saturating_sub(1)) {
        let generation = parse_unit(&commit.unit).map_or(0, |(g, _)| g);
        let rolled: RolloutRecord = record(commit)?;
        let mut by_task: BTreeMap<&str, Vec<Option<bool>>> = BTreeMap::new();
        for trial in &rolled.trials {
            by_task.entry(&trial.task).or_default().push(trial.passed);
        }
        for (task, outcomes) in by_task {
            rates.observe(task, u64::from(generation), &outcomes);
        }
    }
    Ok(rates)
}

/// What every spending stage recorded, summed.
pub(super) fn spent(commits: &[Commit]) -> Result<Spent, CampaignError> {
    let mut spent = Spent::default();
    for commit in commits {
        match parse_unit(&commit.unit) {
            Some((_, "rollout")) => {
                let rolled: RolloutRecord = record(commit)?;
                spent.rollouts += rolled.rollouts;
                spent.tokens += rolled.tokens.unwrap_or(0);
            }
            Some((_, "train")) => {
                let trained: TrainRecord = record(commit)?;
                spent.gpu_hours += trained.gpu_hours;
            }
            _ => {}
        }
    }
    Ok(spent)
}

/// The report of campaign `name` from its ledger's `commits`.
///
/// # Errors
/// A record of another build.
pub fn report(
    name: &str,
    seed: u64,
    base: &Model,
    commits: &[Commit],
) -> Result<CampaignReport, CampaignError> {
    let latest = |unit_name: &str| commits.iter().rev().find(|c| c.unit == unit_name);
    let mut generations = Vec::new();
    for commit in latest_per_generation(commits, "promote", u32::MAX) {
        let generation = parse_unit(&commit.unit).map_or(0, |(g, _)| g);
        generations.push(GenerationReport {
            generation,
            candidate: latest(&unit(generation, "train"))
                .map(record::<TrainRecord>)
                .transpose()?
                .map(|t| t.candidate),
            promotion: latest(&unit(generation, "gate"))
                .map(record::<Promotion>)
                .transpose()?,
            moved: record(commit)?,
        });
    }
    let (champion, history) = champion::history(base, commits)?;
    Ok(CampaignReport {
        campaign: name.to_string(),
        seed,
        generations,
        champion,
        history,
        spent: spent(commits)?,
        stopped: None,
    })
}

/// The ledger of the campaign `spec` names, read under its lock, with the
/// seed it began with; `None` for a campaign never begun.
pub(super) fn begun(
    ctx: &Context,
    spec: &CampaignSpec,
) -> Result<Option<(StageLedger, u64)>, CampaignError> {
    spec.validate()?;
    let dir = campaign_dir(ctx, &spec.name);
    if !dir
        .join(splinter_store::stage_ledger::LEDGER_FILE)
        .is_file()
    {
        return Ok(None);
    }
    let ledger = StageLedger::open(&dir)?;
    let seed = ledger
        .commits()?
        .iter()
        .find(|c| c.unit == CAMPAIGN_UNIT)
        .and_then(|c| c.summary["seed"].as_u64())
        .ok_or_else(|| CampaignError::Config(format!("campaign {} records no seed", spec.name)))?;
    Ok(Some((ledger, seed)))
}

pub(super) fn base_of(spec: &CampaignSpec) -> Result<Model, CampaignError> {
    Ok(Model {
        generation: 0,
        model: spec.base_ref()?.to_string(),
        candidate: None,
        adapter: None,
    })
}

/// What the campaign `spec` names has done, from its ledger; nothing runs.
///
/// # Errors
/// A campaign never begun, another process holding it, or its ledger.
pub fn campaign_status(
    ctx: &Context,
    spec: &CampaignSpec,
) -> Result<CampaignReport, CampaignError> {
    let (ledger, seed) = begun(ctx, spec)?
        .ok_or_else(|| CampaignError::Config(format!("campaign {} has not begun", spec.name)))?;
    report(&spec.name, seed, &base_of(spec)?, &ledger.commits()?)
}

/// Moves the champion of the campaign `spec` names back to the model
/// generation `to` made (0: the base), for `reason`; nothing else runs.
///
/// # Errors
/// A campaign never begun, another process holding it, and
/// [`CampaignController::rollback`]'s.
pub fn campaign_rollback(
    ctx: &Context,
    spec: &CampaignSpec,
    to: u32,
    reason: &str,
) -> Result<Move, CampaignError> {
    let (ledger, _) = begun(ctx, spec)?
        .ok_or_else(|| CampaignError::Config(format!("campaign {} has not begun", spec.name)))?;
    champion::rollback(&ledger, &base_of(spec)?, to, reason)
}

impl From<CampaignError> for OrchestratorError {
    fn from(e: CampaignError) -> Self {
        match e {
            CampaignError::Cancelled => Self::Cancelled,
            CampaignError::Orchestrator(inner) => *inner,
            CampaignError::Trainer(reason) => Self::Train(reason),
            other => Self::Refused(other.to_string()),
        }
    }
}
