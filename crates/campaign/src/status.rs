// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `status`: the policy in use, the most recent runs, what the stores
//! hold, and the concepts the policy has mastered least.

use std::path::PathBuf;

use serde::Serialize;
use splinter_policy::ModelSelection;
use splinter_record::runs::list_runs;

use crate::context::Context;
use crate::curriculum::mastery::{weakest, MasteryReport, DEFAULT_WEAKEST};
use crate::error::CampaignError;
use crate::model_ref::{ModelRef, POLICY_DEFAULT};
use crate::release::ReleaseId;
use crate::runs::{RunSummary, RECENT_RUNS};
use crate::train::candidate_count;

/// The policy in use.
#[derive(Clone, Debug, Serialize)]
pub struct PolicyStatus {
    /// The reference commands use for it: `policy:default`.
    pub reference: String,
    /// The identity records give it.
    pub model: String,
    /// Its base checkpoint.
    pub base: PathBuf,
    /// The adapter it serves, if any: `null` until releases exist.
    pub adapter: Option<PathBuf>,
    /// The release `policy:default` points at: `null` until one exists.
    pub release: Option<ReleaseId>,
}

/// How much each store holds.
#[derive(Clone, Debug, Serialize)]
pub struct Counts {
    /// Stored sources.
    pub sources: usize,
    /// Stored task sets.
    pub task_sets: usize,
    /// Stored tasks.
    pub tasks: usize,
    /// Stored experiences.
    pub experiences: usize,
    /// Stored experience sets.
    pub experience_sets: usize,
    /// Stored datasets.
    pub datasets: usize,
    /// Trained candidates.
    pub candidates: usize,
}

/// What `status` reports.
#[derive(Clone, Debug, Serialize)]
pub struct Status {
    /// The state root.
    pub state: PathBuf,
    /// The policy in use.
    pub policy: PolicyStatus,
    /// The most recent runs, oldest first.
    pub recent_runs: Vec<RunSummary>,
    /// What the stores hold.
    pub counts: Counts,
    /// The weakest concepts under the release the policy is now.
    pub concepts: MasteryReport,
}

/// The status of the state root `ctx` works in.
pub fn status(ctx: &Context) -> Result<Status, CampaignError> {
    let reference = ModelRef::policy_default();
    let selection = ctx.selection(&reference)?;
    let (base, adapter) = match &selection {
        ModelSelection::Local(weights) => (weights.base.clone(), weights.adapter.clone()),
        ModelSelection::Remote(_) => (PathBuf::new(), None),
    };
    let runs = list_runs(ctx.root())?;
    let recent_runs = runs
        .iter()
        .skip(runs.len().saturating_sub(RECENT_RUNS))
        .map(RunSummary::from)
        .collect();
    Ok(Status {
        state: ctx.root().path().to_path_buf(),
        policy: PolicyStatus {
            reference: reference.to_string(),
            model: selection.identity(),
            base,
            adapter,
            release: ctx.policy_pin(POLICY_DEFAULT)?.map(|pin| pin.release),
        },
        recent_runs,
        counts: Counts {
            sources: ctx.sources().list()?.len(),
            task_sets: ctx.tasks().list_sets()?.len(),
            tasks: ctx.tasks().list()?.len(),
            experiences: ctx.experiences().list()?.len(),
            experience_sets: ctx.experiences().list_sets()?.len(),
            datasets: ctx.datasets().list()?.len(),
            candidates: candidate_count(ctx)?,
        },
        concepts: weakest(ctx, DEFAULT_WEAKEST)?,
    })
}
