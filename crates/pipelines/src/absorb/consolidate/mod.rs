// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning that keeps accumulating
// capabilities generation after generation, for its clients. If your team
// needs expertise in consolidating what a model learned into its own
// weights, you can procure our services by sending an email to
// info@swedishembedded.com.

//! `consolidate`: a generation taken into the weights.
//!
//! Within a generation claims accumulate and every update trains an adapter
//! again from the generation's base. When retraining from scratch each time
//! has become impractical, the generation is consolidated: a full checkpoint,
//! which is the base of the next generation, is made from the generation's
//! base and the claims of its release, and the next generation starts from an empty
//! claim set on top of it. The claims taken in become the next generation's
//! retention: they are never taught again, a bounded sample of their records
//! stays in the training mix, and every release must keep answering them.
//!
//! 1. **policy** - the alias, the release consolidated (`--from`, else the
//!    one the alias points at) and the generation it belongs to;
//! 2. **claims** - the live claims split into the ones the release's adapter
//!    is for, which are taken in, and the ones its base already holds; when
//!    the base holds a claim that is no longer live, the weights cannot give
//!    it back and the plan starts again from the last base that did not;
//! 3. **consolidate** - `full`: every weight of the base trained on the
//!    claims' records beside a sample of the retained claims' records and the
//!    base's own answers; `merge`: the adapter folded into the base's
//!    weights, nothing trained ([`job`]); then the checkpoint is gated
//!    against the release in use ([`gate`]) and released as a full
//!    checkpoint, the base of the generation it opens. A checkpoint the gate
//!    refuses is not kept and the current release stays in use.

pub mod gate;
pub mod job;
pub mod plan;

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::Serialize;
use splinter_core::claim::ClaimId;
use splinter_core::model_ref::ModelRef;
use splinter_core::release::ReleaseId;
use splinter_eval::gate::GateConfig;
use splinter_knowledge::claims::Ledger;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_orchestrator::generation::ConsolidationMode;
use splinter_orchestrator::ids;
use splinter_orchestrator::pipeline::{FnStage, Pipeline, StageEnd};
use splinter_orchestrator::releases::StoredRelease;
use splinter_orchestrator::runs::{record, to_json, Recorded, Recorder};

pub use job::{Consolidated, Params};

use super::generation::{
    Generation, DEFAULT_CONSOLIDATE_AFTER_RECORDS, DEFAULT_RETENTION_REPLAY_SHARE,
};
use super::kit::{self, ClaimKit, Live};
use super::sealed::SealedProbes;
use crate::learn::PolicyUsed;
use crate::rehearsal::DEFAULT_REHEARSAL_SHARE;
use crate::train::{Trainer, Tuning};

/// The most passes over the records a full consolidation makes.
pub const DEFAULT_CONSOLIDATION_EPOCHS: u32 = 6;

/// The stages, in order, as runs and reports name them.
pub const STAGES: [&str; 3] = ["policy", "claims", "consolidate"];

/// When `absorb` takes a generation into the weights, and how.
#[derive(Clone, Debug, Serialize)]
pub struct ConsolidationPolicy {
    /// Consolidate once the generation's claims hold more trained records
    /// than this; `None` never consolidates.
    pub after_records: Option<usize>,
    /// Train every weight, or fold the adapter in.
    pub mode: ConsolidationMode,
    /// The most passes over the records.
    pub epochs: u32,
    /// The step budget, in place of `epochs`.
    pub steps: Option<u32>,
    /// The learning rate; [`crate::train::DEFAULT_FULL_LEARNING_RATE`] when
    /// `None`.
    pub learning_rate: Option<f32>,
    /// The share of the training draws that are records of the claims the
    /// base already holds, in a night's adapter and in a full consolidation.
    pub retention_replay_share: f64,
}

impl Default for ConsolidationPolicy {
    fn default() -> Self {
        Self {
            after_records: Some(DEFAULT_CONSOLIDATE_AFTER_RECORDS),
            mode: ConsolidationMode::Full,
            epochs: DEFAULT_CONSOLIDATION_EPOCHS,
            steps: None,
            learning_rate: None,
            retention_replay_share: DEFAULT_RETENTION_REPLAY_SHARE,
        }
    }
}

/// One `consolidate`.
#[derive(Clone, Debug, Serialize)]
pub struct ConsolidateRequest {
    /// The policy: `policy:<alias>`; the release is made under the alias.
    pub policy: ModelRef,
    /// Train every weight, or fold the adapter in.
    pub mode: ConsolidationMode,
    /// The release to consolidate, by id or unique prefix; the one the alias
    /// points at when `None`.
    pub from: Option<String>,
    /// The share of the training draws that are the base's own answers.
    pub rehearsal_share: f64,
    /// The share of the training draws that are records of the claims the
    /// base already holds.
    pub retention_replay_share: f64,
    /// The most passes over the records.
    pub epochs: u32,
    /// The step budget, in place of `epochs`.
    pub steps: Option<u32>,
    /// The learning rate and how the run is watched.
    pub tuning: Tuning,
    /// Files of sealed probes no training record may contain.
    pub sealed_probes: Vec<PathBuf>,
    /// Stop at the checkpoint: do not release it.
    pub no_release: bool,
    /// The serve check's bounds.
    pub gate: GateConfig,
}

impl Default for ConsolidateRequest {
    fn default() -> Self {
        Self {
            policy: ModelRef::policy_default(),
            mode: ConsolidationMode::Full,
            from: None,
            rehearsal_share: DEFAULT_REHEARSAL_SHARE,
            retention_replay_share: DEFAULT_RETENTION_REPLAY_SHARE,
            epochs: DEFAULT_CONSOLIDATION_EPOCHS,
            steps: None,
            tuning: Tuning::default(),
            sealed_probes: Vec::new(),
            no_release: false,
            gate: GateConfig::default(),
        }
    }
}

impl ConsolidateRequest {
    /// The parameters the engine carries a consolidation out by.
    #[must_use]
    pub fn params(&self) -> Params {
        Params {
            mode: self.mode,
            rehearsal_share: self.rehearsal_share,
            retention_replay_share: self.retention_replay_share,
            epochs: self.epochs,
            steps: self.steps,
            tuning: self.tuning,
            gate: self.gate,
            no_release: self.no_release,
        }
    }
}

/// What a `consolidate` reports.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Consolidation {
    /// The policy the run worked from.
    pub policy: PolicyUsed,
    /// The generation of the release consolidated: its epoch and base.
    pub generation: Option<Generation>,
    /// Live claims after the rulings.
    pub live: usize,
    /// The claims taken into the weights.
    pub closing: Vec<ClaimId>,
    /// The claims the base already held.
    pub retained: Vec<ClaimId>,
    /// The consolidation itself, once it ran.
    pub consolidated: Option<Consolidated>,
    /// Why the run ended before a release, if it did.
    pub stopped: Option<String>,
}

impl Consolidation {
    /// The release made, when the gate passed.
    #[must_use]
    pub fn release(&self) -> Option<&ReleaseId> {
        self.consolidated.as_ref()?.release.as_ref()
    }

    /// Whether the run did what was asked: a release, or the checkpoint
    /// when no release was asked for.
    #[must_use]
    pub fn finished(&self, release_asked: bool) -> bool {
        self.consolidated
            .as_ref()
            .is_some_and(|c| c.gate.as_ref().is_some_and(|g| g.passed))
            && (!release_asked || self.release().is_some())
    }
}

struct State<'a> {
    request: &'a ConsolidateRequest,
    trainer: &'a dyn Trainer,
    alias: String,
    sealed: SealedProbes,
    system: Option<String>,
    current: Option<ReleaseId>,
    release: Option<StoredRelease>,
    live: Vec<Live>,
    kits: BTreeMap<ClaimId, ClaimKit>,
    plan: Option<plan::Plan>,
    report: Consolidation,
}

/// Consolidates the generation of `request.policy`, training with `trainer`.
pub fn consolidate(
    ctx: &Context,
    request: &ConsolidateRequest,
    trainer: &dyn Trainer,
) -> Result<Recorded<Consolidation>, OrchestratorError> {
    let ModelRef::Policy(alias) = &request.policy else {
        return Err(OrchestratorError::Refused(format!(
            "{} is not a policy: a release is made under an alias, so name `policy:<alias>`",
            request.policy
        )));
    };
    if !(0.0..1.0).contains(&request.rehearsal_share)
        || !(0.0..1.0).contains(&request.retention_replay_share)
    {
        return Err(OrchestratorError::Refused(
            "the rehearsal and retention replay shares are in [0, 1)".into(),
        ));
    }
    if request.epochs == 0 {
        return Err(OrchestratorError::Refused("epochs are at least one".into()));
    }
    record(ctx, "consolidate", request, |run| {
        let mut state = State {
            request,
            trainer,
            alias: alias.clone(),
            sealed: SealedProbes::load(&request.sealed_probes)?,
            system: None,
            current: None,
            release: None,
            live: Vec::new(),
            kits: BTreeMap::new(),
            plan: None,
            report: Consolidation::default(),
        };
        let stopped = pipeline().run(ctx, run, &mut state, None)?;
        let mut report = state.report;
        report.stopped = stopped.or(report.stopped);
        Ok(report)
    })
}

fn pipeline<'a>() -> Pipeline<'a, State<'a>> {
    Pipeline::new()
        .then(FnStage::new("policy", policy_stage).ignoring_budget())
        .then(FnStage::new("claims", claims_stage).ignoring_budget())
        .then(FnStage::new("consolidate", consolidate_stage).ignoring_budget())
}

type Done = Result<StageEnd, OrchestratorError>;

fn value(summary: &impl Serialize) -> Result<serde_json::Value, OrchestratorError> {
    to_json("the stage summary", summary)
}

fn policy_stage(ctx: &Context, _: &mut Recorder<'_>, st: &mut State<'_>) -> Done {
    let pin = ctx.policy_pin(&st.alias)?;
    let Some(pin) = pin else {
        return Err(OrchestratorError::NotFound {
            what: "release alias",
            id: st.alias.clone(),
        });
    };
    st.system = pin.system_prompt.clone();
    st.current = Some(pin.release.clone());
    st.report.policy = PolicyUsed {
        alias: st.alias.clone(),
        release: Some(pin.release.clone()),
    };
    let store = ctx.releases();
    let id = match &st.request.from {
        Some(given) => ReleaseId(ids::resolve(
            "release",
            given,
            store.list()?.into_iter().map(|r| r.0),
        )?),
        None => pin.release,
    };
    let release = store.get(&id)?;
    if release.is_consolidation() {
        return Ok(StageEnd::stop(
            value(&st.report.policy)?,
            format!(
                "release {id} is a consolidation: its generation has no adapter to take in; \
                 absorb new sessions first"
            ),
        ));
    }
    let generation = Generation::of_release(ctx, &release)?;
    st.report.generation = Some(generation);
    st.release = Some(release);
    Ok(StageEnd::done(value(&st.report.policy)?))
}

fn claims_stage(ctx: &Context, _: &mut Recorder<'_>, st: &mut State<'_>) -> Done {
    let Some(release) = &st.release else {
        unreachable!("the claims stage follows a policy stage that found a release")
    };
    let ledger = Ledger::new(ctx.claims().entries()?);
    st.live = ledger
        .live()
        .map_err(|e| OrchestratorError::Refused(e.to_string()))?
        .into_iter()
        .map(|(id, claim)| (id, claim.clone()))
        .collect();
    st.report.live = st.live.len();
    st.kits = kit::kept(ctx)?;
    let closing = plan::own_claims(ctx, release, &st.live)?;
    let planned = plan::plan(ctx, release, &st.live, closing)?;
    if planned.rebuilt && st.request.mode == ConsolidationMode::Merge {
        return Err(OrchestratorError::Refused(format!(
            "the base of release {} holds {} claim(s) that are no longer live; taking them out \
             trains every weight again from the last base that did not hold them, which a merge \
             does not: consolidate with --mode full",
            release.id,
            planned.dead.len()
        )));
    }
    st.report.closing = planned.train.iter().map(|(id, _)| id.clone()).collect();
    st.report.retained = planned.retention.iter().map(|(id, _)| id.clone()).collect();
    let summary = value(&(&st.report.closing, &st.report.retained, &planned.dead))?;
    let nothing = planned.train.is_empty();
    st.plan = Some(planned);
    Ok(if nothing {
        StageEnd::stop(
            summary,
            "the release's generation holds no live claim to take into the weights",
        )
    } else {
        StageEnd::done(summary)
    })
}

fn consolidate_stage(ctx: &Context, run: &mut Recorder<'_>, st: &mut State<'_>) -> Done {
    let (Some(release), Some(planned), Some(current)) = (&st.release, &st.plan, &st.current) else {
        unreachable!("the consolidate stage follows the claims stage")
    };
    let params = st.request.params();
    let done = job::run(
        ctx,
        &job::Job {
            alias: &st.alias,
            champion: release,
            replaces: current,
            from: &planned.from,
            train: &planned.train,
            retention: &planned.retention,
            rebuilt: planned.rebuilt,
            dead: &planned.dead,
            kits: &st.kits,
            system: st.system.as_deref(),
            sealed: &st.sealed,
            params: &params,
            cancel: run.cancel_token(),
        },
        st.trainer,
    )?;
    let summary = value(&done)?;
    let stopped = done.stopped.clone();
    st.report.consolidated = Some(done);
    Ok(match stopped {
        Some(why) => StageEnd::stop(summary, why),
        None => StageEnd::done(summary),
    })
}

#[cfg(test)]
mod tests {
    /// The list a report names is the pipeline that runs.
    #[test]
    fn the_stage_list_is_the_pipeline() {
        assert_eq!(super::pipeline().names(), super::STAGES);
    }
}
