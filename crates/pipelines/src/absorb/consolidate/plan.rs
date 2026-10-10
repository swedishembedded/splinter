// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning that keeps accumulating
// capabilities generation after generation, for its clients. If your team
// needs expertise in consolidating what a model learned into its own
// weights, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Which claims a consolidation takes into which base.
//!
//! Ordinarily the base is the one the release in use sits on, the claims
//! taken in are the ones the release was trained on, and the claims the base
//! already holds are kept. When the base holds a claim that is no longer live
//! the weights have absorbed it and cannot give it back: the plan then starts
//! from the last base that did not hold it and trains the claims consolidated
//! since, that are still live, in one checkpoint ([`plan_rebuild`]).

use std::collections::BTreeSet;

use splinter_core::claim::ClaimId;
use splinter_core::release::ReleaseId;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_orchestrator::releases::StoredRelease;

use crate::absorb::generation::{history, plan_rebuild, Generation};
use crate::absorb::kit::Live;

/// What a consolidation takes into what.
#[derive(Clone, Debug)]
pub(crate) struct Plan {
    /// The generation whose base is trained from or merged into.
    pub(crate) from: Generation,
    /// The claims taken into the weights.
    pub(crate) train: Vec<Live>,
    /// The claims the base already holds.
    pub(crate) retention: Vec<Live>,
    /// The claims the weights of the release in use hold that are no longer
    /// live.
    pub(crate) dead: Vec<ClaimId>,
    /// Whether the plan trains again from an earlier base to take them out.
    pub(crate) rebuilt: bool,
}

/// The claims `release` was trained on that are live and that its base does
/// not hold: the ones its adapter is for.
pub(crate) fn own_claims(
    ctx: &Context,
    release: &StoredRelease,
    live: &[Live],
) -> Result<Vec<Live>, OrchestratorError> {
    let generation = Generation::of_release(ctx, release)?;
    let trained: BTreeSet<ClaimId> = ctx
        .claims()
        .trained_on(&release.id)?
        .unwrap_or_default()
        .into_iter()
        .collect();
    Ok(generation
        .split(live)
        .current
        .into_iter()
        .filter(|(id, _)| trained.contains(id))
        .collect())
}

/// The plan for consolidating `release`, whose own claims are `closing`
/// (more claims can be named in `extra`, taken into the weights beside
/// them). When the base holds a dead claim the plan is a rebuild, and
/// `rebuild_allowed` says whether the caller can run one.
pub(crate) fn plan(
    ctx: &Context,
    release: &StoredRelease,
    live: &[Live],
    extra: Vec<Live>,
) -> Result<Plan, OrchestratorError> {
    let generation = Generation::of_release(ctx, release)?;
    let dead = generation.dead(live);
    if dead.is_empty() {
        let split = generation.split(live);
        return Ok(Plan {
            from: generation,
            train: extra,
            retention: split.retention,
            dead,
            rebuilt: false,
        });
    }
    let links = history(ctx, generation.base_release.as_ref())?;
    let live_ids: BTreeSet<ClaimId> = live.iter().map(|(id, _)| id.clone()).collect();
    let Some(rebuild) = plan_rebuild(&links, &dead, &live_ids) else {
        return Err(OrchestratorError::Refused(format!(
            "release {} holds claims that are no longer live, but no consolidation behind it \
             took them in: the history of its base is incomplete",
            release.id
        )));
    };
    let from = generation_of(ctx, rebuild.from.as_ref())?;
    let mut train: Vec<Live> = live
        .iter()
        .filter(|(id, _)| rebuild.retrain.contains(id))
        .cloned()
        .collect();
    for claim in extra {
        if !train.iter().any(|(id, _)| *id == claim.0) {
            train.push(claim);
        }
    }
    let retention = from.split(live).retention;
    Ok(Plan {
        from,
        train,
        retention,
        dead,
        rebuilt: true,
    })
}

/// The generation whose base is the consolidation `base`; the configured
/// base for `None`.
fn generation_of(ctx: &Context, base: Option<&ReleaseId>) -> Result<Generation, OrchestratorError> {
    match base {
        Some(id) => Generation::of_release(ctx, &ctx.releases().get(id)?),
        None => Ok(Generation::root(ctx)),
    }
}
