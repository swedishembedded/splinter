// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning that keeps accumulating
// capabilities generation after generation, for its clients. If your team
// needs expertise in continual learning that consolidates into the weights,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! The bookkeeping of generations.
//!
//! A policy learns in generations. Within one, claims accumulate and every
//! update trains an adapter again from the generation's base on the
//! generation's live claims. A **consolidation** takes the generation into
//! the weights: the next generation's base is a full checkpoint, its claim
//! set starts empty, and the claims the base holds - its **retention** - are
//! never taught again but must keep passing: they are graded by their
//! stopping paraphrases at every release, and a bounded, deterministic
//! sample of their records stays in the training mix so the adapter on top
//! does not pull the base away from them.
//!
//! What a base holds is the claim ledger's trained set of the consolidation
//! release ([`splinter_store::claims::ClaimStore::trained_on`]); there is no
//! second list. Forgetting a claim the base holds is not a deletion: the
//! weights have absorbed it, and the only honest removal is training again
//! from the last base that did not hold it ([`plan_rebuild`]).

use std::collections::BTreeSet;
use std::path::PathBuf;

use serde::Serialize;
use splinter_core::claim::ClaimId;
use splinter_core::digest::Digest;
use splinter_core::release::ReleaseId;
use splinter_data::Record;

use super::dataset::Replayed;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_orchestrator::generation::GenerationMark;
use splinter_orchestrator::releases::StoredRelease;

use super::kit::{ClaimKit, Live};
use crate::learn::records_at_share;
use std::collections::BTreeMap;

/// The fewest live-claim records that make retraining the adapter from the
/// base each night impractical, and so a consolidation due: above this many
/// trained records in the generation's claims, the next update consolidates.
pub const DEFAULT_CONSOLIDATE_AFTER_RECORDS: usize = 2000;

/// The share of the training draws a sample of the retained claims' records
/// takes when a generation has retention to keep: as large as the replay of
/// earlier releases, so the generation's own claims keep most of the steps.
pub const DEFAULT_RETENTION_REPLAY_SHARE: f64 = 0.25;

/// The generation a policy alias is in.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Generation {
    /// Consolidations behind the alias's release: 0 on the configured base.
    pub epoch: u32,
    /// The consolidation whose checkpoint is the base; `None` is the
    /// configured base.
    pub base_release: Option<ReleaseId>,
    /// The base directory this generation's adapters are trained on.
    pub base: PathBuf,
    /// The claims the base holds, as its consolidation was trained on them.
    pub held: Vec<ClaimId>,
}

impl Generation {
    /// The generation `alias` is in: that of the release it points at, or
    /// generation 0 on the configured base before any release.
    pub fn of(ctx: &Context, alias: &str) -> Result<Self, OrchestratorError> {
        let Some(pin) = ctx.policy_pin(alias)? else {
            return Ok(Self::root(ctx));
        };
        let stored = ctx.releases().get(&pin.release)?;
        Self::of_release(ctx, &stored)
    }

    /// Generation 0: the configured base, holding no claim.
    #[must_use]
    pub fn root(ctx: &Context) -> Self {
        Self {
            epoch: 0,
            base_release: None,
            base: ctx.config().policy_base.clone(),
            held: Vec::new(),
        }
    }

    /// The generation `release` belongs to.
    pub fn of_release(ctx: &Context, release: &StoredRelease) -> Result<Self, OrchestratorError> {
        let base_release = release.base_release();
        let held = match &base_release {
            Some(base) => ctx.claims().trained_on(base)?.unwrap_or_default(),
            None => Vec::new(),
        };
        let base = match &base_release {
            Some(base) => ctx.generation_base(base)?,
            None => ctx.config().policy_base.clone(),
        };
        Ok(Self {
            epoch: release.epoch(),
            base_release,
            base,
            held,
        })
    }

    /// The generation mark of an adapter released in this generation.
    #[must_use]
    pub fn mark(&self) -> GenerationMark {
        GenerationMark {
            epoch: self.epoch,
            base: self.base_release.clone(),
            consolidation: None,
        }
    }

    /// The live claims split into the generation's own - those the base does
    /// not hold, which the adapter is trained on - and the retention, which
    /// the base holds and the gate keeps asking about.
    #[must_use]
    pub fn split(&self, live: &[Live]) -> Split {
        let held: BTreeSet<&ClaimId> = self.held.iter().collect();
        let (retention, current) = live.iter().cloned().partition(|(id, _)| held.contains(id));
        Split { current, retention }
    }

    /// The claims the base holds that are no longer live: forgotten or
    /// superseded since the base was trained, still in its weights.
    #[must_use]
    pub fn dead(&self, live: &[Live]) -> Vec<ClaimId> {
        let live: BTreeSet<&ClaimId> = live.iter().map(|(id, _)| id).collect();
        self.held
            .iter()
            .filter(|id| !live.contains(id))
            .cloned()
            .collect()
    }
}

/// Live claims split by whether the generation's base holds them.
#[derive(Clone, Debug, Default)]
pub struct Split {
    /// The generation's own claims.
    pub current: Vec<Live>,
    /// The claims the base holds.
    pub retention: Vec<Live>,
}

/// The trained records of `claims`, by the kits that hold them.
#[must_use]
pub fn records_of(kits: &BTreeMap<ClaimId, ClaimKit>, claims: &[Live]) -> usize {
    claims
        .iter()
        .filter_map(|(id, _)| kits.get(id))
        .map(|kit| kit.records.len())
        .sum()
}

/// Whether a generation whose claims hold `records` trained records is due
/// to be consolidated at `threshold`: strictly above it.
#[must_use]
pub fn consolidation_due(records: usize, threshold: usize) -> bool {
    records > threshold
}

/// How many retained records to replay beside `current` records so that the
/// replay takes `share` of the draws, never more than the `available`.
#[must_use]
pub fn replay_budget(current: usize, available: usize, share: f64) -> usize {
    if current == 0 {
        return 0;
    }
    records_at_share(current, share).min(available)
}

/// A bounded sample of the retained claims' records: `budget` of them, taken
/// in turn from each claim so that every claim is replayed before any is
/// replayed twice, each claim's records in the stable order of their
/// content. The same kits and budget give the same sample.
#[must_use]
pub fn retention_sample(
    kits: &BTreeMap<ClaimId, ClaimKit>,
    retention: &[Live],
    budget: usize,
) -> Vec<Replayed> {
    let mut queues: Vec<(&Live, Vec<&Record>)> = retention
        .iter()
        .filter_map(|live| kits.get(&live.0).map(|kit| (live, kit)))
        .map(|(live, kit)| {
            let mut records: Vec<&Record> = kit.records.iter().collect();
            records.sort_by_cached_key(|r| Digest::of(&serde_json::to_vec(r).unwrap_or_default()));
            (live, records)
        })
        .collect();
    let mut sample = Vec::new();
    let mut round = 0;
    while sample.len() < budget {
        let mut took = false;
        for ((id, claim), queue) in &mut queues {
            if let Some(record) = queue.get(round) {
                sample.push(Replayed {
                    claim: id.clone(),
                    statement: claim.statement.clone(),
                    record: (*record).clone(),
                });
                took = true;
                if sample.len() == budget {
                    break;
                }
            }
        }
        if !took {
            break;
        }
        round += 1;
    }
    sample
}

/// One consolidation of a policy's history: the checkpoint it made, the base
/// it was trained from and the claims it closed with.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Link {
    /// The consolidation release.
    pub release: ReleaseId,
    /// The consolidation it was trained from; `None` is the configured base.
    pub from: Option<ReleaseId>,
    /// The claims it took in.
    pub closing: Vec<ClaimId>,
}

/// The consolidations behind `base`, oldest first, ending with `base`
/// itself; empty for the configured base.
pub fn history(ctx: &Context, base: Option<&ReleaseId>) -> Result<Vec<Link>, OrchestratorError> {
    let mut links = Vec::new();
    let mut next = base.cloned();
    while let Some(id) = next {
        if links.iter().any(|l: &Link| l.release == id) {
            return Err(OrchestratorError::Refused(format!(
                "the consolidations behind release {id} do not end; the release store is corrupt"
            )));
        }
        let stored = ctx.releases().get(&id)?;
        let mark = stored.manifest.generation.as_ref();
        let closing = mark
            .and_then(|m| m.consolidation.as_ref())
            .map(|c| c.closing.clone())
            .unwrap_or_default();
        next = mark.and_then(|m| m.base.clone());
        links.push(Link {
            release: id,
            from: next.clone(),
            closing,
        });
    }
    links.reverse();
    Ok(links)
}

/// How to take claims out of weights that hold them: train again from the
/// last base that did not.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Rebuild {
    /// The last base that held none of the dead claims; `None` is the
    /// configured base.
    pub from: Option<ReleaseId>,
    /// The dead claims the weights hold.
    pub dead: Vec<ClaimId>,
    /// The claims the consolidations after `from` took in that are still
    /// live: what the new checkpoint is trained on.
    pub retrain: Vec<ClaimId>,
}

/// The rebuild that takes `dead` out of the consolidations in `links`
/// (oldest first), or `None` when no consolidation holds any of them. The
/// earliest consolidation that took one in decides: its base is the last that
/// did not hold it, and every claim consolidated from there on that is still
/// `live` is trained again, in one checkpoint.
#[must_use]
pub fn plan_rebuild(links: &[Link], dead: &[ClaimId], live: &BTreeSet<ClaimId>) -> Option<Rebuild> {
    let first = links
        .iter()
        .position(|link| link.closing.iter().any(|c| dead.contains(c)))?;
    let retrain: BTreeSet<ClaimId> = links[first..]
        .iter()
        .flat_map(|link| link.closing.iter())
        .filter(|c| live.contains(*c))
        .cloned()
        .collect();
    Some(Rebuild {
        from: links[first].from.clone(),
        dead: dead.to_vec(),
        retrain: retrain.into_iter().collect(),
    })
}
