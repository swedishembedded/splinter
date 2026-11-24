// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gated self-improvement loops where a model
// update is adopted only on held-out evidence. If your team needs expertise
// in evaluation-gated fine-tuning, you can procure our services by sending
// an email to info@swedishembedded.com.

//! The releases: what the gates that absorb, consolidation and predictive
//! release share, `release list`, and `rollback`.
//!
//! A factory run releases through its `promotion` stage. What lives here is
//! what the other gates grade an arm with: the candidate and its champion
//! graded closed-book on the same suites ([`probe`]), the anchor suite
//! ([`anchor`]), without the tasks the candidate was trained on
//! ([`leakage`]), and served by plain brain ([`serve`]); a candidate that is
//! not a language model is released through [`predictive`]. The continual
//! results matrix of a lineage is [`continual`].
//!
//! `rollback` points an alias at the release its current one was trained
//! from, and refuses when there is none.
//!
//! Whenever an alias moves - a release, a rollback - the manifest
//! `brain serve --adapter-manifest` follows is rewritten to name what the
//! alias points at ([`splinter_orchestrator::releases::ReleaseStore::publish_manifest`]).

pub(crate) mod agentic;
pub mod anchor;
pub mod continual;
mod gating;
pub mod leakage;
mod listing;
pub mod meaning;
pub mod predictive;
pub mod probe;
pub mod serve;

pub(crate) use gating::{grade_arm, Asked, Graded};
pub use listing::{list, rollback, ReleaseLine, ReleaseList, RolledBack};

use std::path::Path;

use splinter_core::digest::Digest;
use splinter_core::terms::{combine_stated, Distribution, Terms};
use splinter_model::selection::local_model_name;
use splinter_orchestrator::releases::{
    Provenance, ReleaseGate, ReleaseManifest, ReleasedArtifact, StoredRelease, RELEASE_FORMAT,
};

use crate::train::Candidate;
use splinter_core::model_ref::ModelRef;
use splinter_core::training::TrainingSummary;
use splinter_orchestrator::config::Config;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

/// The reference the gate grades the configured base with `adapter`
/// through: an absolute `local:` checkpoint, so a caller holding a model for
/// it (a test's scripted one) can hand it in under the same reference.
#[must_use]
pub fn arm(config: &Config, adapter: Option<&Path>) -> ModelRef {
    arm_on(config, &config.policy_base, adapter)
}

/// [`arm`] on `base`, the base of the generation a policy is in: the
/// configured base in generation 0, a consolidated checkpoint after.
#[must_use]
pub fn arm_on(config: &Config, base: &Path, adapter: Option<&Path>) -> ModelRef {
    let base = std::path::absolute(base).unwrap_or_else(|_| base.to_path_buf());
    ModelRef::Local {
        checkpoint: base.display().to_string(),
        adapter: adapter.map(|a| a.display().to_string()),
        // An arm is the policy's base, so it is asked at the policy's context.
        context_tokens: config.policy_context_tokens,
    }
}

/// The name the gate's interval seed is recorded under in a run.
pub const GATE_INTERVAL_SEED: &str = "gate-interval";

/// The terms a release of `candidate` is made under: what training recorded,
/// combined with the champion's, which it continues. Unknown when neither
/// states any, so an unstated licence never counts as a granted one.
pub(crate) fn release_terms(candidate: &Candidate, champion: Option<&StoredRelease>) -> Terms {
    combine_stated(
        std::iter::once(candidate.terms.as_ref()).chain(champion.map(|c| Some(&c.manifest.terms))),
    )
    .unwrap_or_else(|| Terms::unknown("unstated"))
}

/// The manifest of `candidate` released after `gate`. The base digest is
/// the one training recorded on the adapter's card: the serve check has
/// just had brain bind the adapter to the base on disk, which it refuses
/// for any other base, so it is the base the release runs on.
pub(crate) fn manifest(
    ctx: &Context,
    candidate: &Candidate,
    gate: ReleaseGate,
    terms: Terms,
    distribution: Distribution,
) -> Result<ReleaseManifest, OrchestratorError> {
    let base_digest = Digest::parse(&candidate.base_digest).map_err(|e| {
        OrchestratorError::Refused(format!("candidate {}: {e}", candidate.candidate))
    })?;
    let adapter_digest = Digest::parse(&candidate.adapter_digest).map_err(|e| {
        OrchestratorError::Refused(format!("candidate {}: {e}", candidate.candidate))
    })?;
    let record = candidate.training_record.clone();
    let dataset_snapshots = candidate
        .datasets
        .iter()
        .map(|id| ctx.datasets().get(id).map(|d| d.manifest.dataset))
        .collect::<Result<Vec<_>, _>>()?;
    let revisions = &ctx.config().build.revisions;
    Ok(ReleaseManifest {
        format: RELEASE_FORMAT.into(),
        artifact: ReleasedArtifact::Adapter {
            base_model: local_model_name(&candidate.base),
            base_digest,
            adapter_digest,
            adapter_artifact: candidate.adapter_artifact.clone(),
        },
        parent: candidate.parent.clone(),
        candidate: candidate.candidate.clone(),
        datasets: candidate.datasets.clone(),
        replay: candidate.replay.clone(),
        rehearsal: candidate.rehearsal.clone(),
        training: TrainingSummary {
            from: candidate.from.clone(),
            steps: candidate.steps,
            rank: candidate.rank,
            records: candidate.records,
            regime: candidate.regime,
            base_score: candidate.base_score,
            tuned_score: candidate.tuned_score,
            preference: candidate.preference.clone(),
            curve: candidate.curve.clone(),
            record,
            terms: candidate.terms.clone(),
        },
        gate,
        metrics: None,
        terms,
        distribution,
        provenance: Provenance {
            dataset_snapshots,
            brain_commit: revisions.brain.clone(),
            splinter_commit: revisions.splinter.clone(),
            sven_commit: revisions.sven.clone(),
            ..Provenance::default()
        },
        generation: None,
        created_at: ctx.clock().utc_now(),
    })
}
