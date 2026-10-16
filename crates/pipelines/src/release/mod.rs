// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gated self-improvement loops where a model
// update is adopted only on held-out evidence. If your team needs expertise
// in evaluation-gated fine-tuning, you can procure our services by sending
// an email to info@swedishembedded.com.

//! `release`, `release list` and `rollback`: a trained candidate becomes
//! the policy only through the gate, and an alias moves back along the
//! lineage it came by.
//!
//! A candidate is decided against the release its alias points at - the
//! champion - and only if it was trained from that release (or, before any
//! release, from the base): a candidate trained from the base once a
//! champion exists, or from a champion since replaced, is refused before
//! anything is measured. The gate ([`gate`]) grades the candidate and the
//! champion (the base, before any release) closed-book on the same suites
//! ([`probe`]): the new data's held-out tasks - the records training held
//! out and the variants of the tasks it trained on - every earlier release's, and
//! the anchor suite ([`anchor`]), each without the tasks the candidate was
//! trained on ([`leakage`]); then plain brain serves the candidate
//! ([`serve`]). Only when every check passes is the release written
//! ([`store`]) - the adapter copied, the manifest recording every number -
//! and the alias moved to it, from the champion it was measured against.
//! Whatever the decision, the concepts of the tasks a failed retention
//! suite shows forgotten are queued for new tasks
//! ([`crate::curriculum::queue`]).
//!
//! A candidate trained by `practice` works in a workspace, and its suites are
//! changes to a repository ([`exams`]): the exam its run reserved before any
//! attempt is the improvement suite, every earlier release's own exam is its
//! retention suite, and the frozen anchor suite's coding-agent tasks, if any,
//! are attempted too. The champion and the candidate attempt each task as the
//! solve stage does, under the same bounds and seeds (the `agentic` module), the
//! paired outcomes are clustered by the file each fault was seeded in, and the
//! served candidate is attempted through `brain serve` and held to the
//! verdicts of the in-process attempts ([`serve`]). Attempts that read the
//! hidden tests, and attempts the harness could not carry out, are set aside
//! and counted for each arm; they are in no denominator.
//!
//! A candidate that is not a language model - a full checkpoint scored by
//! metrics on held-out units - is released through [`predictive`], which
//! decides the predictive gate over the numbers its caller measured and
//! writes the same kind of release.
//!
//! `rollback` points an alias at the release its current one was trained
//! from, and refuses when there is none.
//!
//! Whenever an alias moves - a release, a rollback - the manifest
//! `brain serve --adapter-manifest` follows is rewritten to name what the
//! alias points at ([`splinter_orchestrator::releases::ReleaseStore::publish_manifest`]).

pub(crate) mod agentic;
pub mod anchor;
mod canary;
pub mod continual;
pub mod exams;
mod gating;
pub mod leakage;
mod listing;
pub mod meaning;
pub mod predictive;
pub mod probe;
pub mod serve;
mod statistics;

pub(crate) use gating::{grade_arm, Asked, Graded};
use gating::{run_gate, GateRun};
pub use listing::{list, rollback, ReleaseLine, ReleaseList, RolledBack};

use std::path::{Path, PathBuf};

use serde::Serialize;
use splinter_agent::CancelToken;
use splinter_core::digest::Digest;
use splinter_core::release::ReleaseId;
use splinter_core::rng::Seed;
use splinter_core::terms::{combine_stated, Distribution, Terms};
use splinter_knowledge::concepts::Concept;
use splinter_model::local::{load_source, resolve_base};
use splinter_model::selection::local_model_name;
use splinter_orchestrator::releases::{
    Provenance, ReleaseGate, ReleaseManifest, ReleasedArtifact, StoredRelease, RELEASE_FORMAT,
};

use crate::curriculum::queue::enqueue_retention;
use crate::train::{load_candidate, Candidate};
use splinter_core::model_ref::{is_alias_name, ModelRef, POLICY_DEFAULT};
use splinter_core::training::{TrainingCurve, TrainingSummary};
use splinter_eval::gate::{GateConfig, GateReport};
use splinter_orchestrator::config::Config;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_orchestrator::runs::Recorder;

/// One `release`.
#[derive(Clone, Debug, Serialize)]
pub struct ReleaseRequest {
    /// The candidate, by id or unique prefix.
    pub candidate: String,
    /// The alias it is released under.
    pub alias: String,
    /// The gate's thresholds.
    pub gate: GateConfig,
    /// How widely the release may be handed on. [`Distribution::Restricted`]
    /// (the default) records the terms and keeps the release local;
    /// [`Distribution::Unrestricted`] is refused, before anything is
    /// measured, unless the terms of everything the candidate was made from
    /// allow it.
    pub distribution: Distribution,
    /// The seed the gate's bootstrap intervals are resampled with; drawn,
    /// reported and kept in the gate report ([`GateReport::interval_seed`])
    /// when left out.
    pub interval_seed: Option<Seed>,
    /// Tasks that cannot be solved honestly. The candidate attempts them
    /// beside the exam, and a pass on any one refuses it with a typed reason
    /// ([`GateReport::canary_passes`]), whatever else it passed. Empty: no
    /// canary check.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub canaries: Vec<splinter_core::experience::Task>,
}

impl ReleaseRequest {
    /// `candidate` released under `default` through the default gate.
    #[must_use]
    pub fn new(candidate: impl Into<String>) -> Self {
        Self {
            candidate: candidate.into(),
            alias: POLICY_DEFAULT.into(),
            gate: GateConfig::default(),
            distribution: Distribution::default(),
            interval_seed: None,
            canaries: Vec::new(),
        }
    }
}

/// What `release` reports.
#[derive(Clone, Debug, Serialize)]
pub struct Released {
    /// The candidate.
    pub candidate: String,
    /// The alias.
    pub alias: String,
    /// The release it was measured against; `None` before any release,
    /// when the base is the champion.
    pub champion: Option<ReleaseId>,
    /// The gate, with every number.
    pub gate: GateReport,
    /// The concepts the candidate forgot on a retention suite that failed,
    /// queued for new tasks.
    pub requeued: Vec<Concept>,
    /// What the candidate's training curve warns of about the adapter the
    /// gate judged ([`Candidate::warnings`]): a pass is not silent about a
    /// large generalisation gap or a monitoring loss that rose before the
    /// end.
    pub warnings: Vec<String>,
    /// The release written; `None` when the gate blocked it.
    pub release: Option<ReleaseId>,
    /// The release's adapter file.
    pub adapter: Option<PathBuf>,
}

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

/// [`release`] as the work of the recorded run `run`: the gate's interval seed
/// is drawn through the recorder when the request names none, so the run
/// record states it as well as the gate report. A candidate that is already
/// released is not gated again, and its report keeps the seed of the gate
/// that decided it.
pub fn release_recorded(
    ctx: &Context,
    run: &mut Recorder<'_>,
    request: &ReleaseRequest,
) -> Result<Released, OrchestratorError> {
    let seeded = ReleaseRequest {
        interval_seed: Some(run.seed(GATE_INTERVAL_SEED, request.interval_seed)?),
        ..request.clone()
    };
    release(ctx, &seeded, &run.cancel_token())
}

/// Runs the gate on `request.candidate` and releases it if it passes.
pub fn release(
    ctx: &Context,
    request: &ReleaseRequest,
    cancel: &CancelToken,
) -> Result<Released, OrchestratorError> {
    if !is_alias_name(&request.alias) {
        return Err(OrchestratorError::Refused(format!(
            "{:?} is not an alias name",
            request.alias
        )));
    }
    let candidate = load_candidate(ctx, &request.candidate)?;
    let store = ctx.releases();
    let champion_id = store.alias(&request.alias)?;
    if let Some(made) = store.of_candidate(&candidate.candidate)? {
        return resume(ctx, request, &made, champion_id);
    }
    if candidate.parent != champion_id {
        let named = |id: &Option<ReleaseId>| {
            id.as_ref()
                .map_or("the base alone".to_string(), |id| format!("release {id}"))
        };
        return Err(OrchestratorError::Refused(format!(
            "candidate {} was trained from {}, but {} points at {}; only a candidate trained \
             from the champion can replace it: train again from policy:{}",
            candidate.candidate,
            named(&candidate.parent),
            request.alias,
            named(&champion_id),
            request.alias
        )));
    }
    let config = ctx.config();
    if candidate.base != config.policy_base {
        return Err(OrchestratorError::Refused(format!(
            "candidate {} sits on {}, not the policy base {}",
            candidate.candidate,
            candidate.base.display(),
            config.policy_base.display()
        )));
    }
    let champion = champion_id.as_ref().map(|id| store.get(id)).transpose()?;
    let terms = release_terms(&candidate, champion.as_ref());
    if request.distribution == Distribution::Unrestricted {
        terms.permits_unrestricted_release().map_err(|why| {
            OrchestratorError::Refused(format!(
                "candidate {} cannot be released unrestricted: {why}; release it restricted to \
                 keep it local",
                candidate.candidate
            ))
        })?;
    }
    let base_source = load_source(
        &resolve_base(&config.policy_base)
            .map_err(|e| OrchestratorError::Refused(format!("the policy base: {e}")))?,
    );
    let interval_seed = request.interval_seed.unwrap_or_else(|| {
        let seed = splinter_model::seed::draw();
        ctx.report_stage(
            "seed",
            &serde_json::json!({ "name": GATE_INTERVAL_SEED, "seed": seed, "drawn": true }),
        );
        seed
    });
    let gate = run_gate(
        ctx,
        &candidate,
        champion.as_ref(),
        &base_source,
        &GateRun {
            config: &request.gate,
            canaries: &request.canaries,
            interval_seed,
        },
        cancel,
    )?;
    let requeued = enqueue_retention(ctx, &gate)?;
    let mut released = Released {
        candidate: candidate.candidate.clone(),
        alias: request.alias.clone(),
        champion: champion_id.clone(),
        gate,
        requeued,
        warnings: candidate.warnings(),
        release: None,
        adapter: None,
    };
    if !released.gate.passed {
        return Ok(released);
    }
    let gate = ReleaseGate::Llm {
        report: Box::new(released.gate.clone()),
    };
    let manifest = manifest(ctx, &candidate, gate, terms, request.distribution)?;
    // The release is made official first, with its place in the lineage, in
    // one commit; only then does the alias move.
    let stored = store.put(&manifest)?;
    store.move_alias(
        &request.alias,
        champion_id.as_ref(),
        &stored.id,
        &ctx.clock().utc_now(),
    )?;
    ctx.repin_policy(&request.alias);
    released.release = Some(stored.id);
    released.adapter = Some(stored.artifact);
    Ok(released)
}

/// A candidate that already has a release: either the alias points at it and
/// there is nothing to do, or the process that made it official died before
/// the alias moved and the move is made now, against the champion it was
/// decided against. The gate is not run again: its verdict is in the
/// release.
fn resume(
    ctx: &Context,
    request: &ReleaseRequest,
    made: &StoredRelease,
    champion: Option<ReleaseId>,
) -> Result<Released, OrchestratorError> {
    let store = ctx.releases();
    let gate = made.manifest.gate.llm().ok_or_else(|| {
        OrchestratorError::Refused(format!(
            "candidate {} was released as {} through another gate, not the answer-grading one",
            made.manifest.candidate, made.id
        ))
    })?;
    made.adapter()?;
    if champion.as_ref() != Some(&made.id) {
        if champion != made.manifest.parent {
            return Err(OrchestratorError::Refused(format!(
                "candidate {} was released as {}, and {} has moved on since; \
                 train again from policy:{}",
                made.manifest.candidate, made.id, request.alias, request.alias
            )));
        }
        store.move_alias(
            &request.alias,
            champion.as_ref(),
            &made.id,
            &ctx.clock().utc_now(),
        )?;
        ctx.repin_policy(&request.alias);
    } else {
        // The alias is already there. A process that died after moving it and
        // before the manifest a server follows was written left that one
        // behind.
        store.publish_manifest(&request.alias)?;
    }
    Ok(Released {
        candidate: made.manifest.candidate.clone(),
        alias: request.alias.clone(),
        champion: made.manifest.parent.clone(),
        gate: gate.clone(),
        requeued: Vec::new(),
        warnings: made
            .manifest
            .training
            .curve
            .as_ref()
            .map(TrainingCurve::warnings)
            .unwrap_or_default(),
        release: Some(made.id.clone()),
        adapter: Some(made.artifact.clone()),
    })
}

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
            ..Provenance::default()
        },
        generation: None,
        created_at: ctx.clock().utc_now(),
    })
}
