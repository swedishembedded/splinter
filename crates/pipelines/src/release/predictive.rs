// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pre-registered release gates for
// predictive models, where a candidate replaces the champion only on
// paired held-out evidence. If your team needs expertise in validating
// risk models or forecasters before they ship, you can procure our
// services by sending an email to info@swedishembedded.com.

//! `release_predictive`: a full-checkpoint candidate that is not a language
//! model becomes the champion of an alias only through the predictive gate.
//!
//! The answer-grading gate ([`super::release`]) grades an adapter's answers
//! on tasks; a predictive model has none to grade. This stage is the other
//! way to a release: the caller trains with brain, scores champion and
//! candidate on the same held-out units, and hands over the checkpoint file,
//! the numbers with their intervals and the requirements they were written
//! against ([`PredictiveRelease`]). Splinter computes none of it. It checks
//! the units pair up, decides the five checks
//! ([`splinter_eval::predictive_gate`]), records a candidate that fails as
//! rejected (the champion stays where it is), and, only when all pass, keeps the
//! checkpoint immutably, makes the release official with its place in the
//! lineage and moves the alias from the champion it was measured against.
//!
//! The candidate id is the caller's and the operation's identity: asking
//! again for a candidate that was already released moves the alias if the
//! process died before it did, and changes nothing else.

use std::path::PathBuf;

use serde::Serialize;
use splinter_core::dataset::DatasetId;
use splinter_core::digest::Digest;
use splinter_core::release::ReleaseId;
use splinter_core::terms::{combine_stated, Distribution};
use splinter_core::training::{Regime, TrainingSummary};
use splinter_eval::predictive_gate::{
    decide_predictive, Measurements, PredictiveReport, PredictiveSpec,
};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_orchestrator::releases::{
    Provenance, Rejection, ReleaseGate, ReleaseManifest, ReleasedArtifact, StoredRelease,
    RELEASE_FORMAT,
};
use splinter_store::artifacts::ArtifactSpec;

use crate::dataset_ref::{record_lineage, resolve_any, AnyDataset};
use splinter_core::model_ref::is_alias_name;

/// The checkpoint file a candidate is, as its producer reports it.
#[derive(Clone, Debug, Serialize)]
pub struct CheckpointFile {
    /// Where the file is now; it is copied into the artifact store.
    pub path: PathBuf,
    /// The model architecture, as brain names it.
    pub architecture: String,
    /// The SHA-256 of the file as brain reports it; the stored file is held
    /// to it.
    pub digest: Digest,
}

/// How the candidate was trained, as far as the release records it.
#[derive(Clone, Debug, Serialize)]
pub struct CheckpointTraining {
    /// What it was trained from, in the caller's words.
    pub from: String,
    /// The objective its datasets serve, recorded on the training run in the
    /// lineage.
    pub objective: String,
    /// Optimizer steps.
    pub steps: u32,
    /// Records in the datasets.
    pub records: usize,
    /// brain's own training record of the checkpoint.
    pub record: serde_json::Value,
}

/// One `release_predictive`.
#[derive(Clone, Debug, Serialize)]
pub struct PredictiveRelease {
    /// The candidate's id, chosen by the caller: the release is made at most
    /// once for it.
    pub candidate: String,
    /// The alias it is released under, replacing its champion.
    pub alias: String,
    /// The stored datasets it was trained on, by id or unique prefix.
    pub datasets: Vec<String>,
    /// The checkpoint.
    pub checkpoint: CheckpointFile,
    /// How it was trained.
    pub training: CheckpointTraining,
    /// The requirements of each check, written before it was scored.
    pub spec: PredictiveSpec,
    /// What was measured, on which units.
    pub measurements: Measurements,
    /// How widely the release is to be handed on; the policy check holds the
    /// terms of everything it was made from to it.
    pub distribution: Distribution,
    /// The digests of the training configuration, the calibration artifact
    /// and the commits, as far as the caller has them. The datasets' and the
    /// gate's evaluation split digests are added here.
    pub provenance: Provenance,
}

/// What `release_predictive` reports.
#[derive(Clone, Debug, Serialize)]
pub struct PredictiveReleased {
    /// The candidate.
    pub candidate: String,
    /// The alias.
    pub alias: String,
    /// The release it was measured against; `None` before any release.
    pub champion: Option<ReleaseId>,
    /// The gate, with every number.
    pub report: PredictiveReport,
    /// The release written; `None` when the gate blocked it.
    pub release: Option<ReleaseId>,
    /// The release's checkpoint file.
    pub artifact: Option<PathBuf>,
}

/// Runs the predictive gate on `request.candidate` and releases it if it
/// passes. Refused, with nothing recorded, when the champion and candidate
/// were not scored on the same units or the checkpoint is not the file its
/// digest names; a gate that fails is not an error but a report with no
/// release.
pub fn release_predictive(
    ctx: &Context,
    request: &PredictiveRelease,
) -> Result<PredictiveReleased, OrchestratorError> {
    if !is_alias_name(&request.alias) {
        return Err(OrchestratorError::Refused(format!(
            "{:?} is not an alias name",
            request.alias
        )));
    }
    if request.candidate.trim().is_empty() {
        return Err(OrchestratorError::Refused(
            "name the candidate: its id is what makes a release happen once".into(),
        ));
    }
    let store = ctx.releases();
    let champion_id = store.alias(&request.alias)?;
    if let Some(made) = store.of_candidate(&request.candidate)? {
        return resume(ctx, request, &made, champion_id);
    }
    let champion = champion_id.as_ref().map(|id| store.get(id)).transpose()?;
    if request.datasets.is_empty() {
        return Err(OrchestratorError::Refused(format!(
            "candidate {} names no dataset it was trained on",
            request.candidate
        )));
    }
    let datasets = request
        .datasets
        .iter()
        .map(|id| resolve_any(ctx, id))
        .collect::<Result<Vec<_>, _>>()?;
    // The terms of what it was trained on, and of the champion it continues,
    // whose own terms already hold everything before it.
    let terms = combine_stated(
        datasets
            .iter()
            .map(|d| d.terms())
            .chain(champion.as_ref().map(|c| Some(&c.manifest.terms))),
    );
    let report = decide_predictive(
        &request.spec,
        &request.measurements,
        terms.as_ref(),
        request.distribution,
    )
    .map_err(|e| {
        OrchestratorError::Refused(format!(
            "candidate {} against {}: {e}",
            request.candidate, request.alias
        ))
    })?;
    let mut released = PredictiveReleased {
        candidate: request.candidate.clone(),
        alias: request.alias.clone(),
        champion: champion_id.clone(),
        report,
        release: None,
        artifact: None,
    };
    if !released.report.passed {
        store.record_rejection(&Rejection {
            candidate: request.candidate.clone(),
            alias: request.alias.clone(),
            champion: champion_id.clone(),
            datasets: datasets.iter().map(|d| d.id().clone()).collect(),
            checkpoint: request.checkpoint.digest.clone(),
            distribution: request.distribution,
            report: released.report.clone(),
            metrics: request.measurements.evidence.clone(),
        })?;
        return Ok(released);
    }

    let kept = ctx.artifacts().put_file(
        &request.checkpoint.path,
        &ArtifactSpec::new("checkpoint", "brain-trainer")
            .with_extension(&extension_of(&request.checkpoint.path))
            .with_sha256(),
    )?;
    for dataset in &datasets {
        record_lineage(ctx, dataset)?;
    }
    let dataset_ids: Vec<DatasetId> = datasets.iter().map(|d| d.id().clone()).collect();
    let dataset_digests: Vec<Digest> = dataset_ids.iter().map(|d| d.0.clone()).collect();
    ctx.workspace().record_external_candidate(
        &request.candidate,
        &dataset_digests,
        &request.training.objective,
        champion_id.as_ref().map(|id| &id.0),
    )?;
    let mut provenance = request.provenance.clone();
    provenance
        .dataset_snapshots
        .extend(datasets.iter().map(|d| d.snapshot().clone()));
    if !provenance
        .evaluation_splits
        .contains(&released.report.units_digest)
    {
        provenance
            .evaluation_splits
            .push(released.report.units_digest.clone());
    }
    let manifest = ReleaseManifest {
        format: RELEASE_FORMAT.into(),
        artifact: ReleasedArtifact::FullCheckpoint {
            architecture: request.checkpoint.architecture.clone(),
            checkpoint_digest: request.checkpoint.digest.clone(),
            checkpoint_artifact: kept.digest,
        },
        parent: champion_id.clone(),
        candidate: request.candidate.clone(),
        datasets: dataset_ids,
        replay: None,
        training: TrainingSummary {
            from: request.training.from.clone(),
            steps: request.training.steps,
            rank: 0,
            records: request.training.records,
            regime: Regime::Other,
            base_score: None,
            tuned_score: None,
            preference: None,
            curve: None,
            record: request.training.record.clone(),
            terms: combine_stated(datasets.iter().map(AnyDataset::terms)),
        },
        gate: ReleaseGate::Predictive {
            report: Box::new(released.report.clone()),
        },
        metrics: Some(request.measurements.evidence.clone()),
        // The report passed, so these terms were stated and allow the
        // distribution asked for.
        terms: terms.unwrap_or_else(|| splinter_core::terms::Terms::unknown("unstated")),
        distribution: request.distribution,
        provenance,
        created_at: ctx.clock().utc_now(),
    };
    let stored = store.put(&manifest)?;
    store.move_alias(
        &request.alias,
        champion_id.as_ref(),
        &stored.id,
        &ctx.clock().utc_now(),
    )?;
    ctx.repin_policy(&request.alias);
    released.release = Some(stored.id);
    released.artifact = Some(stored.artifact);
    Ok(released)
}

/// The extension (with its dot) a kept checkpoint keeps, empty for none.
fn extension_of(path: &std::path::Path) -> String {
    path.extension()
        .map(|e| format!(".{}", e.to_string_lossy()))
        .unwrap_or_default()
}

/// A candidate that already has a release: the alias is moved if the process
/// that made it died before it did, against the champion it was decided
/// against. The gate is not run again: its verdict is in the release.
fn resume(
    ctx: &Context,
    request: &PredictiveRelease,
    made: &StoredRelease,
    champion: Option<ReleaseId>,
) -> Result<PredictiveReleased, OrchestratorError> {
    let report = made.manifest.gate.predictive().ok_or_else(|| {
        OrchestratorError::Refused(format!(
            "candidate {} was released as {} through another gate, not the predictive one",
            made.manifest.candidate, made.id
        ))
    })?;
    if champion.as_ref() != Some(&made.id) {
        if champion != made.manifest.parent {
            return Err(OrchestratorError::Refused(format!(
                "candidate {} was released as {}, and {} has moved on since; release a candidate \
                 measured against its current champion",
                made.manifest.candidate, made.id, request.alias
            )));
        }
        ctx.releases().move_alias(
            &request.alias,
            champion.as_ref(),
            &made.id,
            &ctx.clock().utc_now(),
        )?;
        ctx.repin_policy(&request.alias);
    }
    Ok(PredictiveReleased {
        candidate: made.manifest.candidate.clone(),
        alias: request.alias.clone(),
        champion: made.manifest.parent.clone(),
        report: report.clone(),
        release: Some(made.id.clone()),
        artifact: Some(made.artifact.clone()),
    })
}
