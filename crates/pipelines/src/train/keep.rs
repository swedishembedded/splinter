// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gated self-improvement loops where a model
// update is adopted only on held-out evidence. If your team needs expertise
// in evaluation-gated fine-tuning, you can procure our services by sending
// an email to info@swedishembedded.com.

//! Keeping what a training produced: the adapter, the adapters of its
//! evaluations and the replayed records as artifacts, then the candidate's
//! record.

use std::path::PathBuf;

use splinter_core::digest::Digest;
use splinter_core::terms::{combine_stated, Terms};
use splinter_core::training::{
    HeldOutScore, PreferenceSummary, Regime, RehearsalSample, ReplaySample, TrainingCurve,
};
use splinter_model::train::{Trained, TrainedPreference};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::{io, OrchestratorError};
use splinter_store::artifacts::ArtifactSpec;

use super::{candidate, Candidate, Checkpoint, TrainPlan, TrainRequest};

/// A finished training: what was planned, asked and produced, for the
/// candidate it made.
pub(super) struct Finished<'a> {
    pub(super) plan: &'a TrainPlan,
    pub(super) replay: &'a Option<ReplaySample>,
    pub(super) trained: Outcome,
    pub(super) request: &'a TrainRequest,
    pub(super) candidate: &'a str,
    pub(super) regime: Regime,
}

/// Keeps what training produced: the adapter and the replayed records as
/// artifacts, then the candidate's record and the training that made it in one
/// commit.
pub(super) fn keep_candidate(
    ctx: &Context,
    finished: Finished<'_>,
) -> Result<Candidate, OrchestratorError> {
    let Finished {
        plan,
        replay,
        trained,
        request,
        candidate,
        regime,
    } = finished;
    let artifacts = ctx.artifacts();
    let adapter = artifacts.put_file(
        &trained.adapter,
        &ArtifactSpec::new("adapter", "brain-trainer")
            .with_extension(".safetensors")
            .with_sha256(),
    )?;
    let reported = Digest::parse(&trained.adapter_digest)
        .map_err(|e| OrchestratorError::Train(format!("brain's adapter digest: {e}")))?;
    if adapter.sha256.as_ref() != Some(&reported) {
        return Err(OrchestratorError::Train(format!(
            "brain reported the adapter as {reported} but its file hashes to {}",
            adapter
                .sha256
                .as_ref()
                .map_or_else(|| "nothing".to_string(), ToString::to_string)
        )));
    }
    if let Some(file) = &plan.replay_file {
        let kept = artifacts.put_file(
            file,
            &ArtifactSpec::new("replay", "splinter-train").with_extension(".jsonl"),
        )?;
        if replay.as_ref().and_then(|r| r.digest.as_ref()) != Some(&kept.digest) {
            return Err(OrchestratorError::Train(
                "the replayed records changed while they were being kept".into(),
            ));
        }
    }
    let rehearsal = match &plan.rehearsal {
        Some(rehearsed) => {
            let kept = artifacts.put_file(
                &rehearsed.fit,
                &ArtifactSpec::new("rehearsal", "splinter-train").with_extension(".jsonl"),
            )?;
            Some(RehearsalSample {
                dataset: rehearsed.dataset.id.clone(),
                share: rehearsed.share,
                trained: rehearsed.trained,
                monitored: rehearsed.monitored,
                digest: Some(kept.digest),
            })
        }
        None => None,
    };
    let record_text =
        std::fs::read_to_string(&trained.training_record).map_err(io(&trained.training_record))?;
    let training_record =
        serde_json::from_str(&record_text).map_err(|source| OrchestratorError::Json {
            what: trained.training_record.display().to_string(),
            source,
        })?;
    let mut checkpoints = Vec::with_capacity(trained.evaluations.len());
    for (step, path) in &trained.evaluations {
        let kept = artifacts.put_file(
            path,
            &ArtifactSpec::new("adapter", "brain-trainer")
                .with_extension(".safetensors")
                .with_sha256(),
        )?;
        let monitor_loss = trained
            .curve
            .as_ref()
            .and_then(|c| c.points.iter().find(|p| p.step == *step))
            .map(|p| p.monitor_loss)
            .ok_or_else(|| {
                OrchestratorError::Train(format!("no evaluation of step {step} on the curve"))
            })?;
        checkpoints.push(Checkpoint {
            step: *step,
            monitor_loss,
            adapter_artifact: kept.digest,
            adapter_digest: kept
                .sha256
                .as_ref()
                .map(ToString::to_string)
                .ok_or_else(|| {
                    OrchestratorError::Train(format!("{} was not hashed", path.display()))
                })?,
        });
    }
    let record = Candidate {
        candidate: candidate.to_string(),
        from: request.from.to_string(),
        regime,
        base: plan.base.clone(),
        parent: plan.parent.clone(),
        datasets: plan.datasets.iter().map(|d| d.id.clone()).collect(),
        replay: replay.clone(),
        rehearsal,
        adapter: artifacts.path(&adapter.digest)?,
        adapter_artifact: adapter.digest,
        adapter_digest: trained.adapter_digest,
        base_digest: trained.base_digest,
        training_record,
        steps: plan.steps,
        rank: request.rank,
        base_score: trained.base_score,
        tuned_score: trained.tuned_score,
        preference: trained.preference,
        curve: trained.curve,
        checkpoints,
        records: trained.records,
        terms: plan_terms(ctx, plan)?,
    };
    candidate::record_candidate(ctx, &record, regime)?;
    Ok(record)
}

/// The terms of what `plan` trains on: its datasets' and those of the release
/// it continues, whose own terms already hold everything it was trained on, so
/// the strongest restriction anywhere upstream is the one that carries on.
fn plan_terms(ctx: &Context, plan: &TrainPlan) -> Result<Option<Terms>, OrchestratorError> {
    let parent = plan
        .parent
        .as_ref()
        .map(|id| ctx.releases().get(id))
        .transpose()?;
    Ok(combine_stated(
        plan.datasets
            .iter()
            .map(|d| d.manifest.terms.as_ref())
            .chain(parent.as_ref().map(|p| Some(&p.manifest.terms))),
    ))
}

/// What either regime's trainer produced, as a candidate records it.
pub(super) struct Outcome {
    adapter: PathBuf,
    adapter_digest: String,
    base_digest: String,
    training_record: PathBuf,
    records: usize,
    base_score: Option<HeldOutScore>,
    tuned_score: Option<HeldOutScore>,
    preference: Option<PreferenceSummary>,
    curve: Option<TrainingCurve>,
    evaluations: Vec<(u32, PathBuf)>,
}

impl From<Trained> for Outcome {
    fn from(t: Trained) -> Self {
        Self {
            adapter: t.adapter,
            adapter_digest: t.adapter_digest,
            base_digest: t.base_digest,
            training_record: t.training_record,
            records: t.records,
            base_score: Some(t.base),
            tuned_score: Some(t.tuned),
            preference: None,
            curve: Some(t.curve),
            evaluations: t.evaluations,
        }
    }
}

impl From<TrainedPreference> for Outcome {
    fn from(t: TrainedPreference) -> Self {
        Self {
            adapter: t.adapter,
            adapter_digest: t.adapter_digest,
            base_digest: t.base_digest,
            training_record: t.training_record,
            records: t.records,
            base_score: None,
            tuned_score: None,
            preference: Some(PreferenceSummary {
                beta: t.beta,
                reference_adapter: t.reference_adapter,
                train_score: t.train_score,
                held_out_score: t.held_out_score,
            }),
            curve: None,
            evaluations: Vec::new(),
        }
    }
}
