// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gated self-improvement loops where a model
// update is adopted only on held-out evidence. If your team needs expertise
// in evaluation-gated fine-tuning, you can procure our services by sending
// an email to info@swedishembedded.com.

//! A trained candidate as it is recorded and read back: what it was trained
//! from and on, the adapter it carries and what was measured of it.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use splinter_core::dataset::DatasetId;
use splinter_core::digest::Digest;
use splinter_core::release::ReleaseId;
use splinter_core::terms::Terms;
use splinter_core::training::{
    HeldOutScore, PreferenceSummary, Regime, RehearsalSample, ReplaySample, TrainingCurve,
};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

/// The class a candidate's record is stored under.
const CANDIDATE: &str = "candidate";

/// The adapter of one evaluation of a run, kept so a step can be chosen on
/// something other than the monitoring loss.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Steps trained when the adapter was taken.
    pub step: u32,
    /// The monitoring loss at that step.
    pub monitor_loss: f32,
    /// The artifact the adapter is kept as.
    pub adapter_artifact: Digest,
    /// Its SHA-256, as brain reports an adapter's digest.
    pub adapter_digest: String,
}

/// A trained candidate, as stored: everything but where its adapter file is.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct StoredCandidate {
    candidate: String,
    from: String,
    #[serde(default)]
    regime: Regime,
    base: PathBuf,
    parent: Option<ReleaseId>,
    datasets: Vec<DatasetId>,
    replay: Option<ReplaySample>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    rehearsal: Option<RehearsalSample>,
    adapter_artifact: Digest,
    adapter_digest: String,
    base_digest: String,
    training_record: serde_json::Value,
    steps: u32,
    rank: u32,
    #[serde(default)]
    base_score: Option<HeldOutScore>,
    #[serde(default)]
    tuned_score: Option<HeldOutScore>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    preference: Option<PreferenceSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    curve: Option<TrainingCurve>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    checkpoints: Vec<Checkpoint>,
    records: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    terms: Option<Terms>,
}

/// A trained candidate.
#[derive(Clone, Debug, Serialize)]
pub struct Candidate {
    /// Its id.
    pub candidate: String,
    /// The reference it was trained from.
    pub from: String,
    /// How it was trained.
    pub regime: Regime,
    /// The base checkpoint.
    pub base: PathBuf,
    /// The release it was trained from; `None` from a base or a `local:`
    /// adapter.
    pub parent: Option<ReleaseId>,
    /// The new datasets trained on.
    pub datasets: Vec<DatasetId>,
    /// The earlier records replayed; `None` with no release to replay.
    pub replay: Option<ReplaySample>,
    /// The base's own answers rehearsed beside the new records; `None` when
    /// none were.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rehearsal: Option<RehearsalSample>,
    /// The adapter file: an artifact, a real file at a stable path.
    pub adapter: PathBuf,
    /// The artifact the adapter is kept as.
    pub adapter_artifact: Digest,
    /// Its digest, as brain reports it.
    pub adapter_digest: String,
    /// The digest of the base it was trained on, as its card records it.
    pub base_digest: String,
    /// brain's training record for it.
    pub training_record: serde_json::Value,
    /// The step budget: the most steps the run was allowed.
    pub steps: u32,
    /// LoRA rank asked for.
    pub rank: u32,
    /// The base on the held-out records; `None` when not measured (the
    /// preference regime measures preferences instead).
    pub base_score: Option<HeldOutScore>,
    /// The base with the adapter on the same records; `None` when not
    /// measured.
    pub tuned_score: Option<HeldOutScore>,
    /// The preference measurements; `None` for the supervised regime.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preference: Option<PreferenceSummary>,
    /// The monitoring curve and which step the adapter is; `None` for the
    /// preference regime and for a candidate recorded before curves were.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub curve: Option<TrainingCurve>,
    /// The adapter of every evaluation of the run, when it was asked to keep
    /// them: the one the candidate carries is among them.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub checkpoints: Vec<Checkpoint>,
    /// Records in the new datasets, trained and held out together.
    pub records: usize,
    /// The terms of what it was trained on: its datasets' and those of the
    /// release it continued, combined (the most restrictive of each axis);
    /// `None` when none were stated, which is unknown.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub terms: Option<Terms>,
}

impl Candidate {
    /// The adapter file of `checkpoint`.
    pub fn checkpoint_adapter(
        &self,
        ctx: &Context,
        checkpoint: &Checkpoint,
    ) -> Result<PathBuf, OrchestratorError> {
        Ok(ctx.artifacts().path(&checkpoint.adapter_artifact)?)
    }

    /// What the candidate's curve warns of about the adapter it carries
    /// ([`TrainingCurve::warnings`]); nothing for a candidate without one.
    #[must_use]
    pub fn warnings(&self) -> Vec<String> {
        self.curve
            .as_ref()
            .map(TrainingCurve::warnings)
            .unwrap_or_default()
    }

    pub(super) fn stored(&self) -> StoredCandidate {
        StoredCandidate {
            candidate: self.candidate.clone(),
            from: self.from.clone(),
            regime: self.regime,
            base: self.base.clone(),
            parent: self.parent.clone(),
            datasets: self.datasets.clone(),
            replay: self.replay.clone(),
            rehearsal: self.rehearsal.clone(),
            adapter_artifact: self.adapter_artifact.clone(),
            adapter_digest: self.adapter_digest.clone(),
            base_digest: self.base_digest.clone(),
            training_record: self.training_record.clone(),
            steps: self.steps,
            rank: self.rank,
            base_score: self.base_score,
            tuned_score: self.tuned_score,
            preference: self.preference.clone(),
            curve: self.curve.clone(),
            checkpoints: self.checkpoints.clone(),
            records: self.records,
            terms: self.terms.clone(),
        }
    }

    fn from_stored(ctx: &Context, stored: StoredCandidate) -> Result<Self, OrchestratorError> {
        Ok(Self {
            adapter: ctx.artifacts().path(&stored.adapter_artifact)?,
            candidate: stored.candidate,
            from: stored.from,
            regime: stored.regime,
            base: stored.base,
            parent: stored.parent,
            datasets: stored.datasets,
            replay: stored.replay,
            rehearsal: stored.rehearsal,
            adapter_artifact: stored.adapter_artifact,
            adapter_digest: stored.adapter_digest,
            base_digest: stored.base_digest,
            training_record: stored.training_record,
            steps: stored.steps,
            rank: stored.rank,
            base_score: stored.base_score,
            tuned_score: stored.tuned_score,
            preference: stored.preference,
            curve: stored.curve,
            checkpoints: stored.checkpoints,
            records: stored.records,
            terms: stored.terms,
        })
    }
}

/// Records `candidate` under its class, with the datasets and the parent it
/// is indexed by.
pub(super) fn record_candidate(
    ctx: &Context,
    candidate: &Candidate,
    regime: Regime,
) -> Result<(), OrchestratorError> {
    let datasets: Vec<Digest> = candidate.datasets.iter().map(|d| d.0.clone()).collect();
    ctx.workspace().record_candidate(
        &candidate.stored(),
        &candidate.candidate,
        &datasets,
        match regime {
            Regime::Sft => "sft",
            Regime::Dpo => "dpo",
            Regime::Other => "other",
        },
        candidate.parent.as_ref().map(|release| &release.0),
    )?;
    Ok(())
}

/// The candidate `id` (or a unique prefix of it) names, as trained.
pub fn load_candidate(ctx: &Context, id: &str) -> Result<Candidate, OrchestratorError> {
    let all = stored_candidates(ctx)?;
    let matching: Vec<&StoredCandidate> =
        all.iter().filter(|c| c.candidate.starts_with(id)).collect();
    let found = match matching.as_slice() {
        [] => {
            return Err(OrchestratorError::NotFound {
                what: "candidate",
                id: id.into(),
            })
        }
        [one] => *one,
        many => match many.iter().find(|c| c.candidate == id) {
            Some(exact) => *exact,
            None => {
                return Err(OrchestratorError::AmbiguousId {
                    what: "candidate",
                    id: id.into(),
                    matches: many.len(),
                })
            }
        },
    };
    Candidate::from_stored(ctx, found.clone())
}

fn stored_candidates(ctx: &Context) -> Result<Vec<StoredCandidate>, OrchestratorError> {
    ctx.workspace().refresh()?;
    let mut all = Vec::new();
    for id in ctx.workspace().documents_in_order(CANDIDATE)? {
        if let Some(stored) = ctx
            .workspace()
            .get_document::<StoredCandidate>(CANDIDATE, &id)?
        {
            all.push(stored);
        }
    }
    all.sort_by(|a, b| a.candidate.cmp(&b.candidate));
    Ok(all)
}

/// Every trained candidate's id, oldest first.
pub fn candidate_ids(ctx: &Context) -> Result<Vec<String>, OrchestratorError> {
    Ok(stored_candidates(ctx)?
        .into_iter()
        .map(|c| c.candidate)
        .collect())
}

/// How many candidates were trained under the state root.
pub fn candidate_count(ctx: &Context) -> Result<usize, OrchestratorError> {
    Ok(candidate_ids(ctx)?.len())
}

/// The candidate that is `candidate` with the adapter of the evaluation at
/// `step` instead of the one it carries: a step chosen on something other than
/// the monitoring loss ([`crate::checkpoints`]). It is a candidate of its
/// own, trained on what the first was; what was measured of the exported
/// adapter alone (its held-out score) is not carried over, since it is not
/// measured of this one, and the curve says the step was chosen on the dev
/// suite.
pub fn adopt_checkpoint(
    ctx: &Context,
    candidate: &Candidate,
    step: u32,
) -> Result<Candidate, OrchestratorError> {
    let checkpoint = candidate
        .checkpoints
        .iter()
        .find(|c| c.step == step)
        .ok_or_else(|| {
            OrchestratorError::Refused(format!(
                "candidate {} kept no evaluation at step {step}",
                candidate.candidate
            ))
        })?;
    let mut adopted = candidate.clone();
    adopted.candidate = splinter_store::new_id_with_prefix("candidate");
    adopted.adapter = candidate.checkpoint_adapter(ctx, checkpoint)?;
    adopted.adapter_artifact = checkpoint.adapter_artifact.clone();
    adopted.adapter_digest = checkpoint.adapter_digest.clone();
    adopted.tuned_score = None;
    if let Some(curve) = adopted.curve.as_mut() {
        curve.selected_step = step;
        curve.selection = splinter_core::training::Selection::DevSuite;
    }
    record_candidate(ctx, &adopted, adopted.regime)?;
    Ok(adopted)
}
