// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gated self-improvement loops where a model
// update is adopted only on held-out evidence. If your team needs expertise
// in evaluation-gated fine-tuning, you can procure our services by sending
// an email to info@swedishembedded.com.

//! The train stage: stored datasets become a candidate adapter, scored on
//! held-out records, and never released - releasing a candidate is not
//! part of training.
//!
//! The datasets are concatenated in the order given into the candidate's
//! own directory, and the newest records of that file are held out
//! (`splinter_lab::holdout`); a replayed dataset is mixed into training
//! whole and never held out, so the held-out score measures the new data.
//! `--from` names the base, and an adapter on it to continue training.

use std::path::PathBuf;

use serde::Serialize;
use splinter_policy::train::{fine_tune, FineTune, HeldOutScore};
use splinter_policy::{ModelSelection, PolicyError};
use splinter_store::write_atomic;
use splinter_views::{DatasetId, Format, StoredDataset};
use sven_sdk::CancelToken;

use crate::context::Context;
use crate::datasets::resolve_dataset;
use crate::error::{io, CampaignError};
use crate::model_ref::ModelRef;

/// Training steps when a command names none.
pub const DEFAULT_STEPS: u32 = 40;
/// LoRA rank of a new adapter when a command names none.
pub const DEFAULT_LORA_RANK: u32 = 8;
/// LoRA alpha of a new adapter: the update is scaled by `alpha / rank`.
pub const DEFAULT_LORA_ALPHA: f32 = 16.0;

/// The file a candidate's record is kept in, inside its directory.
pub const CANDIDATE_RECORD: &str = "candidate.json";

/// One training request.
#[derive(Clone, Debug, Serialize)]
pub struct TrainRequest {
    /// The datasets trained on, by id or unique prefix, in order.
    pub datasets: Vec<String>,
    /// The base to train, and an adapter on it to continue.
    pub from: ModelRef,
    /// A dataset mixed into training whole, never held out.
    pub replay: Option<String>,
    /// Optimizer steps.
    pub steps: u32,
    /// LoRA rank of a new adapter.
    pub rank: u32,
}

/// A trained candidate.
#[derive(Clone, Debug, Serialize)]
pub struct Candidate {
    /// Its id, which names its directory under the state root's `train/`.
    pub candidate: String,
    /// The reference it was trained from.
    pub from: String,
    /// The datasets trained on.
    pub datasets: Vec<DatasetId>,
    /// The dataset replayed, if any.
    pub replay: Option<DatasetId>,
    /// The adapter file.
    pub adapter: PathBuf,
    /// Its digest.
    pub adapter_digest: String,
    /// The base on the held-out records.
    pub base_score: HeldOutScore,
    /// The base with the adapter on the same records.
    pub tuned_score: HeldOutScore,
    /// Records trained and held out, together.
    pub records: usize,
    /// Always `false`: training never releases a candidate.
    pub released: bool,
}

/// Trains a candidate; the seam `learn` trains through.
pub trait Trainer {
    /// Trains a candidate on `datasets` (verified, trainable) as `request`
    /// asks, stopping when `cancel` fires.
    fn train(
        &self,
        ctx: &Context,
        request: &TrainRequest,
        datasets: &[StoredDataset],
        cancel: &CancelToken,
    ) -> Result<Candidate, CampaignError>;
}

/// Trains with brain's LoRA fine-tune.
#[derive(Clone, Copy, Debug, Default)]
pub struct BrainTrainer;

impl Trainer for BrainTrainer {
    fn train(
        &self,
        ctx: &Context,
        request: &TrainRequest,
        datasets: &[StoredDataset],
        cancel: &CancelToken,
    ) -> Result<Candidate, CampaignError> {
        let ModelSelection::Local(weights) = ctx.selection(&request.from)? else {
            return Err(CampaignError::Refused(format!(
                "{} is reached over the network and cannot be trained here",
                request.from
            )));
        };
        let replay = request
            .replay
            .as_deref()
            .map(|id| trainable(ctx, id))
            .transpose()?;
        let candidate = splinter_store::new_id_with_prefix("candidate");
        let dir = ctx.root().train().join(&candidate);
        std::fs::create_dir_all(&dir).map_err(io(&dir))?;
        let combined = dir.join("dataset.jsonl");
        let mut text = String::new();
        for dataset in datasets {
            text.push_str(&std::fs::read_to_string(&dataset.path).map_err(io(&dataset.path))?);
        }
        write_atomic(&combined, &text).map_err(io(&combined))?;
        let replayed: Vec<PathBuf> = replay.iter().map(|d| d.path.clone()).collect();
        let trained = fine_tune(&FineTune {
            model_dir: &weights.base,
            dataset: &combined,
            attempt_dir: &dir,
            steps: request.steps,
            rank: request.rank,
            alpha: DEFAULT_LORA_ALPHA,
            replay: &replayed,
            continue_from: weights.adapter.as_deref(),
            cancel: Some(cancel),
        })
        .map_err(|e| match e {
            PolicyError::Cancelled { .. } => CampaignError::Cancelled,
            other => CampaignError::Train(other.to_string()),
        })?;
        let record = Candidate {
            candidate,
            from: request.from.to_string(),
            datasets: datasets.iter().map(|d| d.id.clone()).collect(),
            replay: replay.map(|d| d.id),
            adapter: trained.adapter,
            adapter_digest: trained.adapter_digest,
            base_score: trained.base,
            tuned_score: trained.tuned,
            records: trained.records,
            released: false,
        };
        let path = dir.join(CANDIDATE_RECORD);
        let json = serde_json::to_string_pretty(&record).map_err(|source| CampaignError::Json {
            what: "candidate".into(),
            source,
        })?;
        write_atomic(&path, &json).map_err(io(&path))?;
        Ok(record)
    }
}

/// The stored dataset `id` names, refused unless brain can train it.
pub fn trainable(ctx: &Context, id: &str) -> Result<StoredDataset, CampaignError> {
    let dataset = resolve_dataset(ctx, id)?;
    if dataset.manifest.format != Format::GenericMessagesV2 {
        return Err(CampaignError::Refused(format!(
            "dataset {} is export-only ({:?} records); brain cannot train it",
            dataset.id, dataset.manifest.objective
        )));
    }
    Ok(dataset)
}

/// Trains a candidate on `request`'s datasets with `trainer`.
pub fn train(
    ctx: &Context,
    request: &TrainRequest,
    trainer: &dyn Trainer,
    cancel: &CancelToken,
) -> Result<Candidate, CampaignError> {
    if request.datasets.is_empty() {
        return Err(CampaignError::Refused("name at least one dataset".into()));
    }
    let datasets = request
        .datasets
        .iter()
        .map(|id| trainable(ctx, id))
        .collect::<Result<Vec<_>, _>>()?;
    trainer.train(ctx, request, &datasets, cancel)
}

/// How many candidates were trained under the state root.
pub fn candidate_count(ctx: &Context) -> Result<usize, CampaignError> {
    let dir = ctx.root().train();
    let entries = match std::fs::read_dir(&dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(e) => return Err(io(&dir)(e)),
    };
    let mut count = 0;
    for entry in entries {
        if entry
            .map_err(io(&dir))?
            .path()
            .join(CANDIDATE_RECORD)
            .is_file()
        {
            count += 1;
        }
    }
    Ok(count)
}
