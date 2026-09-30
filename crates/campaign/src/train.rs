// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gated self-improvement loops where a model
// update is adopted only on held-out evidence. If your team needs expertise
// in evaluation-gated fine-tuning, you can procure our services by sending
// an email to info@swedishembedded.com.

//! The train stage: stored datasets become a candidate adapter, scored on
//! held-out records. Training never releases; `release` decides that.
//!
//! The datasets are concatenated in the order given into the candidate's
//! own directory, and the newest records of that file are held out
//! (`splinter_lab::holdout`). Trained from `policy:<alias>`, a candidate
//! continues the adapter of the release the alias was resolved to - never
//! the base weights once a release exists - and replays a seeded sample of
//! every earlier release's training records beside the new ones:
//! `replay_fraction` ([`DEFAULT_REPLAY_FRACTION`]) of each release's
//! records, drawn with [`REPLAY_SEED`], from the part of its datasets that
//! was trained on and never from what it held out, so its held-out tasks
//! stay unseen for the retention check. Replayed records are never held
//! out, so the held-out score measures the new data. From a `local:`
//! reference there is no release, so nothing is replayed.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use splinter_policy::train::{fine_tune, FineTune, HeldOutScore, Trained};
use splinter_policy::{ModelSelection, PolicyError};
use splinter_store::digest::Digest;
use splinter_store::write_atomic;
use splinter_views::{replay_sample, DatasetId, Format, Fraction, StoredDataset};
use sven_sdk::CancelToken;

use crate::context::{Context, PolicyPin};
use crate::datasets::resolve_dataset;
use crate::error::{io, CampaignError};
use crate::model_ref::ModelRef;
use crate::release::probe::split_records;
use crate::release::{ReleaseId, ReleaseStore};

/// Training steps when a command names none.
pub const DEFAULT_STEPS: u32 = 40;
/// LoRA rank of a new adapter when a command names none.
pub const DEFAULT_LORA_RANK: u32 = 8;
/// LoRA alpha of a new adapter: the update is scaled by `alpha / rank`.
pub const DEFAULT_LORA_ALPHA: f32 = 16.0;
/// The fraction of each earlier release's training records replayed.
pub const DEFAULT_REPLAY_FRACTION: f64 = 0.25;
/// The seed of the replay draw: the same records are replayed every time.
pub const REPLAY_SEED: u64 = 0;

/// The file a candidate's record is kept in, inside its directory.
pub const CANDIDATE_RECORD: &str = "candidate.json";
/// The replayed records, inside a candidate's directory.
pub const REPLAY_FILE: &str = "replay.jsonl";

/// One training request.
#[derive(Clone, Debug, Serialize)]
pub struct TrainRequest {
    /// The datasets trained on, by id or unique prefix, in order.
    pub datasets: Vec<String>,
    /// What to train: `policy:<alias>` continues its release, `local:`
    /// a base and an optional adapter on it.
    pub from: ModelRef,
    /// The fraction of each earlier release's training records replayed.
    pub replay_fraction: f64,
    /// Optimizer steps.
    pub steps: u32,
    /// LoRA rank of a new adapter.
    pub rank: u32,
}

/// Where the replayed records came from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReplaySource {
    /// The earlier release.
    pub release: ReleaseId,
    /// Its datasets.
    pub datasets: Vec<DatasetId>,
    /// Its records that were trained on, which the sample is drawn from.
    pub available: usize,
    /// Of those, the ones replayed.
    pub sampled: usize,
}

/// The earlier records replayed beside the new ones.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReplaySample {
    /// The fraction of each release's training records drawn.
    pub fraction: f64,
    /// The draw's seed.
    pub seed: u64,
    /// Each earlier release, oldest first.
    pub sources: Vec<ReplaySource>,
    /// Records replayed in all.
    pub records: usize,
    /// The digest of the replay file; `None` when nothing was drawn.
    pub digest: Option<Digest>,
}

/// Everything a trainer needs, resolved.
#[derive(Clone, Debug)]
pub struct TrainPlan {
    /// The candidate's id.
    pub candidate: String,
    /// Its directory: everything training writes goes here.
    pub dir: PathBuf,
    /// The datasets, in order; the newest records are held out.
    pub datasets: Vec<StoredDataset>,
    /// The base checkpoint.
    pub base: PathBuf,
    /// The adapter continued: the release's, for `policy:<alias>`.
    pub continue_from: Option<PathBuf>,
    /// The release trained from, when there is one.
    pub parent: Option<ReleaseId>,
    /// The replayed records, when any were drawn.
    pub replay_file: Option<PathBuf>,
    /// Optimizer steps.
    pub steps: u32,
    /// LoRA rank of a new adapter.
    pub rank: u32,
}

/// How a candidate was trained, as its release records it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrainingSummary {
    /// The reference it was trained from.
    pub from: String,
    /// Optimizer steps.
    pub steps: u32,
    /// LoRA rank asked for.
    pub rank: u32,
    /// Records in the new datasets.
    pub records: usize,
    /// The base on the held-out records.
    pub base_score: HeldOutScore,
    /// The base with the adapter on the same records.
    pub tuned_score: HeldOutScore,
    /// brain's own training record of the adapter.
    pub record: serde_json::Value,
}

/// A trained candidate.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Candidate {
    /// Its id, which names its directory under the state root's `train/`.
    pub candidate: String,
    /// The reference it was trained from.
    pub from: String,
    /// The base checkpoint.
    pub base: PathBuf,
    /// The release it was trained from; `None` from a base or a `local:`
    /// adapter.
    pub parent: Option<ReleaseId>,
    /// The adapter it continued, if any.
    pub continued_from: Option<PathBuf>,
    /// The new datasets trained on.
    pub datasets: Vec<DatasetId>,
    /// The earlier records replayed; `None` with no release to replay.
    pub replay: Option<ReplaySample>,
    /// The adapter file.
    pub adapter: PathBuf,
    /// Its digest.
    pub adapter_digest: String,
    /// brain's training record for it.
    pub training_record: PathBuf,
    /// Optimizer steps.
    pub steps: u32,
    /// LoRA rank asked for.
    pub rank: u32,
    /// The base on the held-out records.
    pub base_score: HeldOutScore,
    /// The base with the adapter on the same records.
    pub tuned_score: HeldOutScore,
    /// Records in the new datasets, trained and held out together.
    pub records: usize,
    /// Always `false`: training never releases; `release` decides.
    pub released: bool,
}

/// Trains a candidate; the seam `train` and `learn` train through.
pub trait Trainer {
    /// Trains the adapter `plan` describes into `plan.dir`, stopping when
    /// `cancel` fires.
    fn train(
        &self,
        ctx: &Context,
        plan: &TrainPlan,
        cancel: &CancelToken,
    ) -> Result<Trained, CampaignError>;
}

/// Trains with brain's LoRA fine-tune.
#[derive(Clone, Copy, Debug, Default)]
pub struct BrainTrainer;

impl Trainer for BrainTrainer {
    fn train(
        &self,
        _ctx: &Context,
        plan: &TrainPlan,
        cancel: &CancelToken,
    ) -> Result<Trained, CampaignError> {
        let combined = plan.dir.join("dataset.jsonl");
        let mut text = String::new();
        for dataset in &plan.datasets {
            text.push_str(&std::fs::read_to_string(&dataset.path).map_err(io(&dataset.path))?);
        }
        write_atomic(&combined, &text).map_err(io(&combined))?;
        let replayed: Vec<PathBuf> = plan.replay_file.iter().cloned().collect();
        fine_tune(&FineTune {
            model_dir: &plan.base,
            dataset: &combined,
            attempt_dir: &plan.dir,
            steps: plan.steps,
            rank: plan.rank,
            alpha: DEFAULT_LORA_ALPHA,
            replay: &replayed,
            continue_from: plan.continue_from.as_deref(),
            cancel: Some(cancel),
        })
        .map_err(|e| match e {
            PolicyError::Cancelled { .. } => CampaignError::Cancelled,
            other => CampaignError::Train(other.to_string()),
        })
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
    let fraction = Fraction::new(request.replay_fraction).map_err(|e| {
        CampaignError::Refused(format!("replay fraction {}: {e}", request.replay_fraction))
    })?;
    let datasets = request
        .datasets
        .iter()
        .map(|id| trainable(ctx, id))
        .collect::<Result<Vec<_>, _>>()?;
    let ModelSelection::Local(weights) = ctx.selection(&request.from)? else {
        return Err(CampaignError::Refused(format!(
            "{} is reached over the network and cannot be trained here",
            request.from
        )));
    };
    let pin: Option<PolicyPin> = match &request.from {
        ModelRef::Policy(alias) => ctx.policy_pin(alias)?,
        _ => None,
    };
    let candidate = splinter_store::new_id_with_prefix("candidate");
    let dir = ctx.root().train().join(&candidate);
    std::fs::create_dir_all(&dir).map_err(io(&dir))?;
    let replay = match &pin {
        Some(pin) => Some(draw_replay(ctx, &pin.release, fraction, &dir)?),
        None => None,
    };
    let plan = TrainPlan {
        candidate: candidate.clone(),
        dir: dir.clone(),
        datasets,
        base: weights.base.clone(),
        continue_from: weights.adapter.clone(),
        parent: pin.map(|p| p.release),
        replay_file: replay
            .as_ref()
            .and_then(|r| r.digest.as_ref())
            .map(|_| dir.join(REPLAY_FILE)),
        steps: request.steps,
        rank: request.rank,
    };
    let trained = trainer.train(ctx, &plan, cancel)?;
    let record = Candidate {
        candidate,
        from: request.from.to_string(),
        base: plan.base,
        parent: plan.parent,
        continued_from: plan.continue_from,
        datasets: plan.datasets.iter().map(|d| d.id.clone()).collect(),
        replay,
        adapter: trained.adapter,
        adapter_digest: trained.adapter_digest,
        training_record: trained.training_record,
        steps: request.steps,
        rank: request.rank,
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

/// Draws the replay sample of every release in `release`'s lineage into
/// `dir`'s replay file.
fn draw_replay(
    ctx: &Context,
    release: &ReleaseId,
    fraction: Fraction,
    dir: &Path,
) -> Result<ReplaySample, CampaignError> {
    let lineage = ReleaseStore::open(ctx.root()).lineage(release)?;
    let mut sample = ReplaySample {
        fraction: fraction.get(),
        seed: REPLAY_SEED,
        sources: Vec::new(),
        records: 0,
        digest: None,
    };
    let mut text = String::new();
    for earlier in lineage.iter().rev() {
        let datasets = earlier.manifest.datasets.clone();
        let (trained_on, _held_out) = split_records(ctx, &datasets)?;
        let picked = replay_sample(&trained_on, fraction, REPLAY_SEED);
        for &index in &picked {
            text.push_str(&trained_on[index]);
            text.push('\n');
        }
        sample.records += picked.len();
        sample.sources.push(ReplaySource {
            release: earlier.id.clone(),
            datasets,
            available: trained_on.len(),
            sampled: picked.len(),
        });
    }
    if !text.is_empty() {
        let path = dir.join(REPLAY_FILE);
        write_atomic(&path, &text).map_err(io(&path))?;
        sample.digest = Some(Digest::of(text.as_bytes()));
    }
    Ok(sample)
}

/// The candidate `id` (or a unique prefix of it) names, as trained.
pub fn load_candidate(ctx: &Context, id: &str) -> Result<Candidate, CampaignError> {
    let matching: Vec<String> = candidate_ids(ctx)?
        .into_iter()
        .filter(|c| c.starts_with(id))
        .collect();
    let found = match matching.as_slice() {
        [] => {
            return Err(CampaignError::NotFound {
                what: "candidate",
                id: id.into(),
            })
        }
        [one] => one.clone(),
        many if many.iter().any(|c| c == id) => id.to_string(),
        many => {
            return Err(CampaignError::AmbiguousId {
                what: "candidate",
                id: id.into(),
                matches: many.len(),
            })
        }
    };
    let path = ctx.root().train().join(&found).join(CANDIDATE_RECORD);
    let text = std::fs::read_to_string(&path).map_err(io(&path))?;
    serde_json::from_str(&text).map_err(|source| CampaignError::Json {
        what: path.display().to_string(),
        source,
    })
}

/// Every trained candidate's id, oldest first.
pub fn candidate_ids(ctx: &Context) -> Result<Vec<String>, CampaignError> {
    let dir = ctx.root().train();
    let entries = match std::fs::read_dir(&dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(io(&dir)(e)),
    };
    let mut ids = Vec::new();
    for entry in entries {
        let entry = entry.map_err(io(&dir))?;
        if entry.path().join(CANDIDATE_RECORD).is_file() {
            ids.push(entry.file_name().to_string_lossy().into_owned());
        }
    }
    ids.sort();
    Ok(ids)
}

/// How many candidates were trained under the state root.
pub fn candidate_count(ctx: &Context) -> Result<usize, CampaignError> {
    Ok(candidate_ids(ctx)?.len())
}
