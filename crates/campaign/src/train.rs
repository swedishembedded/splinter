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
//! The datasets' objective decides the [`Regime`]: chat datasets (SFT, and
//! classification rendered as SFT) are trained by supervised fine-tuning,
//! preference pair datasets (DPO) by direct preference optimisation. Every
//! dataset of one run must be of one regime, and an export-only dataset is
//! refused: brain cannot train it. The candidate records its regime; the
//! release gate's checks grade the model's answers and do not depend on
//! it.
//!
//! The datasets are concatenated in the order given into the candidate's
//! own directory, and the newest records of that file are held out
//! (`splinter_eval::holdout`). Trained from `policy:<alias>`, a candidate
//! continues the adapter of the release the alias was resolved to - never
//! the base weights once a release exists. A supervised candidate trained
//! so also replays a seeded sample of every earlier release's chat
//! training records beside the new ones: `replay_fraction`
//! ([`DEFAULT_REPLAY_FRACTION`]) of each such release's records, drawn with
//! [`REPLAY_SEED`], from the part of its datasets that was trained on and
//! never from what it held out, so its held-out tasks stay unseen for the
//! retention check. Replayed records are never held out, so the held-out
//! score measures the new data. From a `local:` reference there is no
//! release, so nothing is replayed. brain's preference fine-tune trains on
//! its pairs alone, so a preference candidate replays nothing; the frozen
//! reference it is trained against - the model it continues - is what keeps
//! it close to what came before, and the gate's retention check measures
//! whether it did.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use splinter_core::digest::Digest;
use splinter_data::holdout::split_dataset_file;
use splinter_data::{replay_sample, DatasetId, Format, Fraction, StoredDataset};
use splinter_policy::train::{
    fine_tune, train_preference, FineTune, HeldOutScore, PreferenceScore, PreferenceTune, Trained,
    TrainedPreference,
};
use splinter_policy::{ModelSelection, PolicyError};
use splinter_store::artifacts::ArtifactSpec;
use sven_sdk::CancelToken;

use crate::context::{Context, PolicyPin};
use crate::datasets::{record_dataset_lineage, resolve_dataset};
use crate::error::{io, CampaignError};
use crate::model_ref::ModelRef;
use crate::release::probe::split_records;
use crate::release::ReleaseId;

/// The peak learning rate a `learn` run trains a LoRA adapter at when none is
/// given: the rate that moves a low-rank update in the few hundred steps of a
/// short run, which brain's own default (set for long runs) does not.
pub const DEFAULT_LEARNING_RATE: f32 = 2e-4;

/// Training steps when a command names none.
pub const DEFAULT_STEPS: u32 = 40;
/// LoRA rank of a new adapter when a command names none.
pub const DEFAULT_LORA_RANK: u32 = 8;
/// LoRA alpha of a new adapter: the update is scaled by `alpha / rank`.
pub const DEFAULT_LORA_ALPHA: f32 = 16.0;
/// The fraction of each earlier release's training records replayed.
pub const DEFAULT_REPLAY_FRACTION: f64 = 0.25;
/// The DPO temperature of a preference fine-tune when a command names none:
/// brain's default.
pub use splinter_policy::train::DEFAULT_DPO_BETA;
/// The seed of the replay draw: the same records are replayed every time.
pub const REPLAY_SEED: u64 = 0;

/// The class a candidate's record is stored under.
const CANDIDATE: &str = "candidate";
/// The replayed records, inside a candidate's directory.
pub const REPLAY_FILE: &str = "replay.jsonl";

/// How a candidate was trained, decided by its datasets' objective.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Regime {
    /// Supervised fine-tuning on chat records (`generic-messages-v2`); the
    /// regime of a record that names none.
    #[default]
    Sft,
    /// Direct preference optimisation on chosen/rejected pairs
    /// (`generic-preference-v1`).
    Dpo,
}

impl Regime {
    /// The regime a dataset of `format` is trained by; `None` for a format
    /// brain does not train.
    #[must_use]
    pub fn of(format: Format) -> Option<Self> {
        match format {
            Format::GenericMessagesV2 => Some(Self::Sft),
            Format::GenericPreferenceV1 => Some(Self::Dpo),
            Format::SplinterExportV1 => None,
        }
    }
}

/// How a run trains beyond its step count and rank: what depends on the
/// base and the card, not on the data.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct Tuning {
    /// Hold the frozen base at bf16, half the bytes of fp32: what lets a 7B
    /// base train on one 24 GiB card. Supervised runs only; brain's
    /// preference trainer holds its base at its own tier.
    pub bf16_base: bool,
    /// The peak learning rate; brain's default when `None`.
    pub learning_rate: Option<f32>,
}

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
    /// The DPO temperature, for preference datasets only;
    /// [`DEFAULT_DPO_BETA`] when `None`.
    pub beta: Option<f32>,
    /// The base's precision and the learning rate.
    pub tuning: Tuning,
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
    /// Each earlier release trained on chat records, oldest first.
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
    /// How it is trained.
    pub regime: Regime,
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
    /// The DPO temperature; used by the preference regime only.
    pub beta: f32,
    /// The base's precision and the learning rate.
    pub tuning: Tuning,
}

/// A preference candidate's measurements: brain's preference score of the
/// adapter against the reference it was trained against.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PreferenceSummary {
    /// The DPO temperature it trained with.
    pub beta: f32,
    /// The digest of the adapter the reference carried (the one
    /// continued); `None` when the reference was the base alone.
    pub reference_adapter: Option<String>,
    /// On the pairs trained on; `None` when not measured.
    pub train_score: Option<PreferenceScore>,
    /// On the held-out pairs; `None` when not measured.
    pub held_out_score: Option<PreferenceScore>,
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
    /// How it was trained.
    #[serde(default)]
    pub regime: Regime,
    /// The base on the held-out records; `None` when not measured (the
    /// preference regime measures preferences instead).
    #[serde(default)]
    pub base_score: Option<HeldOutScore>,
    /// The base with the adapter on the same records; `None` when not
    /// measured.
    #[serde(default)]
    pub tuned_score: Option<HeldOutScore>,
    /// The preference measurements; `None` for the supervised regime.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preference: Option<PreferenceSummary>,
    /// brain's own training record of the adapter.
    pub record: serde_json::Value,
}

/// A trained candidate, as stored: everything but where its adapter file is.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct StoredCandidate {
    candidate: String,
    from: String,
    #[serde(default)]
    regime: Regime,
    base: PathBuf,
    parent: Option<ReleaseId>,
    datasets: Vec<DatasetId>,
    replay: Option<ReplaySample>,
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
    records: usize,
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
    /// Optimizer steps.
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
    /// Records in the new datasets, trained and held out together.
    pub records: usize,
}

impl Candidate {
    fn stored(&self) -> StoredCandidate {
        StoredCandidate {
            candidate: self.candidate.clone(),
            from: self.from.clone(),
            regime: self.regime,
            base: self.base.clone(),
            parent: self.parent.clone(),
            datasets: self.datasets.clone(),
            replay: self.replay.clone(),
            adapter_artifact: self.adapter_artifact.clone(),
            adapter_digest: self.adapter_digest.clone(),
            base_digest: self.base_digest.clone(),
            training_record: self.training_record.clone(),
            steps: self.steps,
            rank: self.rank,
            base_score: self.base_score,
            tuned_score: self.tuned_score,
            preference: self.preference.clone(),
            records: self.records,
        }
    }

    fn from_stored(ctx: &Context, stored: StoredCandidate) -> Result<Self, CampaignError> {
        Ok(Self {
            adapter: ctx.artifacts().path(&stored.adapter_artifact)?,
            candidate: stored.candidate,
            from: stored.from,
            regime: stored.regime,
            base: stored.base,
            parent: stored.parent,
            datasets: stored.datasets,
            replay: stored.replay,
            adapter_artifact: stored.adapter_artifact,
            adapter_digest: stored.adapter_digest,
            base_digest: stored.base_digest,
            training_record: stored.training_record,
            steps: stored.steps,
            rank: stored.rank,
            base_score: stored.base_score,
            tuned_score: stored.tuned_score,
            preference: stored.preference,
            records: stored.records,
        })
    }
}

/// Trains a candidate; the seam `train` and `learn` train through. `train`
/// calls the method of the plan's regime.
pub trait Trainer {
    /// Trains the supervised adapter `plan` describes into `plan.dir`,
    /// stopping when `cancel` fires.
    fn train(
        &self,
        ctx: &Context,
        plan: &TrainPlan,
        cancel: &CancelToken,
    ) -> Result<Trained, CampaignError>;

    /// Trains the preference adapter `plan` describes into `plan.dir`,
    /// stopping when `cancel` fires.
    fn train_preference(
        &self,
        ctx: &Context,
        plan: &TrainPlan,
        cancel: &CancelToken,
    ) -> Result<TrainedPreference, CampaignError>;
}

/// Trains with brain's LoRA fine-tunes: chat fine-tuning for the
/// supervised regime, preference fine-tuning for DPO.
#[derive(Clone, Copy, Debug, Default)]
pub struct BrainTrainer;

impl Trainer for BrainTrainer {
    fn train(
        &self,
        _ctx: &Context,
        plan: &TrainPlan,
        cancel: &CancelToken,
    ) -> Result<Trained, CampaignError> {
        let split = split_dataset_file(&combine(plan)?, &plan.dir)?;
        let replayed: Vec<PathBuf> = plan.replay_file.iter().cloned().collect();
        fine_tune(&FineTune {
            model_dir: &plan.base,
            train: &split.train,
            held_out: &split.held_out,
            attempt_dir: &plan.dir,
            steps: plan.steps,
            rank: plan.rank,
            alpha: DEFAULT_LORA_ALPHA,
            replay: &replayed,
            continue_from: plan.continue_from.as_deref(),
            cancel: Some(cancel),
            bf16_base: plan.tuning.bf16_base,
            learning_rate: plan.tuning.learning_rate,
            on_step: None,
        })
        .map_err(trainer_error)
    }

    fn train_preference(
        &self,
        _ctx: &Context,
        plan: &TrainPlan,
        cancel: &CancelToken,
    ) -> Result<TrainedPreference, CampaignError> {
        let split = split_dataset_file(&combine(plan)?, &plan.dir)?;
        train_preference(&PreferenceTune {
            model_dir: &plan.base,
            train: &split.train,
            held_out: &split.held_out,
            attempt_dir: &plan.dir,
            steps: plan.steps,
            rank: plan.rank,
            alpha: DEFAULT_LORA_ALPHA,
            beta: plan.beta,
            continue_from: plan.continue_from.as_deref(),
            cancel: Some(cancel),
        })
        .map_err(trainer_error)
    }
}

/// `plan`'s datasets concatenated in order into one file in its directory.
fn combine(plan: &TrainPlan) -> Result<PathBuf, CampaignError> {
    let combined = plan.dir.join("dataset.jsonl");
    let mut text = String::new();
    for dataset in &plan.datasets {
        text.push_str(&std::fs::read_to_string(&dataset.path).map_err(io(&dataset.path))?);
    }
    std::fs::write(&combined, &text).map_err(io(&combined))?;
    Ok(combined)
}

fn trainer_error(e: PolicyError) -> CampaignError {
    match e {
        PolicyError::Cancelled { .. } => CampaignError::Cancelled,
        other => CampaignError::Train(other.to_string()),
    }
}

/// The stored dataset `id` names and the regime brain trains it by,
/// refused when brain cannot train it.
pub fn trainable(ctx: &Context, id: &str) -> Result<(StoredDataset, Regime), CampaignError> {
    let dataset = resolve_dataset(ctx, id)?;
    let Some(regime) = Regime::of(dataset.manifest.format) else {
        return Err(CampaignError::Refused(format!(
            "dataset {} is export-only ({:?} records); brain cannot train it",
            dataset.id, dataset.manifest.objective
        )));
    };
    Ok((dataset, regime))
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
    let resolved = request
        .datasets
        .iter()
        .map(|id| trainable(ctx, id))
        .collect::<Result<Vec<_>, _>>()?;
    let regime = resolved[0].1;
    if let Some((other, _)) = resolved.iter().find(|(_, r)| *r != regime) {
        return Err(CampaignError::Refused(format!(
            "dataset {} holds {:?} records but {} holds {:?} records; one run trains one \
             regime",
            other.id, other.manifest.objective, resolved[0].0.id, resolved[0].0.manifest.objective
        )));
    }
    let beta = match (regime, request.beta) {
        (Regime::Dpo, beta) => beta.unwrap_or(DEFAULT_DPO_BETA),
        (Regime::Sft, None) => DEFAULT_DPO_BETA,
        (Regime::Sft, Some(_)) => {
            return Err(CampaignError::Refused(
                "beta applies to preference datasets only; these are chat datasets".into(),
            ))
        }
    };
    let datasets: Vec<StoredDataset> = resolved.into_iter().map(|(d, _)| d).collect();
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
    // Every dataset trained on must be traceable before hours are spent
    // training on it.
    for dataset in &datasets {
        record_dataset_lineage(ctx, dataset)?;
    }
    let candidate = splinter_store::new_id_with_prefix("candidate");
    let dir = ctx.root().work().join("train").join(&candidate);
    std::fs::create_dir_all(&dir).map_err(io(&dir))?;
    let replay = match (&pin, regime) {
        (Some(pin), Regime::Sft) => Some(draw_replay(ctx, &pin.release, fraction, &dir)?),
        _ => None,
    };
    let plan = TrainPlan {
        candidate: candidate.clone(),
        regime,
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
        beta,
        tuning: request.tuning,
    };
    // A fine-tune loads its own copy of the base: the device holds no
    // other while it trains.
    ctx.release_bases();
    let trained = match regime {
        Regime::Sft => Outcome::from(trainer.train(ctx, &plan, cancel)?),
        Regime::Dpo => Outcome::from(trainer.train_preference(ctx, &plan, cancel)?),
    };
    let record = keep_candidate(ctx, &plan, &replay, trained, request, &candidate, regime);
    // The directory was only ever a working place; the adapter and the
    // replay are kept as artifacts and everything else is reproducible.
    let _ = std::fs::remove_dir_all(&dir);
    record
}

/// Keeps what training produced: the adapter and the replayed records as
/// artifacts, then the candidate's record and the training that made it in one
/// commit.
fn keep_candidate(
    ctx: &Context,
    plan: &TrainPlan,
    replay: &Option<ReplaySample>,
    trained: Outcome,
    request: &TrainRequest,
    candidate: &str,
    regime: Regime,
) -> Result<Candidate, CampaignError> {
    let artifacts = ctx.artifacts();
    let adapter = artifacts.put_file(
        &trained.adapter,
        &ArtifactSpec::new("adapter", "brain-trainer")
            .with_extension(".safetensors")
            .with_sha256(),
    )?;
    let reported = Digest::parse(&trained.adapter_digest)
        .map_err(|e| CampaignError::Train(format!("brain's adapter digest: {e}")))?;
    if adapter.sha256.as_ref() != Some(&reported) {
        return Err(CampaignError::Train(format!(
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
            return Err(CampaignError::Train(
                "the replayed records changed while they were being kept".into(),
            ));
        }
    }
    let record_text =
        std::fs::read_to_string(&trained.training_record).map_err(io(&trained.training_record))?;
    let training_record =
        serde_json::from_str(&record_text).map_err(|source| CampaignError::Json {
            what: trained.training_record.display().to_string(),
            source,
        })?;
    let record = Candidate {
        candidate: candidate.to_string(),
        from: request.from.to_string(),
        regime,
        base: plan.base.clone(),
        parent: plan.parent.clone(),
        datasets: plan.datasets.iter().map(|d| d.id.clone()).collect(),
        replay: replay.clone(),
        adapter: artifacts.path(&adapter.digest)?,
        adapter_artifact: adapter.digest,
        adapter_digest: trained.adapter_digest,
        base_digest: trained.base_digest,
        training_record,
        steps: request.steps,
        rank: request.rank,
        base_score: trained.base_score,
        tuned_score: trained.tuned_score,
        preference: trained.preference,
        records: trained.records,
    };
    let datasets: Vec<Digest> = record.datasets.iter().map(|d| d.0.clone()).collect();
    ctx.workspace().record_candidate(
        &record.stored(),
        &record.candidate,
        &datasets,
        match regime {
            Regime::Sft => "sft",
            Regime::Dpo => "dpo",
        },
        record.parent.as_ref().map(|release| &release.0),
    )?;
    Ok(record)
}

/// What either regime's trainer produced, as a candidate records it.
struct Outcome {
    adapter: PathBuf,
    adapter_digest: String,
    base_digest: String,
    training_record: PathBuf,
    records: usize,
    base_score: Option<HeldOutScore>,
    tuned_score: Option<HeldOutScore>,
    preference: Option<PreferenceSummary>,
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
        }
    }
}

/// Draws the replay sample of every release in `release`'s lineage that
/// was trained on chat records into `dir`'s replay file; a preference
/// release's pairs are not chat records and are not replayed.
fn draw_replay(
    ctx: &Context,
    release: &ReleaseId,
    fraction: Fraction,
    dir: &Path,
) -> Result<ReplaySample, CampaignError> {
    let lineage = ctx.releases().lineage(release)?;
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
        if earlier.manifest.training.regime != Regime::Sft {
            continue;
        }
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
        std::fs::write(&path, &text).map_err(io(&path))?;
        sample.digest = Some(Digest::of(text.as_bytes()));
    }
    Ok(sample)
}

/// The candidate `id` (or a unique prefix of it) names, as trained.
pub fn load_candidate(ctx: &Context, id: &str) -> Result<Candidate, CampaignError> {
    let all = stored_candidates(ctx)?;
    let matching: Vec<&StoredCandidate> =
        all.iter().filter(|c| c.candidate.starts_with(id)).collect();
    let found = match matching.as_slice() {
        [] => {
            return Err(CampaignError::NotFound {
                what: "candidate",
                id: id.into(),
            })
        }
        [one] => *one,
        many => match many.iter().find(|c| c.candidate == id) {
            Some(exact) => *exact,
            None => {
                return Err(CampaignError::AmbiguousId {
                    what: "candidate",
                    id: id.into(),
                    matches: many.len(),
                })
            }
        },
    };
    Candidate::from_stored(ctx, found.clone())
}

fn stored_candidates(ctx: &Context) -> Result<Vec<StoredCandidate>, CampaignError> {
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
pub fn candidate_ids(ctx: &Context) -> Result<Vec<String>, CampaignError> {
    Ok(stored_candidates(ctx)?
        .into_iter()
        .map(|c| c.candidate)
        .collect())
}

/// How many candidates were trained under the state root.
pub fn candidate_count(ctx: &Context) -> Result<usize, CampaignError> {
    Ok(candidate_ids(ctx)?.len())
}
