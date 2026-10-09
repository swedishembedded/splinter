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
//! (`splinter_data::holdout`). A supervised run then sets a share of the
//! training families aside as its monitoring set ([`sizing`]), scores it
//! every few steps as it trains, exports the adapter of the evaluation with
//! the lowest monitoring loss and stops once that loss has gone a patience of
//! evaluations without improving; the step budget is a ceiling of passes
//! over the data, not a target. The curve, the step carried and what the
//! curve warns of ([`Candidate::warnings`]) are recorded with the candidate,
//! and the release gate and the exam repeat the warnings. Trained from `policy:<alias>`, a candidate
//! continues the adapter of the release the alias was resolved to - never
//! the base weights once a release exists. A supervised candidate trained
//! so also replays a seeded sample of every earlier release's chat
//! training records beside the new ones: `replay_fraction`
//! ([`DEFAULT_REPLAY_FRACTION`]) of each such release's records, drawn with
//! [`REPLAY_SEED`], from the part of its datasets that was trained on and
//! never from what it held out, so its held-out tasks stay unseen for the
//! retention check. Replayed records are never held out, so the held-out
//! score measures the new data, and they take a fixed share of the training
//! draws ([`DEFAULT_REPLAY_SHARE`]), so a large replay set cannot take the
//! steps from the new data. From a `local:` reference there is no
//! release, so nothing is replayed. A supervised run may also rehearse a
//! dataset of the base model's own answers ([`TrainRequest::rehearsal`]):
//! its records are mixed in at their own share of the draws, never held
//! out, and a monitoring share of them goes into the monitoring set beside
//! the training families', so the step selected is the one that keeps the
//! base's answers as well as it fits the new ones; the records rehearsed
//! are kept as an artifact ([`REHEARSAL_FILE`]) the leakage check reads.
//! brain's preference fine-tune trains on
//! its pairs alone, so a preference candidate replays nothing; the frozen
//! reference it is trained against - the model it continues - is what keeps
//! it close to what came before, and the gate's retention check measures
//! whether it did.

mod candidate;
mod keep;
mod rehearsal;
pub mod sizing;

pub use candidate::{
    adopt_checkpoint, candidate_count, candidate_ids, load_candidate, Candidate, Checkpoint,
};
use keep::{keep_candidate, Finished, Outcome};
use rehearsal::{prepare_rehearsal, RehearsalSplit};
pub use rehearsal::{RehearsalPlan, Rehearse, REHEARSAL_FILE};
pub use sizing::{
    auto_records_per_step, auto_steps, eval_every_for, steps_for, DEFAULT_MONITOR_SHARE,
    DEFAULT_PATIENCE, EVALUATIONS_PER_BUDGET, MAX_AUTO_STEPS, MAX_MONITOR_SHARE, MAX_PASSES,
};

use std::path::{Path, PathBuf};

use serde::Serialize;
use splinter_agent::CancelToken;
use splinter_core::digest::Digest;
use splinter_core::training::{Regime, ReplaySample, ReplaySource};
use splinter_data::holdout::{monitor_split_file, split_dataset_file};
use splinter_data::{replay_sample, Format, Fraction, StoredDataset};
use splinter_model::train::{
    fine_tune, train_preference, FineTune, PreferenceTune, Trained, TrainedPreference,
};
use splinter_model::{ModelSelection, PolicyError};

use crate::datasets::{examples_in, record_dataset_lineage, resolve_dataset};
use crate::release::probe::split_records;
use splinter_core::model_ref::ModelRef;
use splinter_core::release::ReleaseId;
use splinter_orchestrator::context::{Context, PolicyPin};
use splinter_orchestrator::error::{io, OrchestratorError};

/// The peak learning rate a run (`train` and `learn` alike) trains a LoRA
/// adapter at when none is given: the rate that moves a low-rank update in
/// the few hundred steps of a short run, which brain's own default (set for
/// long runs) does not. The plan records it, so a run says what it ran at.
pub const DEFAULT_LEARNING_RATE: f32 = 2e-4;

/// The fewest steps a run's budget is when a command names none; the budget
/// itself is passes over the data ([`sizing`]).
pub const DEFAULT_STEPS: u32 = 40;
/// LoRA rank of a new adapter when a command names none.
pub const DEFAULT_LORA_RANK: u32 = 8;
/// LoRA alpha of a new adapter per unit of rank: the update is scaled by
/// `alpha / rank`, so a scale of two holds at whatever rank is trained and a
/// rank compared with another differs in capacity alone.
pub const DEFAULT_LORA_ALPHA_PER_RANK: f32 = 2.0;
/// The AdamW weight decay of an adapter when none is given: none. An adapter
/// starts at zero and decay only pulls what it has learned back to it.
pub const DEFAULT_WEIGHT_DECAY: f32 = 0.0;

/// The LoRA alpha of a new adapter of `rank` when none is given.
#[must_use]
pub fn default_alpha(rank: u32) -> f32 {
    DEFAULT_LORA_ALPHA_PER_RANK * rank as f32
}
/// The fraction of each earlier release's training records replayed.
pub const DEFAULT_REPLAY_FRACTION: f64 = 0.25;
/// The share of a supervised run's training draws that come from the replayed
/// records when there are any. A plain union of the new records and the replay
/// lets a large replay set take most steps and starve the new data, which is
/// what the run is for; the share holds the replay to a quarter of the draws
/// however many records it holds.
pub const DEFAULT_REPLAY_SHARE: f32 = 0.25;
/// The DPO temperature of a preference fine-tune when a command names none:
/// brain's default.
pub use splinter_model::train::DEFAULT_DPO_BETA;
/// The seed of the replay draw: the same records are replayed every time.
pub const REPLAY_SEED: u64 = 0;

/// Where the adapters of a run's evaluations are written, inside its
/// directory.
const EVALUATIONS_DIR: &str = "evaluations";

/// The replayed records, inside a candidate's directory.
pub const REPLAY_FILE: &str = "replay.jsonl";

/// The monitoring set a run scores when it is made of several files: the
/// training families' monitoring records, the writer's text of those
/// families and the rehearsal's records together.
const MONITOR_ALL_FILE: &str = "monitor_all.jsonl";

/// The regime a dataset of `format` is trained by; `None` for a format
/// brain does not train.
#[must_use]
pub fn regime_of(format: Format) -> Option<Regime> {
    match format {
        Format::GenericMessagesV2 => Some(Regime::Sft),
        Format::GenericPreferenceV1 => Some(Regime::Dpo),
        Format::SplinterExportV1 => None,
        // Trained by the timeline objective, not by an adapter regime.
        Format::TimelineV1 => None,
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
    /// The peak learning rate; [`DEFAULT_LEARNING_RATE`] when `None`.
    pub learning_rate: Option<f32>,
    /// The LoRA alpha of a new adapter; [`default_alpha`] of the rank when
    /// `None`.
    pub alpha: Option<f32>,
    /// The AdamW weight decay; [`DEFAULT_WEIGHT_DECAY`] when `None`.
    pub weight_decay: Option<f32>,
    /// Records averaged into one optimizer step. `None` is one when the
    /// steps are named, else [`auto_records_per_step`] of the examples.
    pub records_per_step: Option<u32>,
    /// Steps between evaluations of the monitoring set; `None` is
    /// [`eval_every_for`] the budget, 0 monitors nothing and the candidate
    /// carries its last step.
    pub eval_every: Option<u32>,
    /// Evaluations without improvement before the run stops; `None` is
    /// [`DEFAULT_PATIENCE`], 0 runs the whole budget (the best evaluation is
    /// carried either way).
    pub patience: Option<u32>,
    /// The share of the training families set aside as the monitoring set;
    /// `None` is [`DEFAULT_MONITOR_SHARE`].
    pub monitor_share: Option<f64>,
    /// The seed of a fresh adapter's initialisation and the batch order;
    /// brain's default when `None`. Two runs that differ only here show
    /// how much of a measured difference is the draw.
    pub seed: Option<u64>,
    /// Keep the adapter of every evaluation with the candidate, so a step can
    /// be chosen on something other than the monitoring loss
    /// ([`crate::checkpoints`]); only the exported one is kept otherwise.
    pub keep_evaluations: bool,
}

/// One training request.
#[derive(Clone, Debug, Serialize)]
pub struct TrainRequest {
    /// The datasets trained on, by id or unique prefix, in order.
    pub datasets: Vec<String>,
    /// A dataset of the base's own answers rehearsed beside them; `None`
    /// rehearses nothing.
    pub rehearsal: Option<Rehearse>,
    /// What to train: `policy:<alias>` continues its release, `local:`
    /// a base and an optional adapter on it.
    pub from: ModelRef,
    /// The fraction of each earlier release's training records replayed.
    pub replay_fraction: f64,
    /// The step budget; `None` is [`auto_steps`] of the datasets' examples.
    pub steps: Option<u32>,
    /// LoRA rank of a new adapter.
    pub rank: u32,
    /// The DPO temperature, for preference datasets only;
    /// [`DEFAULT_DPO_BETA`] when `None`.
    pub beta: Option<f32>,
    /// The base's precision, the learning rate and how the run is watched.
    pub tuning: Tuning,
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
    /// The share of the training draws that come from `replay_file`:
    /// [`DEFAULT_REPLAY_SHARE`] when there is one, else `None`.
    pub replay_share: Option<f32>,
    /// The rehearsed records, when a dataset is rehearsed.
    pub rehearsal: Option<RehearsalPlan>,
    /// The step budget.
    pub steps: u32,
    /// LoRA rank of a new adapter.
    pub rank: u32,
    /// The DPO temperature; used by the preference regime only.
    pub beta: f32,
    /// The base's precision and the learning rate, every option resolved.
    pub tuning: Tuning,
    /// Steps between monitoring evaluations; 0 monitors nothing.
    pub eval_every: u32,
    /// Evaluations without improvement before the run stops; 0 never stops.
    pub patience: u32,
    /// The share of the training families set aside to monitor on.
    pub monitor_share: f64,
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
    ) -> Result<Trained, OrchestratorError>;

    /// Trains the preference adapter `plan` describes into `plan.dir`,
    /// stopping when `cancel` fires.
    fn train_preference(
        &self,
        ctx: &Context,
        plan: &TrainPlan,
        cancel: &CancelToken,
    ) -> Result<TrainedPreference, OrchestratorError>;
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
    ) -> Result<Trained, OrchestratorError> {
        let split = split_dataset_file(&combine(plan)?, &plan.dir)?;
        // The monitoring set is carved from the training half: the held-out
        // half is the gate's and the exam's evidence, and a step chosen on
        // it would be chosen on what it is then judged by.
        let monitored = (plan.eval_every > 0)
            .then(|| monitor_split_file(&split.train, &plan.dir, plan.monitor_share))
            .transpose()?;
        // The monitoring set is the training families' monitoring records,
        // the writer's text of those families and the rehearsal's records
        // together: the step selected is the one that generalises best on
        // the mixture the run trains - the writer's text included, which is
        // most of the tokens of a persona run - not on the dialogues alone.
        let monitor = match &monitored {
            Some(m) => {
                let parts: Vec<&Path> = std::iter::once(m.monitor.as_path())
                    .chain(m.monitor_text.as_deref())
                    .chain(plan.rehearsal.as_ref().and_then(|r| r.monitor.as_deref()))
                    .collect();
                Some(if parts.len() == 1 {
                    m.monitor.clone()
                } else {
                    concatenate(&parts, &plan.dir.join(MONITOR_ALL_FILE))?
                })
            }
            None => None,
        };
        let kept = plan
            .tuning
            .keep_evaluations
            .then(|| plan.dir.join(EVALUATIONS_DIR));
        let replayed: Vec<PathBuf> = plan.replay_file.iter().cloned().collect();
        let rehearsed: Vec<(PathBuf, f32)> = plan
            .rehearsal
            .iter()
            .map(|r| (r.fit.clone(), r.share))
            .collect();
        fine_tune(&FineTune {
            model_dir: &plan.base,
            train: monitored.as_ref().map_or(&split.train, |m| &m.fit),
            held_out: &split.held_out,
            held_out_text: split.held_out_text.as_deref(),
            monitor: monitor.as_deref(),
            eval_every: plan.eval_every,
            patience: plan.patience,
            attempt_dir: &plan.dir,
            steps: plan.steps,
            rank: plan.rank,
            alpha: plan
                .tuning
                .alpha
                .unwrap_or_else(|| default_alpha(plan.rank)),
            weight_decay: plan.tuning.weight_decay.unwrap_or(DEFAULT_WEIGHT_DECAY),
            replay: &replayed,
            replay_share: plan.replay_share,
            weighted_replay: &rehearsed,
            grad_accum: plan.tuning.records_per_step.unwrap_or(1),
            continue_from: plan.continue_from.as_deref(),
            cancel: Some(cancel),
            bf16_base: plan.tuning.bf16_base,
            learning_rate: plan.tuning.learning_rate,
            seed: plan.tuning.seed,
            // The records hold answers and no reasoning, so the model is
            // trained for no-think mode whatever it is later asked.
            thinking: false,
            on_step: None,
            keep_evaluations: kept.as_deref(),
        })
        .map_err(trainer_error)
    }

    fn train_preference(
        &self,
        _ctx: &Context,
        plan: &TrainPlan,
        cancel: &CancelToken,
    ) -> Result<TrainedPreference, OrchestratorError> {
        let split = split_dataset_file(&combine(plan)?, &plan.dir)?;
        train_preference(&PreferenceTune {
            model_dir: &plan.base,
            train: &split.train,
            held_out: &split.held_out,
            attempt_dir: &plan.dir,
            steps: plan.steps,
            rank: plan.rank,
            alpha: plan
                .tuning
                .alpha
                .unwrap_or_else(|| default_alpha(plan.rank)),
            beta: plan.beta,
            nll_weight: 0.0,
            grad_accum: 1,
            learning_rate: plan.tuning.learning_rate,
            continue_from: plan.continue_from.as_deref(),
            cancel: Some(cancel),
        })
        .map_err(trainer_error)
    }
}

/// `plan`'s datasets concatenated in order into one file in its directory.
fn combine(plan: &TrainPlan) -> Result<PathBuf, OrchestratorError> {
    let paths: Vec<&Path> = plan.datasets.iter().map(|d| d.path.as_path()).collect();
    concatenate(&paths, &plan.dir.join("dataset.jsonl"))
}

/// The records of `files`, in order, written to `into`; a record is a
/// non-blank line.
pub(super) fn concatenate(files: &[&Path], into: &Path) -> Result<PathBuf, OrchestratorError> {
    let mut text = String::new();
    for file in files {
        for line in std::fs::read_to_string(file)
            .map_err(io(file))?
            .lines()
            .filter(|l| !l.trim().is_empty())
        {
            text.push_str(line);
            text.push('\n');
        }
    }
    std::fs::write(into, &text).map_err(io(into))?;
    Ok(into.to_path_buf())
}

fn trainer_error(e: PolicyError) -> OrchestratorError {
    match e {
        PolicyError::Cancelled { .. } => OrchestratorError::Cancelled,
        other => OrchestratorError::Train(other.to_string()),
    }
}

/// The stored dataset `id` names and the regime brain trains it by,
/// refused when brain cannot train it.
pub fn trainable(ctx: &Context, id: &str) -> Result<(StoredDataset, Regime), OrchestratorError> {
    let dataset = resolve_dataset(ctx, id)?;
    let Some(regime) = regime_of(dataset.manifest.format) else {
        return Err(OrchestratorError::Refused(format!(
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
) -> Result<Candidate, OrchestratorError> {
    if request.datasets.is_empty() {
        return Err(OrchestratorError::Refused(
            "name at least one dataset".into(),
        ));
    }
    let fraction = Fraction::new(request.replay_fraction).map_err(|e| {
        OrchestratorError::Refused(format!("replay fraction {}: {e}", request.replay_fraction))
    })?;
    let resolved = request
        .datasets
        .iter()
        .map(|id| trainable(ctx, id))
        .collect::<Result<Vec<_>, _>>()?;
    let regime = resolved[0].1;
    if let Some((other, _)) = resolved.iter().find(|(_, r)| *r != regime) {
        return Err(OrchestratorError::Refused(format!(
            "dataset {} holds {:?} records but {} holds {:?} records; one run trains one \
             regime",
            other.id, other.manifest.objective, resolved[0].0.id, resolved[0].0.manifest.objective
        )));
    }
    let beta = match (regime, request.beta) {
        (Regime::Other, _) => {
            return Err(OrchestratorError::Refused(
                "the datasets are for a regime brain's adapter trainers do not run".into(),
            ))
        }
        (Regime::Dpo, beta) => beta.unwrap_or(DEFAULT_DPO_BETA),
        (Regime::Sft, None) => DEFAULT_DPO_BETA,
        (Regime::Sft, Some(_)) => {
            return Err(OrchestratorError::Refused(
                "beta applies to preference datasets only; these are chat datasets".into(),
            ))
        }
    };
    let datasets: Vec<StoredDataset> = resolved.into_iter().map(|(d, _)| d).collect();
    let monitor_share = request
        .tuning
        .monitor_share
        .unwrap_or(DEFAULT_MONITOR_SHARE);
    if !(monitor_share > 0.0 && monitor_share <= MAX_MONITOR_SHARE) {
        return Err(OrchestratorError::Refused(format!(
            "the monitor share {monitor_share} is not in (0, {MAX_MONITOR_SHARE}]: a run monitors \
             some of its training families and fits the rest"
        )));
    }
    // The budget and what a step averages follow the data unless the command
    // named them: a command that names its steps means optimizer steps of
    // single records, as it always did.
    let examples = datasets
        .iter()
        .map(|d| examples_in(&d.path).map_err(io(&d.path)))
        .sum::<Result<usize, _>>()?;
    let records_per_step = request.tuning.records_per_step.or_else(|| {
        request
            .steps
            .is_none()
            .then(|| auto_records_per_step(examples))
    });
    let steps = request
        .steps
        .unwrap_or_else(|| steps_for(examples, records_per_step.unwrap_or(1)));
    let eval_every = request
        .tuning
        .eval_every
        .unwrap_or_else(|| eval_every_for(steps));
    // A run with no evaluations has none to be patient about.
    let patience = if eval_every == 0 {
        0
    } else {
        request.tuning.patience.unwrap_or(DEFAULT_PATIENCE)
    };
    let ModelSelection::Local(weights) = ctx.selection(&request.from)? else {
        return Err(OrchestratorError::Refused(format!(
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
    let replay_share = replay
        .as_ref()
        .and_then(|r| r.digest.as_ref())
        .map(|_| DEFAULT_REPLAY_SHARE);
    let rehearsal = match (&request.rehearsal, regime) {
        (None, _) => None,
        (Some(_), Regime::Dpo) => {
            return Err(OrchestratorError::Refused(
                "a preference run trains on its pairs alone and rehearses nothing".into(),
            ))
        }
        (Some(_), Regime::Other) => return Err(OrchestratorError::Refused(
            "a run of an objective brain owns trains on its records alone and rehearses nothing"
                .into(),
        )),
        (Some(rehearse), Regime::Sft) => Some(prepare_rehearsal(
            ctx,
            rehearse,
            &RehearsalSplit {
                dir: &dir,
                monitored: eval_every > 0,
                monitor_share,
                replay_share,
            },
        )?),
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
        replay_share,
        rehearsal,
        steps,
        rank: request.rank,
        beta,
        // A base too large for the card at fp32 is a fact about the machine,
        // not about the command that trains it.
        tuning: Tuning {
            bf16_base: request.tuning.bf16_base || ctx.config().bf16_base,
            learning_rate: Some(
                request
                    .tuning
                    .learning_rate
                    .unwrap_or(DEFAULT_LEARNING_RATE),
            ),
            alpha: Some(request.tuning.alpha.unwrap_or(default_alpha(request.rank))),
            weight_decay: Some(request.tuning.weight_decay.unwrap_or(DEFAULT_WEIGHT_DECAY)),
            records_per_step,
            eval_every: Some(eval_every),
            patience: Some(patience),
            monitor_share: Some(monitor_share),
            ..request.tuning
        },
        eval_every,
        patience,
        monitor_share,
    };
    // A fine-tune loads its own copy of the base: the device holds no
    // other while it trains.
    ctx.release_bases();
    let trained = match regime {
        Regime::Sft => Outcome::from(trainer.train(ctx, &plan, cancel)?),
        Regime::Dpo => Outcome::from(trainer.train_preference(ctx, &plan, cancel)?),
        Regime::Other => {
            return Err(OrchestratorError::Refused(
                "a full checkpoint is not trained by the adapter trainers".into(),
            ))
        }
    };
    let record = keep_candidate(
        ctx,
        Finished {
            plan: &plan,
            replay: &replay,
            trained,
            request,
            candidate: &candidate,
            regime,
        },
    );
    // The directory was only ever a working place; the adapter and the
    // replay are kept as artifacts and everything else is reproducible.
    let _ = std::fs::remove_dir_all(&dir);
    record
}

/// Draws the replay sample of every release in `release`'s lineage that
/// was trained on chat records into `dir`'s replay file; a preference
/// release's pairs are not chat records and are not replayed.
fn draw_replay(
    ctx: &Context,
    release: &ReleaseId,
    fraction: Fraction,
    dir: &Path,
) -> Result<ReplaySample, OrchestratorError> {
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
