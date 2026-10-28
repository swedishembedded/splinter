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
//! the run's seed ([`Tuning::seed`]), from the part of its datasets that was
//! trained on and never from what it held out, so its held-out tasks stay unseen for the
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
mod full;
mod keep;
mod monitoring;
mod rehearsal;
mod seeds;
pub mod sizing;
mod tuning;

pub use candidate::{
    adopt_checkpoint, candidate_count, candidate_ids, load_candidate, Candidate, Checkpoint,
};
pub use full::{
    train_full_candidate, train_full_recorded_as, FullCandidate, FullRequest,
    DEFAULT_FULL_LEARNING_RATE,
};
use keep::{keep_candidate, Finished, Outcome};
use rehearsal::{prepare_rehearsal, RehearsalSplit};
pub use rehearsal::{RehearsalPlan, Rehearse, REHEARSAL_FILE};
pub use seeds::{record_seed, train_recorded, train_recorded_as, SEED as TRAIN_SEED};
pub use sizing::{
    auto_records_per_step, auto_steps, eval_every_for, steps_for, DEFAULT_MONITOR_SHARE,
    DEFAULT_PATIENCE, EVALUATIONS_PER_BUDGET, MAX_AUTO_STEPS, MAX_MONITOR_SHARE, MAX_PASSES,
};
pub use splinter_model::consolidate::Checkpoint as ModelCheckpoint;
pub use tuning::Tuning;

use std::path::{Path, PathBuf};

use serde::Serialize;
use splinter_agent::CancelToken;
use splinter_core::digest::Digest;
use splinter_core::training::{Regime, ReplaySample, ReplaySource};
use splinter_data::holdout::split_dataset_file;
use splinter_data::{select_replay, Format, Fraction, ReplayPolicy, StoredDataset};
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
/// Where the adapters of a run's evaluations are written, inside its
/// directory.
const EVALUATIONS_DIR: &str = "evaluations";

/// The replayed records, inside a candidate's directory.
pub const REPLAY_FILE: &str = "replay.jsonl";

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
    /// Which of them: how the replay is spread over the capabilities the
    /// earlier records are of.
    pub replay_policy: ReplayPolicy,
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
    /// What a full fine-tune stores each parameter as, in brain's textual
    /// form (`state=int8,sr=on`); `None` is brain's default. Adapter
    /// trainers ignore it.
    pub train_precision: Option<String>,
    /// The seed the trainer is handed and the replay is drawn by:
    /// [`Tuning::seed`], or the one drawn when none was given.
    pub seed: u64,
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
    /// Whether this trainer divides its records between training and scoring
    /// by independent source text, so that a set from fewer than two texts
    /// cannot be divided: `learn` then refuses such sources before it spends
    /// a rollout on them. A trainer that holds nothing out says no.
    fn holds_out_by_group(&self) -> bool {
        false
    }

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

    /// Trains every weight of `plan.base` on the plan's records into a
    /// checkpoint directory under `plan.dir`, stopping when `cancel` fires:
    /// the consolidation of a generation. A trainer that trains adapters
    /// only refuses.
    fn train_full(
        &self,
        _ctx: &Context,
        _plan: &TrainPlan,
        _cancel: &CancelToken,
    ) -> Result<ModelCheckpoint, OrchestratorError> {
        Err(OrchestratorError::Refused(
            "this trainer trains adapters only: it cannot train every weight".into(),
        ))
    }

    /// Folds `adapter` into the weights of `base` it targets and writes the
    /// checkpoint directory to `out_dir`.
    fn merge(
        &self,
        _ctx: &Context,
        _base: &Path,
        _adapter: &Path,
        _out_dir: &Path,
    ) -> Result<ModelCheckpoint, OrchestratorError> {
        Err(OrchestratorError::Refused(
            "this trainer trains adapters only: it cannot merge one into a base".into(),
        ))
    }
}

/// Trains with brain's LoRA fine-tunes: chat fine-tuning for the
/// supervised regime, preference fine-tuning for DPO.
#[derive(Clone, Copy, Debug, Default)]
pub struct BrainTrainer;

impl Trainer for BrainTrainer {
    fn holds_out_by_group(&self) -> bool {
        true
    }

    fn train(
        &self,
        _ctx: &Context,
        plan: &TrainPlan,
        cancel: &CancelToken,
    ) -> Result<Trained, OrchestratorError> {
        let split = split_dataset_file(&combine(plan)?, &plan.dir)?;
        let watch = monitoring::plan_monitoring(plan, &split.train)?;
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
            max_block: plan.tuning.max_block,
            model_dir: &plan.base,
            train: watch.fit.as_deref().unwrap_or(&split.train),
            held_out: &split.held_out,
            held_out_text: split.held_out_text.as_deref(),
            monitor: watch.monitor.as_deref(),
            eval_every: watch.eval_every,
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
            base: plan.tuning.base_precision.unwrap_or_else(|| {
                splinter_model::train::BasePrecision::bf16_if(plan.tuning.bf16_base)
            }),
            learning_rate: plan.tuning.learning_rate,
            seed: plan.seed,
            // Records that carry the reasoning of their steps (an agent's
            // trajectory) train the model to reason before it acts, the mode
            // such an agent is served in; records of bare answers train it for
            // no-think mode, whatever it is later asked. Rendered the other
            // way, a reasoning turn cannot follow the no-think prompt and
            // brain refuses the record.
            thinking: carries_reasoning(&split.train)?,
            // Only datasets built packed are kept as one sequence per record; a
            // dialogue dataset in the same run would otherwise change shape.
            pack_answers: plan.datasets.iter().all(|d| d.manifest.packed),
            on_step: None,
            keep_evaluations: kept.as_deref(),
        })
        .map_err(trainer_error)
        .map(|mut trained| {
            trained.curve.monitoring_skipped = watch.skipped;
            trained
        })
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
            seed: plan.seed,
            continue_from: plan.continue_from.as_deref(),
            cancel: Some(cancel),
        })
        .map_err(trainer_error)
    }

    fn train_full(
        &self,
        _ctx: &Context,
        plan: &TrainPlan,
        cancel: &CancelToken,
    ) -> Result<ModelCheckpoint, OrchestratorError> {
        full::train_full(plan, cancel)
    }

    fn merge(
        &self,
        _ctx: &Context,
        base: &Path,
        adapter: &Path,
        out_dir: &Path,
    ) -> Result<ModelCheckpoint, OrchestratorError> {
        full::merge_into(base, adapter, out_dir)
    }
}

/// Whether any message of the JSON Lines chat records at `path` carries the
/// reasoning of its step (`reasoning_content`).
fn carries_reasoning(path: &Path) -> Result<bool, OrchestratorError> {
    use std::io::BufRead;
    let file = std::fs::File::open(path).map_err(io(path))?;
    for line in std::io::BufReader::new(file).lines() {
        let line = line.map_err(io(path))?;
        // Cheap test first: most lines of a dataset without reasoning never
        // parse.
        if !line.contains("\"reasoning_content\"") {
            continue;
        }
        let record: serde_json::Value = serde_json::from_str(&line).map_err(|e| {
            OrchestratorError::Refused(format!("{}: not a JSON record: {e}", path.display()))
        })?;
        let reasons = record["messages"].as_array().is_some_and(|messages| {
            messages.iter().any(|m| {
                m["reasoning_content"]
                    .as_str()
                    .is_some_and(|text| !text.trim().is_empty())
            })
        });
        if reasons {
            return Ok(true);
        }
    }
    Ok(false)
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

/// A training request resolved to what a trainer needs, with the files it
/// works in made.
pub(crate) struct Planned {
    /// What the trainer is handed.
    pub(crate) plan: TrainPlan,
    /// The replay drawn for it.
    pub(crate) replay: Option<ReplaySample>,
}

/// Resolves `request`: the datasets and their regime, the budget the data
/// asks for, the base and the adapter the reference names, the replay drawn
/// from the release it continues and the rehearsal split. Makes the plan's
/// directory.
pub(crate) fn plan_training(
    ctx: &Context,
    request: &TrainRequest,
) -> Result<Planned, OrchestratorError> {
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
    let seed = seeds::resolve(ctx, request.tuning.seed);
    let replay = match (&pin, regime) {
        (Some(pin), Regime::Sft) => Some(draw_replay(
            ctx,
            &pin.release,
            (fraction, &request.replay_policy),
            &dir,
            seed,
        )?),
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
        continue_from: if request.tuning.restart {
            None
        } else {
            weights.adapter.clone()
        },
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
            base_precision: request.tuning.base_precision,
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
            seed: Some(seed),
            ..request.tuning
        },
        train_precision: None,
        seed,
        eval_every,
        patience,
        monitor_share,
    };
    Ok(Planned { plan, replay })
}

/// Trains a candidate on `request`'s datasets with `trainer`. Work done as a
/// recorded run trains through [`train_recorded`], which also writes the seed
/// into the run record; this reports a drawn seed to the progress receiver
/// only.
pub fn train(
    ctx: &Context,
    request: &TrainRequest,
    trainer: &dyn Trainer,
    cancel: &CancelToken,
) -> Result<Candidate, OrchestratorError> {
    let Planned { plan, replay } = plan_training(ctx, request)?;
    let (candidate, dir, regime) = (plan.candidate.clone(), plan.dir.clone(), plan.regime);
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
    (fraction, policy): (Fraction, &ReplayPolicy),
    dir: &Path,
    seed: u64,
) -> Result<ReplaySample, OrchestratorError> {
    let lineage = ctx.releases().lineage(release)?;
    let mut sample = ReplaySample {
        fraction: fraction.get(),
        seed,
        sources: Vec::new(),
        records: 0,
        digest: None,
    };
    // Each release's training records, oldest release first.
    let mut earlier = Vec::new();
    for release in lineage.iter().rev() {
        if release.manifest.training.regime != Regime::Sft {
            continue;
        }
        let datasets = release.manifest.datasets.clone();
        let (trained_on, _held_out) = split_records(ctx, &datasets)?;
        earlier.push((release.id.clone(), datasets, trained_on));
    }
    // Uniform draws from each release on its own; a policy that spreads the
    // replay over capabilities draws from every release's records at once,
    // since a capability is not one release's.
    let picked: Vec<Vec<usize>> = if *policy == ReplayPolicy::Uniform {
        earlier
            .iter()
            .map(|(_, _, records)| select_replay(records, policy, fraction, seed))
            .collect()
    } else {
        let pooled: Vec<&String> = earlier.iter().flat_map(|(_, _, r)| r.iter()).collect();
        let chosen = select_replay(&pooled, policy, fraction, seed);
        let mut per_release = Vec::with_capacity(earlier.len());
        let mut start = 0;
        for (_, _, records) in &earlier {
            per_release.push(
                chosen
                    .iter()
                    .filter(|i| (start..start + records.len()).contains(*i))
                    .map(|i| i - start)
                    .collect(),
            );
            start += records.len();
        }
        per_release
    };
    let mut text = String::new();
    for ((id, datasets, trained_on), picked) in earlier.into_iter().zip(picked) {
        for &index in &picked {
            text.push_str(&trained_on[index]);
            text.push('\n');
        }
        sample.records += picked.len();
        sample.sources.push(ReplaySource {
            release: id,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn dataset(name: &str, lines: &[&str]) -> PathBuf {
        let path = std::env::temp_dir().join(format!("{name}-{}.jsonl", std::process::id()));
        std::fs::write(&path, lines.join("\n")).unwrap();
        path
    }

    /// An agent's records carry the reasoning of their steps and train the
    /// model to reason; bare answers (and an empty reasoning field) do not.
    #[test]
    fn records_with_step_reasoning_train_for_thinking_mode() {
        let bare = dataset(
            "bare",
            &[
                r#"{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a","train":true}]}"#,
                r#"{"messages":[{"role":"assistant","content":"a","reasoning_content":"  "}]}"#,
            ],
        );
        let reasoning = dataset(
            "reasoning",
            &[
                r#"{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}"#,
                r#"{"messages":[{"role":"assistant","content":"","reasoning_content":"look at main.c first"}]}"#,
            ],
        );
        assert!(!carries_reasoning(&bare).unwrap());
        assert!(carries_reasoning(&reasoning).unwrap());
        std::fs::remove_file(bare).unwrap();
        std::fs::remove_file(reasoning).unwrap();
    }
}
