// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements consolidating what a model learned into its
// own weights, for its clients. If your team needs expertise in continual
// learning that keeps accumulating capabilities without forgetting, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Brain's full-parameter fine-tune and adapter merge, as the trainer seam
//! offers them ([`super::Trainer::train_full`], [`super::Trainer::merge`]).

use std::path::{Path, PathBuf};

use splinter_agent::CancelToken;
use splinter_core::model_ref::ModelRef;
use splinter_core::training::Regime;
use splinter_data::holdout::{holdout_split_records, split_dataset_file};
use splinter_data::{select_replay, Fraction, ReplayPolicy};
use splinter_model::consolidate::{full_fine_tune, merge, Checkpoint, FullTune};
use splinter_model::ModelSelection;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::{io, OrchestratorError};
use splinter_orchestrator::runs::Recorder;

use super::{
    combine, plan_training, record_seed, trainable, trainer_error, TrainPlan, TrainRequest,
    Trainer, Tuning, DEFAULT_LORA_RANK, DEFAULT_REPLAY_SHARE, REPLAY_FILE,
};
use crate::release::arm_on;

/// The peak learning rate a full fine-tune trains every weight at when none
/// is given: a tenth of a LoRA's, two orders below what moves an adapter,
/// since each step moves the whole model.
pub const DEFAULT_FULL_LEARNING_RATE: f32 = 1e-5;

/// A full fine-tune of a policy: every weight, from a checkpoint, on one
/// round's records beside a replay of the rounds before it.
#[derive(Clone, Debug)]
pub struct FullRequest {
    /// The datasets trained on, by id or unique prefix; the newest records
    /// of the last are held out, as in every training.
    pub datasets: Vec<String>,
    /// Datasets of rounds already taken into `from`, replayed beside
    /// `datasets` so the new records do not overwrite them.
    pub replay_datasets: Vec<String>,
    /// The fraction of each replayed dataset's trained-on records drawn;
    /// they take [`DEFAULT_REPLAY_SHARE`] of the training draws.
    pub replay_fraction: f64,
    /// The checkpoint every weight starts from. It must be a plain base: an
    /// adapter on it would be dropped by a full fine-tune.
    pub from: ModelRef,
    /// The step budget; sized from the datasets when `None`.
    pub steps: Option<u32>,
    /// The learning rate ([`DEFAULT_FULL_LEARNING_RATE`] when `None`), the
    /// seed and how the run is sized.
    pub tuning: Tuning,
    /// brain's textual training precision (`state=int8,sr=on`); `None` is
    /// brain's default.
    pub train_precision: Option<String>,
}

/// The checkpoint a full fine-tune wrote. It is GBs: whoever holds it
/// removes it with [`Self::discard`] once nothing needs it.
#[derive(Debug)]
pub struct FullCandidate {
    /// The candidate's id.
    pub id: String,
    /// What brain wrote.
    pub checkpoint: Checkpoint,
    /// The checkpoint as a model: `local:<checkpoint directory>`.
    pub model: ModelRef,
    /// Records in the datasets trained on.
    pub records: usize,
    /// Records replayed beside them.
    pub replayed_records: usize,
    /// The training run's working directory, which holds the checkpoint.
    dir: PathBuf,
}

impl FullCandidate {
    /// Removes the checkpoint and everything the run wrote beside it.
    pub fn discard(self) {
        let _ = std::fs::remove_dir_all(&self.dir);
        // A trainer may write the checkpoint outside the run's directory.
        let _ = std::fs::remove_dir_all(&self.checkpoint.dir);
    }
}

/// [`train_full_candidate`] as the work of the recorded run `run`: the
/// training seed is recorded under `name`.
pub fn train_full_recorded_as(
    ctx: &Context,
    run: &mut Recorder<'_>,
    name: &str,
    request: &FullRequest,
    trainer: &dyn Trainer,
) -> Result<FullCandidate, OrchestratorError> {
    let seeded = FullRequest {
        tuning: record_seed(run, name, request.tuning)?,
        ..request.clone()
    };
    train_full_candidate(ctx, &seeded, trainer, &run.cancel_token())
}

/// Trains every weight of `request.from` on the datasets, with `trainer`.
/// The checkpoint is left where the trainer wrote it; a failed run leaves
/// nothing behind.
pub fn train_full_candidate(
    ctx: &Context,
    request: &FullRequest,
    trainer: &dyn Trainer,
    cancel: &CancelToken,
) -> Result<FullCandidate, OrchestratorError> {
    if let ModelSelection::Local(weights) = ctx.selection(&request.from)? {
        if weights.adapter.is_some() {
            return Err(OrchestratorError::Refused(format!(
                "{} is a base with an adapter on it: a full fine-tune trains the base's weights and \
                 would drop the adapter",
                request.from
            )));
        }
    }
    let mut plan = plan_training(
        ctx,
        &TrainRequest {
            datasets: request.datasets.clone(),
            rehearsal: None,
            from: request.from.clone(),
            // The replay a full fine-tune carries is of the rounds before it,
            // drawn below; there is no release lineage to draw from.
            replay_fraction: 0.0,
            replay_policy: ReplayPolicy::default(),
            steps: request.steps,
            rank: DEFAULT_LORA_RANK,
            beta: None,
            tuning: Tuning {
                learning_rate: Some(
                    request
                        .tuning
                        .learning_rate
                        .unwrap_or(DEFAULT_FULL_LEARNING_RATE),
                ),
                // No adapter is monitored and exported: the checkpoint is the
                // run's last step.
                eval_every: Some(0),
                ..request.tuning
            },
        },
    )?
    .plan;
    let abandon = |plan: &TrainPlan, e: OrchestratorError| {
        let _ = std::fs::remove_dir_all(&plan.dir);
        e
    };
    if plan.regime != Regime::Sft {
        let e = OrchestratorError::Refused(
            "a full fine-tune here trains chat records; these are preference pairs".into(),
        );
        return Err(abandon(&plan, e));
    }
    let replayed_records = match replay_of(ctx, request, &plan) {
        Ok(drawn) => drawn,
        Err(e) => return Err(abandon(&plan, e)),
    };
    if replayed_records > 0 {
        plan.replay_file = Some(plan.dir.join(REPLAY_FILE));
        plan.replay_share = Some(DEFAULT_REPLAY_SHARE);
    }
    plan.train_precision = request.train_precision.clone();
    // A full fine-tune loads its own copy of the base: the device holds no
    // other while it trains.
    ctx.release_bases();
    let checkpoint = trainer
        .train_full(ctx, &plan, cancel)
        .map_err(|e| abandon(&plan, e))?;
    Ok(FullCandidate {
        id: plan.candidate.clone(),
        model: arm_on(ctx.config(), &checkpoint.dir, None),
        records: plan
            .datasets
            .iter()
            .map(|d| d.manifest.counts.records)
            .sum(),
        replayed_records,
        checkpoint,
        dir: plan.dir,
    })
}

/// Writes the replay of `request.replay_datasets` into the plan's replay
/// file: `replay_fraction` of each dataset's trained-on records (never what
/// its own training held out), drawn with the run's seed. Returns how many
/// records were drawn; no file is written for none.
fn replay_of(
    ctx: &Context,
    request: &FullRequest,
    plan: &TrainPlan,
) -> Result<usize, OrchestratorError> {
    let fraction = Fraction::new(request.replay_fraction).map_err(|e| {
        OrchestratorError::Refused(format!("replay fraction {}: {e}", request.replay_fraction))
    })?;
    let mut text = String::new();
    let mut drawn = 0;
    for id in &request.replay_datasets {
        let (dataset, regime) = trainable(ctx, id)?;
        if regime != Regime::Sft {
            return Err(OrchestratorError::Refused(format!(
                "dataset {} holds preference pairs, which are not replayed into a full fine-tune",
                dataset.id
            )));
        }
        let contents = std::fs::read_to_string(&dataset.path).map_err(io(&dataset.path))?;
        let records: Vec<&str> = contents
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .collect();
        let trained_on: Vec<&str> = match holdout_split_records(&records) {
            Some((train, _)) => train.into_iter().copied().collect(),
            None => records,
        };
        for index in select_replay(&trained_on, &ReplayPolicy::Uniform, fraction, plan.seed) {
            text.push_str(trained_on[index]);
            text.push('\n');
            drawn += 1;
        }
    }
    if drawn > 0 {
        let path = plan.dir.join(REPLAY_FILE);
        std::fs::write(&path, text).map_err(io(&path))?;
    }
    Ok(drawn)
}

/// Trains every weight of `plan.base` on the plan's records: the trained
/// side of its datasets, with the rehearsal at its share. The stopping
/// paraphrases the datasets hold out are never trained on.
pub(super) fn train_full(
    plan: &TrainPlan,
    cancel: &CancelToken,
) -> Result<Checkpoint, OrchestratorError> {
    let split = split_dataset_file(&combine(plan)?, &plan.dir)?;
    let replayed: Vec<PathBuf> = plan.replay_file.iter().cloned().collect();
    let rehearsed: Vec<(PathBuf, f32)> = plan
        .rehearsal
        .iter()
        .map(|r| (r.fit.clone(), r.share))
        .collect();
    full_fine_tune(&FullTune {
        model_dir: &plan.base,
        train: &split.train,
        replay: &replayed,
        replay_share: plan.replay_share,
        weighted_replay: &rehearsed,
        grad_accum: plan.tuning.records_per_step.unwrap_or(1),
        attempt_dir: &plan.dir,
        steps: plan.steps,
        learning_rate: plan.tuning.learning_rate,
        weight_decay: plan.tuning.weight_decay.unwrap_or(0.0),
        train_precision: plan.train_precision.as_deref(),
        seed: plan.seed,
        // The records hold answers and no reasoning: trained for no-think
        // mode, as the adapter trainer does.
        thinking: false,
        cancel: Some(cancel),
        on_step: None,
    })
    .map_err(trainer_error)
}

/// Folds `adapter` into `base` and writes the checkpoint to `out_dir`.
pub(super) fn merge_into(
    base: &Path,
    adapter: &Path,
    out_dir: &Path,
) -> Result<Checkpoint, OrchestratorError> {
    merge(base, adapter, out_dir).map_err(trainer_error)
}
