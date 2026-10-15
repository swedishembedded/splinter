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
use splinter_data::holdout::split_dataset_file;
use splinter_model::consolidate::{full_fine_tune, merge, Checkpoint, FullTune};
use splinter_orchestrator::error::OrchestratorError;

use super::{combine, trainer_error, TrainPlan};

/// The peak learning rate a full fine-tune trains every weight at when none
/// is given: a tenth of a LoRA's, two orders below what moves an adapter,
/// since each step moves the whole model.
pub const DEFAULT_FULL_LEARNING_RATE: f32 = 1e-5;

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
