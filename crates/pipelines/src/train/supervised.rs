// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gated self-improvement loops where a model
// update is adopted only on held-out evidence. If your team needs expertise
// in evaluation-gated fine-tuning, you can procure our services by sending
// an email to info@swedishembedded.com.

//! The supervised regime of a training run: the plan's chat datasets become
//! one training set and a held-out set, a share of the training families is
//! set aside to monitor on, and brain fine-tunes a LoRA on the rest - as a
//! job of the brain service when the run has endpoints
//! ([`splinter_model::served_train`]), in this process otherwise.

use splinter_agent::CancelToken;
use splinter_data::holdout::split_dataset_file;
use splinter_model::served_train::fine_tune_served;
use splinter_model::train::{BasePrecision, FineTune, Trained};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use std::path::PathBuf;

use super::records::{carries_reasoning, combine};
use super::{
    default_alpha, monitoring, trainer_error, TrainPlan, DEFAULT_WEIGHT_DECAY, EVALUATIONS_DIR,
};

/// Trains the supervised adapter `plan` describes into `plan.dir`.
pub(super) fn train(
    ctx: &Context,
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
    let request = FineTune {
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
        base: plan
            .tuning
            .base_precision
            .unwrap_or_else(|| BasePrecision::bf16_if(plan.tuning.bf16_base)),
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
        // dataset dialogue in the same run would otherwise change shape.
        pack_answers: plan.datasets.iter().all(|d| d.manifest.packed),
        on_step: None,
        keep_evaluations: kept.as_deref(),
    };
    fine_tune_served(&ctx.brain()?, &request)
        .map_err(trainer_error)
        .map(|mut trained| {
            trained.curve.monitoring_skipped = watch.skipped;
            trained
        })
}
