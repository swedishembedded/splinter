// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! One LoRA fine-tune on verified demonstrations, through brain's trainer.
//!
//! This is the sample's choice of how to call the one trainer Splinter has
//! ([`splinter_sdk::model::train::fine_tune`]), not a second one. The choices
//! that matter for this sample:
//!
//! * **no-think mode, one example per answer**: the dialogue's every
//!   supervised assistant turn is trained as the model is asked for it, the
//!   context holding the earlier turns the way the serving renderer renders
//!   them;
//! * **a monitored run**: the held-out demonstrations (wording the model was
//!   not trained on) are scored every few steps, the run stops when that loss
//!   stops improving, and the adapter exported is the best evaluation's. With
//!   so few distinct demonstrations the way to memorise is to train long; the
//!   curve is what shows whether it did;
//! * **an explicit seed**: there is no default, so two runs that differ only
//!   in it can say how much of a result is the draw.
//!
//! Swedish Embedded AB implements closed-loop learning systems - agents that
//! improve from their own verified experience rather than from hand-written
//! training data - for its clients. If your team needs expertise in agent
//! training loops, sample design, or promotion gating for small models,
//! you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::path::{Path, PathBuf};

use splinter_sdk::model::train::{fine_tune, BasePrecision, FineTune, StepHook, StepReport};

/// What to train, on what, and with which recipe.
#[derive(Clone, Debug)]
pub struct TrainRequest {
    /// The base checkpoint directory.
    pub base: PathBuf,
    /// `generic-messages-v2` records to train on.
    pub train: PathBuf,
    /// Records scored before and after and as the run trains; never trained on.
    pub held_out: PathBuf,
    /// Where the adapter and brain's training record land.
    pub attempt_dir: PathBuf,
    /// The optimizer step budget.
    pub steps: u32,
    /// LoRA rank.
    pub rank: u32,
    /// LoRA alpha.
    pub alpha: f32,
    /// Peak learning rate; brain's default when `None`.
    pub learning_rate: Option<f32>,
    /// Records averaged into one optimizer step.
    pub grad_accum: u32,
    /// Seed of the adapter's initialisation and the batch order.
    pub seed: u64,
    /// Steps between held-out evaluations.
    pub eval_every: u32,
    /// Evaluations without improvement before the run stops.
    pub patience: u32,
}

/// What a fine-tune produced, as a JSON document a report can carry.
#[derive(Clone, Debug)]
pub struct TrainSummary {
    /// The adapter file.
    pub adapter: PathBuf,
    /// `sha256:<hex>` of the adapter file, as the server lists it.
    pub adapter_digest: String,
    /// The report: scores before and after, the curve, the recipe.
    pub document: serde_json::Value,
}

/// Fine-tune, narrating each optimizer step to `narrate`.
pub fn train(
    request: &TrainRequest,
    narrate: &dyn Fn(&StepReport),
) -> anyhow::Result<TrainSummary> {
    std::fs::create_dir_all(&request.attempt_dir)?;
    let trained = fine_tune(&FineTune {
        model_dir: &request.base,
        train: &request.train,
        held_out: &request.held_out,
        held_out_text: None,
        monitor: Some(&request.held_out),
        eval_every: request.eval_every,
        patience: request.patience,
        attempt_dir: &request.attempt_dir,
        steps: request.steps,
        rank: request.rank,
        alpha: request.alpha,
        replay: &[],
        replay_share: None,
        weighted_replay: &[],
        grad_accum: request.grad_accum,
        continue_from: None,
        cancel: None,
        base: BasePrecision::Fp32,
        learning_rate: request.learning_rate,
        weight_decay: 0.0,
        seed: request.seed,
        max_block: None,
        thinking: false,
        pack_answers: false,
        on_step: Some(StepHook(narrate)),
        keep_evaluations: None,
    })?;

    let score = |s: &splinter_sdk::vocabulary::training::HeldOutScore| {
        serde_json::json!({
            "loss": s.loss, "token_accuracy": s.token_accuracy,
            "positions": s.positions, "records": s.records,
        })
    };
    let document = serde_json::json!({
        "recipe": {
            "steps": request.steps, "rank": request.rank, "alpha": request.alpha,
            "learning_rate": request.learning_rate, "grad_accum": request.grad_accum,
            "seed": request.seed, "eval_every": request.eval_every,
            "patience": request.patience, "thinking": false,
        },
        "adapter": trained.adapter,
        "adapter_digest": trained.adapter_digest,
        "base_digest": trained.base_digest,
        "records": trained.records,
        "block": trained.block,
        "held_out_before": score(&trained.base),
        "held_out_after": score(&trained.tuned),
        "curve": trained.curve,
    });
    Ok(TrainSummary {
        adapter: trained.adapter,
        adapter_digest: trained.adapter_digest,
        document,
    })
}

/// Where a run's summary is written.
pub fn summary_path(attempt_dir: &Path) -> PathBuf {
    attempt_dir.join("summary.json")
}
