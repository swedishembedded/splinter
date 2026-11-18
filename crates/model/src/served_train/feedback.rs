// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems that align a model on
// judged completions through a shared training service, for its clients. If
// your team needs expertise in KTO run as a service job, you can procure our
// services by sending an email to info@swedishembedded.com.

//! KTO as a job of the brain service: the `feedback` objective of
//! `brain/train`, on `generic-feedback-v1` judged completions.
//!
//! The completions are checked by the service first (`check_dataset`, kind
//! `feedback`): one the trainer refuses is named by its line before a card is
//! asked for, and the row the run reserves is sized to the longest example.
//! The objective takes no held-out set: what it trained is judged by the
//! release gate and the exam. A base that opens a reasoning block when asked
//! trains on completions that follow an empty closed one, as it is served.

use serde_json::{Map, Value};
use splinter_core::training::FeedbackSummary;

use super::check::{check_dataset, CheckSpec, DatasetKind, MAX_BLOCK};
use super::preference::prepared;
use super::request::{set, source};
use crate::endpoint::BrainEndpoints;
use crate::error::PolicyError;
use crate::local::opens_think_block;
use crate::selection::local_model_name;
use crate::train::{candidate_adapter_id, FeedbackTune, TrainedFeedback};
use crate::train_checks::MIN_BLOCK;

/// Fine-tunes a LoRA by KTO on `request.train`'s judged completions as a job
/// of the service.
///
/// # Errors
/// A completion is refused, no artifact store is configured, or the job was
/// refused, failed or cancelled.
pub fn train_feedback_served(
    brain: &BrainEndpoints,
    request: &FeedbackTune<'_>,
) -> Result<TrainedFeedback, PolicyError> {
    let failed = |reason: String| PolicyError::Train {
        dir: request.attempt_dir.to_path_buf(),
        reason,
    };
    let base = local_model_name(request.model_dir);
    let reasoning = opens_think_block(request.model_dir);
    let train = prepared(request.train, reasoning, request.attempt_dir, "feedback")?;
    let spec = CheckSpec {
        base: Some(&base),
        kind: DatasetKind::Feedback,
        thinking: true,
        pack_answers: false,
        max_block: MAX_BLOCK,
        per_record: false,
    };
    let report = check_dataset(brain, &train, &spec)?;
    report.refuse_invalid(&train)?;
    let block = MIN_BLOCK.max(report.longest());
    let mut data = Map::new();
    set(&mut data, "train", source(brain, &train)?);
    if let Some(adapter) = request.continue_from {
        set(&mut data, "start_adapter", source(brain, adapter)?);
    }
    let mut params = Map::new();
    set(&mut params, "base", base.as_str());
    set(&mut params, "data", Value::Object(data).to_string());
    set(&mut params, "objective", "feedback");
    set(&mut params, "steps", request.steps);
    set(&mut params, "rank", request.rank);
    set(&mut params, "alpha", request.alpha);
    if let Some(beta) = request.beta {
        set(&mut params, "beta", beta);
    }
    set(&mut params, "desirable_weight", request.desirable_weight);
    set(
        &mut params,
        "undesirable_weight",
        request.undesirable_weight,
    );
    set(&mut params, "keep_reasoning", reasoning);
    set(&mut params, "grad_accum", request.grad_accum);
    set(&mut params, "seed", request.seed);
    set(&mut params, "max_block", block);
    set(
        &mut params,
        "adapter_id",
        candidate_adapter_id(&base, request.attempt_dir)?,
    );
    if let Some(lr) = request.learning_rate {
        set(&mut params, "lr", lr);
    }
    let report = super::run_action(
        brain,
        request.attempt_dir,
        request.seed,
        "train",
        &Value::Object(params),
        request.cancel,
        &mut |frame| {
            if frame["type"] == "step" {
                eprintln!("{}", super::step_line(frame));
            }
        },
    )?;
    let delivered = super::deliver(brain, request.attempt_dir, &report)?;
    let number = |name: &str| report[name].as_f64().map(|v| v as f32);
    let count =
        |name: &str| -> Result<usize, PolicyError> { super::field(&report, name).map_err(failed) };
    let summary = FeedbackSummary {
        beta: super::field(&report, "beta").map_err(failed)?,
        desirable_weight: super::field(&report, "desirable_weight").map_err(failed)?,
        undesirable_weight: super::field(&report, "undesirable_weight").map_err(failed)?,
        desirable_examples: count("desirable_examples")?,
        undesirable_examples: count("undesirable_examples")?,
        initial_loss: number("initial_loss"),
        final_loss: number("final_loss"),
        desirable_logratio: number("desirable_logratio"),
        undesirable_logratio: number("undesirable_logratio"),
    };
    Ok(TrainedFeedback {
        adapter: delivered.adapter,
        adapter_digest: delivered.adapter_digest,
        base_digest: delivered.base_digest,
        training_record: delivered.record,
        records: summary.desirable_examples + summary.undesirable_examples,
        summary,
    })
}
