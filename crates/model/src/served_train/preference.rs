// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems that align a model on
// preference pairs through a shared training service, for its clients. If
// your team needs expertise in preference optimisation run as a service job,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! Direct preference optimisation as a job of the brain service: the `dpo`
//! objective of `brain/train`.
//!
//! The pairs are checked by the service first (`check_dataset`, kind
//! `preference`): a pair the trainer refuses is named by its line before a
//! card is asked for, and the row the run reserves is sized to the longest
//! candidate. A base that opens a reasoning block when asked trains on
//! candidates that follow an empty closed one, as it is served, and keeps
//! that reasoning in the trained text.

use std::path::{Path, PathBuf};

use serde_json::{Map, Value};
use splinter_core::training::PreferenceScore;

use super::check::{check_dataset, CheckSpec, DatasetKind, MAX_BLOCK};
use super::request::{set, source};
use crate::endpoint::BrainEndpoints;
use crate::error::PolicyError;
use crate::local::{closed_think_pairs, opens_think_block};
use crate::selection::local_model_name;
use crate::train::{candidate_adapter_id, PreferenceTune, TrainedPreference};
use crate::train_checks::MIN_BLOCK;

/// The DPO temperature `beta` a preference fine-tune uses when its caller
/// names none: brain's default for the objective.
pub const DEFAULT_DPO_BETA: f32 = 0.1;

/// Fine-tunes a LoRA by DPO on `request.train`'s pairs as a job of the
/// service, and scores it against its reference on the held-out pairs.
///
/// # Errors
/// A pair is refused, no artifact store is configured, or the job was
/// refused, failed or cancelled.
pub fn train_preference_served(
    brain: &BrainEndpoints,
    request: &PreferenceTune<'_>,
) -> Result<TrainedPreference, PolicyError> {
    let failed = |reason: String| PolicyError::Train {
        dir: request.attempt_dir.to_path_buf(),
        reason,
    };
    let base = local_model_name(request.model_dir);
    let reasoning = opens_think_block(request.model_dir);
    let train = prepared(request.train, reasoning, request.attempt_dir, "pairs")?;
    let held_out = prepared(
        request.held_out,
        reasoning,
        request.attempt_dir,
        "held_out_pairs",
    )?;
    let spec = CheckSpec {
        base: Some(&base),
        kind: DatasetKind::Preference,
        thinking: true,
        pack_answers: false,
        max_block: MAX_BLOCK,
        per_record: false,
    };
    let mut block = MIN_BLOCK;
    let mut pairs = 0usize;
    for dataset in [&train, &held_out] {
        let report = check_dataset(brain, dataset, &spec)?;
        report.refuse_invalid(dataset)?;
        pairs += report.valid_records;
        block = block.max(report.longest());
    }
    let mut data = Map::new();
    set(&mut data, "train", source(brain, &train)?);
    set(&mut data, "held_out", source(brain, &held_out)?);
    if let Some(adapter) = request.continue_from {
        set(&mut data, "start_adapter", source(brain, adapter)?);
    }
    let mut params = Map::new();
    set(&mut params, "base", base.as_str());
    set(&mut params, "data", Value::Object(data).to_string());
    set(&mut params, "objective", "preference");
    set(&mut params, "loss", "dpo");
    set(&mut params, "steps", request.steps);
    set(&mut params, "rank", request.rank);
    set(&mut params, "alpha", request.alpha);
    set(&mut params, "beta", request.beta);
    set(&mut params, "nll_weight", request.nll_weight);
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
    let score = |name: &str| -> Result<Option<PreferenceScore>, PolicyError> {
        serde_json::from_value(report[name].clone())
            .map_err(|e| failed(format!("the service's {name}: {e}")))
    };
    Ok(TrainedPreference {
        adapter: delivered.adapter,
        adapter_digest: delivered.adapter_digest,
        base_digest: delivered.base_digest,
        training_record: delivered.record,
        records: pairs,
        block: super::field(&report, "block").map_err(failed)?,
        beta: super::field(&report, "beta").map_err(failed)?,
        reference_adapter: report["trained_from"].as_str().map(str::to_string),
        train_score: score("train_score")?,
        held_out_score: score("held_out_score")?,
    })
}

/// `path` as the run trains it: unchanged, or, for a base that opens a
/// reasoning block, rewritten so each candidate follows an empty closed one.
fn prepared(
    path: &Path,
    reasoning: bool,
    attempt_dir: &Path,
    name: &str,
) -> Result<PathBuf, PolicyError> {
    if !reasoning {
        return Ok(path.to_path_buf());
    }
    let io = |source| PolicyError::Io {
        path: path.to_path_buf(),
        source,
    };
    let text = std::fs::read_to_string(path).map_err(io)?;
    let rewritten = closed_think_pairs(&text).map_err(|e| PolicyError::Dataset {
        path: path.to_path_buf(),
        reason: e.to_string(),
    })?;
    std::fs::create_dir_all(attempt_dir).map_err(io)?;
    let dest = attempt_dir.join(format!("{name}.think.jsonl"));
    std::fs::write(&dest, rewritten).map_err(io)?;
    Ok(dest)
}
