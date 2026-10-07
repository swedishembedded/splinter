// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements self-improving coding agents whose every
// step is auditable, for its clients. If your team needs expertise in agent
// evaluation or locally operated coding agents, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Training a candidate adapter from a dataset the loop wrote.
//!
//! The trainer, its optimiser and the adapter format are brain's, reached
//! through Splinter's model adapter; this module only decides what to train
//! on (the dataset, with a share of its records held out for monitoring that
//! is never trained on), pins what the run depended on (dataset digest, base
//! model, configuration, seed) and registers the result as a candidate. The
//! candidate is not in use until `models judge` promotes it.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use splinter_sdk::model::train::{fine_tune, FineTune, StepHook, StepReport};
use splinter_sdk::vocabulary::digest::Digest;

use crate::models::register;
use crate::store::{write_json, LoopHome};

/// What a training run is configured with, recorded beside its result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Settings {
    /// Optimiser steps.
    pub steps: u32,
    /// LoRA rank.
    pub rank: u32,
    /// LoRA alpha.
    pub alpha: f32,
    /// Peak learning rate; `None` is brain's default.
    pub learning_rate: Option<f32>,
    /// Records averaged into one step.
    pub records_per_step: u32,
    /// Seed of the weights and the draws.
    pub seed: u64,
    /// Hold the frozen base at bf16.
    pub bf16_base: bool,
}

/// What a training run produced.
#[derive(Clone, Debug, Serialize)]
pub struct Candidate {
    /// The version id it was registered under.
    pub version: String,
    /// The adapter file.
    pub adapter: PathBuf,
    /// Its digest.
    pub adapter_digest: String,
    /// The digest of the dataset trained on.
    pub dataset_digest: String,
    /// Records trained on.
    pub records: usize,
    /// Held-out loss of the base model, where it was measured.
    pub base_loss: Option<f32>,
    /// Held-out loss of the tuned model, where it was measured.
    pub tuned_loss: Option<f32>,
    /// Where brain wrote its record of the run.
    pub training_record: PathBuf,
}

/// Trains an adapter on `dataset` over the base model at `base_dir` (named
/// `base_ref` in the registry) and registers it as a candidate.
pub fn train(
    home: &LoopHome,
    dataset: &Path,
    base_dir: &Path,
    base_ref: &str,
    settings: &Settings,
) -> Result<Candidate> {
    let bytes = std::fs::read(dataset).with_context(|| format!("reading {}", dataset.display()))?;
    let dataset_digest = Digest::sha256_of(&bytes).to_string();
    let attempt = home
        .root()
        .join("models")
        .join(format!(
            "training-{}",
            &dataset_digest[dataset_digest.len() - 12..]
        ))
        .join(format!("seed-{}", settings.seed));
    std::fs::create_dir_all(&attempt)?;
    let split = splinter_sdk::data::holdout::split_dataset_file(dataset, &attempt)
        .with_context(|| format!("splitting {}", dataset.display()))?;
    let report = |r: &StepReport| eprintln!("step {}/{} loss {:.4}", r.step, r.steps, r.loss);
    let request = FineTune {
        model_dir: base_dir,
        train: &split.train,
        held_out: &split.held_out,
        held_out_text: None,
        monitor: None,
        eval_every: 0,
        patience: 0,
        attempt_dir: &attempt,
        steps: settings.steps,
        rank: settings.rank,
        alpha: settings.alpha,
        weight_decay: 0.0,
        replay: &[],
        replay_share: None,
        weighted_replay: &[],
        grad_accum: settings.records_per_step,
        continue_from: None,
        cancel: None,
        bf16_base: settings.bf16_base,
        learning_rate: settings.learning_rate,
        seed: Some(settings.seed),
        // The records hold tool calls and no reasoning: trained for no-think mode.
        thinking: false,
        on_step: Some(StepHook(&report)),
        keep_evaluations: None,
    };
    let trained = fine_tune(&request).map_err(|e| anyhow::anyhow!("fine-tune failed: {e}"))?;
    write_json(&attempt.join("settings.json"), settings)?;
    let version = register(
        home,
        base_ref,
        &trained.adapter,
        &trained.adapter_digest,
        &dataset_digest,
        &trained.training_record,
    )?;
    Ok(Candidate {
        version,
        adapter: trained.adapter,
        adapter_digest: trained.adapter_digest,
        dataset_digest,
        records: trained.records,
        base_loss: trained.base.loss,
        tuned_loss: trained.tuned.loss,
        training_record: trained.training_record,
    })
}
