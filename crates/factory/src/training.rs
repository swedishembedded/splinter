// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fine-tuning pipelines whose interrupted
// runs resume and whose finished candidates are served automatically, for
// its clients. If your team needs expertise in model training
// infrastructure, you can procure our services by sending an email to
// info@swedishembedded.com.

//! What trains and serves candidates: the seam between the controller and
//! the training engine.
//!
//! [`Trainer::train`] trains one variant from a dataset product's files;
//! [`Trainer::serve`] makes a model servable and returns the id the
//! endpoint lists it under, after the endpoint has listed it (the load
//! check). [`BrainTrainer`] is the production trainer: brain's training jobs
//! and adapter serving through its API.

use std::path::{Path, PathBuf};

use serde_json::Value;
use splinter_core::digest::Digest;
use splinter_model::endpoint::BrainEndpoints;
use splinter_model::served_train::{check_dataset, fine_tune_served, CheckSpec};
use splinter_model::train::{BasePrecision, FineTune};
use splinter_model::LocalWeights;

use crate::recipe::ModelRef;

/// Why a trainer refused or failed.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct TrainError(pub String);

/// One variant to train.
#[derive(Clone, Debug)]
pub struct TrainRequest<'a> {
    /// The base model.
    pub base: &'a ModelRef,
    /// The method (`lora`).
    pub method: &'a str,
    /// The trainer's own parameters, as the recipe wrote them.
    pub params: &'a Value,
    /// The training records.
    pub train: &'a Path,
    /// The validation records (may be empty).
    pub validation: &'a Path,
    /// The seed.
    pub seed: u64,
    /// Train reasoning (the dataset's `thinking`).
    pub thinking: bool,
    /// The run's durable identity at the trainer: asked again under it, an
    /// interrupted run continues and a completed one is answered from its
    /// record.
    pub resume_key: &'a str,
    /// Where the trained files go; its name tags the candidate.
    pub out: &'a Path,
}

/// What training produced.
#[derive(Clone, Debug, PartialEq)]
pub struct Trained {
    /// The trained files, in `out`.
    pub files: Vec<PathBuf>,
    /// The adapter, for an adapter.
    pub adapter: Option<PathBuf>,
    /// The parameters as the trainer took them.
    pub params: Value,
    /// The trainer's report.
    pub report: Value,
}

/// What trains and serves.
pub trait Trainer: Send + Sync {
    /// The trainer's identity, part of every training stage's.
    fn identity(&self) -> String;

    /// The length in the target's tokens of each line of a chat dataset.
    ///
    /// # Errors
    /// The target is unknown, or a line is refused.
    fn measure(&self, target: &ModelRef, lines: &[String]) -> Result<Vec<u64>, TrainError>;

    /// Train one variant.
    ///
    /// # Errors
    /// The parameters are refused or the run failed.
    fn train(&self, request: &TrainRequest<'_>) -> Result<Trained, TrainError>;

    /// Serve `base` with `adapter` (or the base alone) and return the id the
    /// endpoint lists it under, once listed.
    ///
    /// # Errors
    /// The endpoint does not serve it.
    fn serve(&self, base: &ModelRef, adapter: Option<&Path>) -> Result<String, TrainError>;
}

/// The LoRA parameters brain's fine-tune takes, from a recipe's `params`.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct LoraParams {
    /// Optimizer steps.
    pub steps: u32,
    /// The adapter's rank.
    pub rank: u32,
    /// LoRA alpha (default: twice the rank).
    #[serde(default)]
    pub alpha: Option<f32>,
    /// Records per optimizer step.
    #[serde(default = "one")]
    pub grad_accum: u32,
    /// The peak learning rate (brain's default when absent).
    #[serde(default)]
    pub learning_rate: Option<f32>,
    /// AdamW weight decay.
    #[serde(default)]
    pub weight_decay: f32,
    /// The longest training row in tokens.
    #[serde(default)]
    pub max_block: Option<u32>,
}

fn one() -> u32 {
    1
}

/// brain, reached through its API.
pub struct BrainTrainer {
    endpoints: BrainEndpoints,
    /// The directory base names are read under (`<models>/<org>/<name>`);
    /// only the names reach brain.
    models: PathBuf,
}

impl BrainTrainer {
    /// A trainer over `endpoints`, which need an artifact store and a way
    /// to offer adapters (a directory or manifest the endpoint follows).
    #[must_use]
    pub fn new(endpoints: BrainEndpoints, models: PathBuf) -> Self {
        Self { endpoints, models }
    }

    fn base_dir(&self, base: &ModelRef) -> PathBuf {
        self.models.join(&base.model)
    }
}

fn error(e: impl ToString) -> TrainError {
    TrainError(e.to_string())
}

impl Trainer for BrainTrainer {
    fn identity(&self) -> String {
        "brain served-train".into()
    }

    fn measure(&self, target: &ModelRef, lines: &[String]) -> Result<Vec<u64>, TrainError> {
        let dir = tempfile::tempdir().map_err(error)?;
        let path = dir.path().join("measure.jsonl");
        std::fs::write(&path, lines.join("\n")).map_err(error)?;
        let report = check_dataset(
            &self.endpoints,
            &path,
            &CheckSpec::chat(&target.model, false, true),
        )
        .map_err(error)?;
        (0..lines.len())
            .map(|line| {
                report
                    .tokens_at(line)
                    .map(|t| t as u64)
                    .ok_or_else(|| TrainError(format!("brain gave no length for record {line}")))
            })
            .collect()
    }

    fn train(&self, request: &TrainRequest<'_>) -> Result<Trained, TrainError> {
        if request.method != "lora" {
            return Err(TrainError(format!(
                "brain trainer: method {} is not supported here (lora)",
                request.method
            )));
        }
        let params: LoraParams = serde_json::from_value(request.params.clone())
            .map_err(|e| TrainError(format!("lora params: {e}")))?;
        let validation_empty = std::fs::metadata(request.validation)
            .map(|m| m.len() == 0)
            .unwrap_or(true);
        let held_out = if validation_empty {
            request.train
        } else {
            request.validation
        };
        let model_dir = self.base_dir(request.base);
        let tune = FineTune {
            model_dir: &model_dir,
            train: request.train,
            held_out,
            held_out_text: None,
            monitor: None,
            eval_every: 0,
            patience: 0,
            attempt_dir: request.out,
            steps: params.steps,
            rank: params.rank,
            alpha: params.alpha.unwrap_or(2.0 * params.rank as f32),
            replay: &[],
            replay_share: None,
            weighted_replay: &[],
            grad_accum: params.grad_accum,
            continue_from: None,
            cancel: None,
            base: BasePrecision::Bf16,
            learning_rate: params.learning_rate,
            weight_decay: params.weight_decay,
            seed: request.seed,
            max_block: params.max_block,
            thinking: request.thinking,
            pack_answers: true,
            on_step: None,
            keep_evaluations: None,
            resume_key: Some(request.resume_key),
        };
        let trained = fine_tune_served(&self.endpoints, &tune).map_err(error)?;
        let report = serde_json::json!({
            "records": trained.records, "block": trained.block,
            "base_score": trained.base, "tuned_score": trained.tuned, "curve": trained.curve,
            "held_out": if validation_empty { "train" } else { "validation" },
            "adapter_sha256": trained.adapter_digest, "base_digest": trained.base_digest,
        });
        Ok(Trained {
            files: vec![trained.adapter.clone(), trained.training_record],
            adapter: Some(trained.adapter),
            params: serde_json::to_value(&params).map_err(error)?,
            report,
        })
    }

    fn serve(&self, base: &ModelRef, adapter: Option<&Path>) -> Result<String, TrainError> {
        let weights = LocalWeights {
            base: self.base_dir(base),
            adapter: adapter.map(Path::to_path_buf),
            context_tokens: None,
        };
        let ids = self.endpoints.locate(&weights).map_err(error)?;
        ids.into_iter()
            .next()
            .ok_or_else(|| TrainError("brain lists no endpoint".into()))
    }
}

/// The SHA-256 of a file, as candidates pin their artifacts.
///
/// # Errors
/// It cannot be read.
pub fn sha256(path: &Path) -> std::io::Result<Digest> {
    Digest::sha256_of_reader(std::fs::File::open(path)?)
}
