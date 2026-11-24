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
use splinter_model::policy_gradient::{Behaviour, PolicyGradientOptions, PolicyGradientTune};
use splinter_model::served_train::{
    check_dataset, fine_tune_served, train_feedback_served, train_policy_gradient_served,
    train_preference_served, CheckSpec, DEFAULT_DPO_BETA,
};
use splinter_model::train::{BasePrecision, FeedbackTune, FineTune, PairLoss, PreferenceTune};
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
    /// The dataset's objective (`sft`, `cpt`, `preference`, `feedback`,
    /// `policy-gradient`): the regime it is trained by.
    pub objective: &'a str,
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
    /// The adapter the candidate continues (a champion's), if it continues one.
    pub start: Option<&'a Path>,
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

    /// A variant's parameters as this trainer will train them: validated,
    /// with every default the trainer would apply written in, so a change of
    /// a default is a change of every stage identity it affects.
    ///
    /// # Errors
    /// The method or a parameter is refused.
    fn resolve(&self, method: &str, params: &Value) -> Result<Value, TrainError>;

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
    /// How the frozen base is held while the adapter trains (`fp32`, brain's
    /// default; `bf16`, `int8`, `nf4`, `fp8` to fit a larger base on a card;
    /// a CPU backend trains `fp32` only).
    #[serde(default)]
    pub base_precision: Option<String>,
    /// For pairs: the pair loss (`dpo`, `ipo`, `simpo`, `orpo`; `dpo` when
    /// absent).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loss: Option<String>,
    /// For pairs: the loss's beta (brain's default for the loss when absent).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub beta: Option<f32>,
    /// For pairs: the weight of the chosen answer's likelihood beside the
    /// pair loss, which keeps what supervised training taught (0 when absent).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nll_weight: Option<f32>,
    /// For pairs, SimPO only: the target margin.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gamma: Option<f32>,
    /// For rollouts: where the importance ratio's behaviour log-probabilities come from
    /// (`recorded`, the default, or `start`: the policy the run starts from, exact only when the
    /// rollouts were sampled from it).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub behaviour: Option<String>,
    /// Steps between evaluations of the validation records, which then
    /// pick the step the adapter is exported at; 0 evaluates nothing and
    /// exports the last step.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub eval_every: u32,
    /// Evaluations without improvement before the run stops; 0 runs the
    /// whole budget (the best evaluation is exported either way).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub patience: u32,
}

fn one() -> u32 {
    1
}

fn is_zero(n: &u32) -> bool {
    *n == 0
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

    /// The held-out records of `request`: its validation file, else its
    /// training file when it held none out.
    fn held_out<'r>(request: &TrainRequest<'r>) -> &'r Path {
        let empty = std::fs::metadata(request.validation)
            .map(|m| m.len() == 0)
            .unwrap_or(true);
        if empty {
            request.train
        } else {
            request.validation
        }
    }

    fn precision(params: &LoraParams) -> Result<BasePrecision, TrainError> {
        match &params.base_precision {
            Some(text) => text
                .parse::<BasePrecision>()
                .map_err(|e| TrainError(format!("lora params: base_precision: {e}"))),
            None => Ok(BasePrecision::default()),
        }
    }

    fn trained(
        params: &LoraParams,
        adapter: PathBuf,
        record: PathBuf,
        report: Value,
    ) -> Result<Trained, TrainError> {
        Ok(Trained {
            files: vec![adapter.clone(), record],
            adapter: Some(adapter),
            params: serde_json::to_value(params).map_err(error)?,
            report,
        })
    }

    /// Direct preference optimisation on the dataset's pairs, against the
    /// base as the frozen reference.
    fn preference(
        &self,
        request: &TrainRequest<'_>,
        params: &LoraParams,
    ) -> Result<Trained, TrainError> {
        let loss = params
            .loss
            .as_deref()
            .map(str::parse::<PairLoss>)
            .transpose()
            .map_err(TrainError)?
            .unwrap_or_default();
        let model_dir = self.base_dir(request.base);
        let trained = train_preference_served(
            &self.endpoints,
            &PreferenceTune {
                model_dir: &model_dir,
                train: request.train,
                held_out: Self::held_out(request),
                attempt_dir: request.out,
                steps: params.steps,
                rank: params.rank,
                alpha: params.alpha.unwrap_or(2.0 * params.rank as f32),
                beta: params.beta.unwrap_or(DEFAULT_DPO_BETA),
                nll_weight: params.nll_weight.unwrap_or(0.0),
                loss,
                gamma: params.gamma,
                grad_accum: params.grad_accum,
                learning_rate: params.learning_rate,
                seed: request.seed,
                continue_from: request.start,
                cancel: None,
            },
        )
        .map_err(error)?;
        let report = serde_json::json!({
            "regime": "preference", "records": trained.records, "block": trained.block,
            "beta": trained.beta, "adapter_sha256": trained.adapter_digest,
            "base_digest": trained.base_digest,
        });
        Self::trained(params, trained.adapter, trained.training_record, report)
    }

    /// KTO on the dataset's judged completions, the smaller class weighed
    /// up so both weigh the same in all.
    fn feedback(
        &self,
        request: &TrainRequest<'_>,
        params: &LoraParams,
    ) -> Result<Trained, TrainError> {
        let text = std::fs::read_to_string(request.train).map_err(error)?;
        let (mut desirable, mut undesirable) = (0usize, 0usize);
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let record: Value = serde_json::from_str(line).map_err(error)?;
            match record.get("desirable").and_then(Value::as_bool) {
                Some(true) => desirable += 1,
                Some(false) => undesirable += 1,
                None => {
                    return Err(TrainError(
                        "a feedback record says not whether it is desirable".into(),
                    ))
                }
            }
        }
        let balance = splinter_data::FeedbackBalance::of(desirable, undesirable);
        let model_dir = self.base_dir(request.base);
        let trained = train_feedback_served(
            &self.endpoints,
            &FeedbackTune {
                model_dir: &model_dir,
                train: request.train,
                attempt_dir: request.out,
                steps: params.steps,
                rank: params.rank,
                alpha: params.alpha.unwrap_or(2.0 * params.rank as f32),
                beta: None,
                desirable_weight: balance.desirable_weight as f32,
                undesirable_weight: balance.undesirable_weight as f32,
                grad_accum: params.grad_accum,
                learning_rate: params.learning_rate,
                seed: request.seed,
                continue_from: request.start,
                cancel: None,
            },
        )
        .map_err(error)?;
        let report = serde_json::json!({
            "regime": "feedback", "records": trained.records, "summary": trained.summary,
            "desirable": desirable, "undesirable": undesirable,
            "adapter_sha256": trained.adapter_digest, "base_digest": trained.base_digest,
        });
        Self::trained(params, trained.adapter, trained.training_record, report)
    }

    /// The clipped policy gradient on the dataset's rollout records, on the
    /// advantages brain computes from each group's rewards.
    fn policy_gradient(
        &self,
        request: &TrainRequest<'_>,
        params: &LoraParams,
    ) -> Result<Trained, TrainError> {
        let model_dir = self.base_dir(request.base);
        let trained = train_policy_gradient_served(
            &self.endpoints,
            &PolicyGradientTune {
                model_dir: &model_dir,
                train: request.train,
                attempt_dir: request.out,
                steps: params.steps,
                rank: params.rank,
                alpha: params.alpha.unwrap_or(2.0 * params.rank as f32),
                weight_decay: params.weight_decay,
                grad_accum: params.grad_accum,
                continue_from: request.start,
                cancel: None,
                base: Self::precision(params)?,
                learning_rate: params.learning_rate,
                seed: request.seed,
                max_block: params.max_block,
                thinking: request.thinking,
                pack_answers: true,
                options: PolicyGradientOptions {
                    behaviour: params
                        .behaviour
                        .as_deref()
                        .map(str::parse::<Behaviour>)
                        .transpose()
                        .map_err(TrainError)?,
                    ..Default::default()
                },
            },
        )
        .map_err(error)?;
        let report = serde_json::json!({
            "regime": "policy-gradient", "records": trained.records, "block": trained.block,
            "statistics": trained.statistics, "adapter_sha256": trained.adapter_digest,
            "base_digest": trained.base_digest,
        });
        Self::trained(params, trained.adapter, trained.training_record, report)
    }
}

fn error(e: impl ToString) -> TrainError {
    TrainError(e.to_string())
}

impl Trainer for BrainTrainer {
    fn identity(&self) -> String {
        "brain served-train".into()
    }

    fn resolve(&self, method: &str, params: &Value) -> Result<Value, TrainError> {
        if method != "lora" {
            return Err(TrainError(format!(
                "brain trainer: method {method} is not supported here (lora)"
            )));
        }
        let mut lora: LoraParams = serde_json::from_value(params.clone())
            .map_err(|e| TrainError(format!("lora params: {e}")))?;
        lora.alpha.get_or_insert(2.0 * lora.rank as f32);
        let precision = match &lora.base_precision {
            Some(text) => text
                .parse::<BasePrecision>()
                .map_err(|e| TrainError(format!("lora params: base_precision: {e}")))?,
            None => BasePrecision::default(),
        };
        lora.base_precision = Some(
            serde_json::to_value(precision)
                .ok()
                .and_then(|v| v.as_str().map(str::to_owned))
                .unwrap_or_default(),
        );
        serde_json::to_value(&lora).map_err(error)
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
        // Brain numbers the file's lines from 1; record `i` is line `i + 1`.
        (0..lines.len())
            .map(|record| {
                report
                    .tokens_at(record + 1)
                    .map(|t| t as u64)
                    .ok_or_else(|| TrainError(format!("brain gave no length for record {record}")))
            })
            .collect()
    }

    fn train(&self, request: &TrainRequest<'_>) -> Result<Trained, TrainError> {
        let resolved = self.resolve(request.method, request.params)?;
        let params: LoraParams = serde_json::from_value(resolved)
            .map_err(|e| TrainError(format!("lora params: {e}")))?;
        match request.objective {
            // A text is one supervised turn: continued pretraining trains as
            // chat records do.
            "sft" | "cpt" => {}
            "preference" => return self.preference(request, &params),
            "feedback" => return self.feedback(request, &params),
            "policy-gradient" => return self.policy_gradient(request, &params),
            other => {
                return Err(TrainError(format!(
                    "brain trainer: a {other} dataset has no LoRA regime here"
                )))
            }
        }
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
            // The validation records are what the best step is chosen on
            // (and what the held-out scores are read over).
            monitor: (params.eval_every > 0 && !validation_empty).then_some(request.validation),
            eval_every: if validation_empty {
                0
            } else {
                params.eval_every
            },
            patience: params.patience,
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
            base: match &params.base_precision {
                Some(text) => text
                    .parse::<BasePrecision>()
                    .map_err(|e| TrainError(format!("lora params: base_precision: {e}")))?,
                None => BasePrecision::default(),
            },
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
