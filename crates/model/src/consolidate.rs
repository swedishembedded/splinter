// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements consolidating what a model learned into its
// own weights, for its clients. If your team needs expertise in continual
// learning that keeps accumulating capabilities without forgetting, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Taking what a generation learned into the weights themselves, through
//! brain's own trainers and exporters: [`full_fine_tune`] trains every weight
//! of a base on a chat dataset ([`brain::ChatFullFineTune`]); [`merge`] folds
//! an adapter into the weights it targets ([`brain::merge_adapter`]). Each
//! produces a [`Checkpoint`]: a directory that is a base like any other, for
//! the next adapter to be trained on and for plain `brain serve`.

use std::path::{Path, PathBuf};

use crate::error::PolicyError;
use crate::train::{base_weights, validate_dataset, StepHook, StepReport};

/// One full-parameter fine-tune: what to train, on what, and where its files
/// go.
#[derive(Clone, Debug)]
pub struct FullTune<'a> {
    /// The base checkpoint (file or directory) every weight starts from.
    pub model_dir: &'a Path,
    /// Chat-format JSONL records to train on.
    pub train: &'a Path,
    /// Chat datasets mixed into training whole, at `replay_share` of the draws.
    pub replay: &'a [PathBuf],
    /// The share of the draws `replay` takes in all; `None` mixes the union.
    pub replay_share: Option<f32>,
    /// Chat datasets each mixed in at a share of the draws of its own.
    pub weighted_replay: &'a [(PathBuf, f32)],
    /// Records averaged into one optimizer step.
    pub grad_accum: u32,
    /// This run's own directory: the packed dataset and the checkpoint land
    /// here.
    pub attempt_dir: &'a Path,
    /// Optimizer steps.
    pub steps: u32,
    /// The peak learning rate; brain's full-fine-tune default when `None`.
    pub learning_rate: Option<f32>,
    /// The AdamW weight decay.
    pub weight_decay: f32,
    /// The seed of the batch order; brain's default when `None`.
    pub seed: Option<u64>,
    /// Whether the model is trained to reason before it answers; off, it is
    /// trained for no-think mode, as the adapter trainer does.
    pub thinking: bool,
    /// Stops training at the next optimizer step once cancelled; a cancelled
    /// run writes no checkpoint and is reported as an error.
    pub cancel: Option<&'a sven_sdk::CancelToken>,
    /// Told after every optimizer step.
    pub on_step: Option<StepHook<'a>>,
}

/// A checkpoint directory brain wrote.
#[derive(Clone, Debug, PartialEq)]
pub struct Checkpoint {
    /// The directory: `model.safetensors`, `config.json` and the tokenizer
    /// files.
    pub dir: PathBuf,
    /// Its weights file.
    pub weights: PathBuf,
    /// `sha256:<hex>` of the weights file.
    pub weights_digest: String,
    /// Optimizer steps completed; 0 for a merge.
    pub steps_completed: u32,
    /// The training loss before the first step; `None` for a merge, which
    /// trains nothing.
    pub initial_loss: Option<f32>,
    /// The last step's training loss; `None` for a merge.
    pub final_loss: Option<f32>,
}

/// Trains every weight of `request.model_dir` on `request.train`.
pub fn full_fine_tune(request: &FullTune<'_>) -> Result<Checkpoint, PolicyError> {
    let failed = |reason: String| PolicyError::Train {
        dir: request.attempt_dir.to_path_buf(),
        reason,
    };
    validate_dataset(request.train)?;
    let (weights, _) = base_weights(request.model_dir)?;
    for replayed in request.replay {
        validate_dataset(replayed)?;
    }
    for (weighted, _) in request.weighted_replay {
        validate_dataset(weighted)?;
    }
    let mut tune = brain::ChatFullFineTune::from_pretrained(weights.as_str())
        .dataset(request.train)
        .out_dir(request.attempt_dir)
        .steps(request.steps)
        .grad_accum(request.grad_accum)
        .weight_decay(request.weight_decay)
        .thinking(request.thinking);
    if let Some(share) = request.replay_share {
        tune = tune.replay_share(share);
    }
    if let Some(lr) = request.learning_rate {
        tune = tune.lr(lr);
    }
    if let Some(seed) = request.seed {
        tune = tune.seed(seed);
    }
    for replayed in request.replay {
        tune = tune.replay(replayed);
    }
    for (weighted, share) in request.weighted_replay {
        tune = tune.replay_at(weighted, *share);
    }
    let brain_cancel = brain::CancelToken::armed();
    let on_step = request.on_step;
    let cancel = request.cancel;
    let outcome = tune
        .run_with(&brain_cancel, |progress| {
            if let Some(StepHook(report)) = on_step {
                report(&StepReport {
                    step: progress.step,
                    steps: progress.steps,
                    loss: progress.loss,
                });
            }
            if cancel.is_some_and(sven_sdk::CancelToken::is_cancelled) {
                brain_cancel.cancel();
            }
        })
        .map_err(|e| failed(format!("training on {}: {e}", request.train.display())))?;
    if brain_cancel.is_cancelled() || outcome.status != brain::FineTuneStatus::Completed {
        return Err(PolicyError::Cancelled {
            dir: request.attempt_dir.to_path_buf(),
        });
    }
    let incomplete = |what: &str| failed(format!("it reported no {what}"));
    Ok(Checkpoint {
        dir: outcome
            .checkpoint
            .ok_or_else(|| incomplete("checkpoint directory"))?,
        weights: outcome.weights.ok_or_else(|| incomplete("weights file"))?,
        weights_digest: outcome
            .weights_digest
            .ok_or_else(|| incomplete("weights digest"))?,
        steps_completed: outcome.steps_completed,
        initial_loss: outcome.initial_loss,
        final_loss: outcome.final_loss,
    })
}

/// Folds `adapter` into the weights of `base` it targets and writes the
/// result to `out_dir`.
pub fn merge(base: &Path, adapter: &Path, out_dir: &Path) -> Result<Checkpoint, PolicyError> {
    let failed = |reason: String| PolicyError::Train {
        dir: out_dir.to_path_buf(),
        reason,
    };
    let (weights, _) = base_weights(base)?;
    let merged = brain::merge_adapter(weights.as_str(), adapter, out_dir).map_err(|e| {
        failed(format!(
            "merging {} into {}: {e}",
            adapter.display(),
            base.display()
        ))
    })?;
    Ok(Checkpoint {
        dir: merged.dir,
        weights: merged.weights,
        weights_digest: merged.weights_digest,
        steps_completed: 0,
        initial_loss: None,
        final_loss: None,
    })
}
