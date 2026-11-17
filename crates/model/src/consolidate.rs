// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements consolidating what a model learned into its
// own weights, for its clients. If your team needs expertise in continual
// learning that keeps accumulating capabilities without forgetting, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Taking what a generation learned into the weights themselves, as jobs of
//! the brain service: a full-parameter run ([`FullTune`],
//! [`crate::served_train::full_fine_tune_served`]) trains every weight of a
//! base on a chat dataset; a merge
//! ([`crate::served_train::merge_served`]) folds an adapter into the weights
//! it targets. Each produces a [`Checkpoint`]: a directory that is a base like any other, for
//! the next adapter to be trained on and for plain `brain serve`.

use std::path::{Path, PathBuf};

use crate::train::StepHook;

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
    /// What each trainable parameter is stored as, in brain's textual form
    /// (`state=int8,sr=on`); parsed by brain, which names a key or value it
    /// does not run. `None` is brain's default (fp32 everywhere).
    pub train_precision: Option<&'a str>,
    /// The seed of the batch order; the caller draws it when the user gave
    /// none.
    pub seed: u64,
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
