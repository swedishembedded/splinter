// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded fine-tuning of language models
// for its clients. If your team needs expertise in training adapters on large
// bases with limited GPU memory, you can procure our services by sending an
// email to info@swedishembedded.com.

//! What a training run is tuned with beyond its datasets, step count and rank.

use serde::Serialize;

/// How a run trains beyond its step count and rank: what depends on the
/// base and the card, not on the data.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct Tuning {
    /// Start a fresh adapter on the frozen base instead of continuing the
    /// release the reference names, which stays the candidate's parent: a
    /// rebuild from the base on all the experience given, not a step on
    /// from the champion.
    pub restart: bool,
    /// Hold the frozen base at bf16, half the bytes of fp32: what lets a 7B
    /// base train on one 24 GiB card. Supervised runs only; brain's
    /// preference trainer holds its base at its own tier.
    pub bf16_base: bool,
    /// How the frozen base is held, when named: it overrides [`Self::bf16_base`].
    /// `None` is bf16 when that flag is set, else brain's fp32. NF4 is what
    /// trains the 27B beside a server on the same card.
    pub base_precision: Option<splinter_model::train::BasePrecision>,
    /// The peak learning rate; [`super::DEFAULT_LEARNING_RATE`] when `None`.
    pub learning_rate: Option<f32>,
    /// The LoRA alpha of a new adapter; [`super::default_alpha`] of the rank when
    /// `None`.
    pub alpha: Option<f32>,
    /// The AdamW weight decay; [`super::DEFAULT_WEIGHT_DECAY`] when `None`.
    pub weight_decay: Option<f32>,
    /// Records averaged into one optimizer step. `None` is one when the
    /// steps are named, else [`super::auto_records_per_step`] of the examples.
    pub records_per_step: Option<u32>,
    /// Steps between evaluations of the monitoring set; `None` is
    /// [`super::eval_every_for`] the budget, 0 monitors nothing and the candidate
    /// carries its last step.
    pub eval_every: Option<u32>,
    /// Evaluations without improvement before the run stops; `None` is
    /// [`super::DEFAULT_PATIENCE`], 0 runs the whole budget (the best evaluation is
    /// carried either way).
    pub patience: Option<u32>,
    /// The share of the training families set aside as the monitoring set;
    /// `None` is [`super::DEFAULT_MONITOR_SHARE`].
    pub monitor_share: Option<f64>,
    /// The seed of a fresh adapter's initialisation, the batch order and the
    /// draw of the records replayed; a fresh one is drawn, reported and
    /// recorded when `None`. Two runs that differ only here show how much of
    /// a measured difference is the draw.
    pub seed: Option<u64>,
    /// Keep the adapter of every evaluation with the candidate, so a step can
    /// be chosen on something other than the monitoring loss
    /// ([`crate::checkpoints`]); only the exported one is kept otherwise.
    pub keep_evaluations: bool,
    /// The longest training row in tokens the card fits, as a record is
    /// measured; `None` sizes the row to the longest record. Set, a record
    /// longer than it is named before any device is claimed and the run does
    /// not start. A dataset built for the run leaves such records out when
    /// it is built with the same bound ([`crate::datasets::BuildRequest::max_block`]).
    pub max_block: Option<u32>,
}
