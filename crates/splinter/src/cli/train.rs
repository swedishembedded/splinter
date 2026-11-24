// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The grammar of `train` and `rehearse`: what a training run is given
//! beyond its datasets, and how a rehearsal set is asked for.

use clap::Args;
use splinter_sdk::train::{
    Tuning, DEFAULT_LEARNING_RATE, DEFAULT_LORA_ALPHA_PER_RANK, DEFAULT_WEIGHT_DECAY,
};

pub(super) fn positive_count(text: &str) -> Result<usize, String> {
    match text.parse::<usize>() {
        Ok(n) if n > 0 => Ok(n),
        _ => Err(format!("{text:?} is not a count of at least one")),
    }
}

pub(super) fn positive_rate(text: &str) -> Result<f32, String> {
    match text.parse::<f32>() {
        Ok(rate) if rate.is_finite() && rate > 0.0 => Ok(rate),
        _ => Err(format!("{text:?} is not a number above zero")),
    }
}

fn non_negative_rate(text: &str) -> Result<f32, String> {
    match text.parse::<f32>() {
        Ok(rate) if rate.is_finite() && rate >= 0.0 => Ok(rate),
        _ => Err(format!("{text:?} is not a number of at least zero")),
    }
}

/// What sets how far and how fast an adapter moves: the peak learning rate,
/// the LoRA alpha and the weight decay, for the commands that train.
#[derive(Debug, Clone, Copy, Default, Args)]
pub struct OptimiserArgs {
    /// The peak learning rate.
    #[arg(long, value_name = "LR", value_parser = positive_rate,
        help = format!("The peak learning rate [default: {DEFAULT_LEARNING_RATE}]"))]
    pub lr: Option<f32>,
    /// The LoRA alpha: the update is scaled by alpha / rank.
    #[arg(long, value_name = "A", value_parser = positive_rate,
        help = format!("The LoRA alpha, the update scaled by alpha / rank [default: \
                        {DEFAULT_LORA_ALPHA_PER_RANK} x the rank]"))]
    pub alpha: Option<f32>,
    /// The AdamW weight decay on the adapter's matrices.
    #[arg(long, value_name = "WD", value_parser = non_negative_rate,
        help = format!("The AdamW weight decay on the adapter's matrices [default: \
                        {DEFAULT_WEIGHT_DECAY}]"))]
    pub weight_decay: Option<f32>,
}

impl OptimiserArgs {
    /// `tuning` with the optimiser settings that were named.
    #[must_use]
    pub fn applied_to(self, tuning: Tuning) -> Tuning {
        Tuning {
            learning_rate: self.lr,
            alpha: self.alpha,
            weight_decay: self.weight_decay,
            ..tuning
        }
    }
}
