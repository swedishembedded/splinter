// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The grammar of `train` and `rehearse`: what a training run is given
//! beyond its datasets, and how a rehearsal set is asked for.

use clap::Args;
use splinter_sdk::learn::DEFAULT_REHEARSAL_SHARE;
use splinter_sdk::train::{
    DEFAULT_DPO_BETA, DEFAULT_LORA_RANK, DEFAULT_REPLAY_FRACTION, MAX_PASSES,
};
use splinter_sdk::vocabulary::model_ref::ModelRef;

use super::{model_ref, MonitoringArgs};

fn rehearsal_share(text: &str) -> Result<f64, String> {
    let share: f64 = text.parse().map_err(|e| format!("{text:?}: {e}"))?;
    if share > 0.0 && share < 1.0 {
        Ok(share)
    } else {
        Err(format!("{text:?} is not a share in (0, 1)"))
    }
}

fn positive_count(text: &str) -> Result<usize, String> {
    match text.parse::<usize>() {
        Ok(n) if n > 0 => Ok(n),
        _ => Err(format!("{text:?} is not a count of at least one")),
    }
}

/// `train`: the datasets' objective decides how - chat datasets by
/// supervised fine-tuning, preference datasets by DPO.
#[derive(Debug, Args)]
pub struct TrainArgs {
    /// The datasets, by id or unique prefix, all chat or all preference
    /// pairs; the newest records of the last are held out for scoring.
    #[arg(required = true, value_name = "DATASET-ID")]
    pub datasets: Vec<String>,
    /// The base to train, and an adapter on it to continue.
    #[arg(long, value_parser = model_ref, default_value_t = ModelRef::policy_default(), value_name = "REF")]
    pub from: ModelRef,
    /// The fraction of each earlier release's training records replayed,
    /// when training from a policy alias.
    #[arg(long, default_value_t = DEFAULT_REPLAY_FRACTION, value_name = "F")]
    pub replay_fraction: f64,
    /// The step budget: the most steps the training may take.
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(u32).range(1..),
        help = format!("The step budget: the most steps the training may take [default: \
                        {MAX_PASSES} passes over the examples, within bounds]"))]
    pub steps: Option<u32>,
    /// LoRA rank of a new adapter.
    #[arg(long, default_value_t = DEFAULT_LORA_RANK, value_name = "R")]
    pub rank: u32,
    /// The DPO temperature, for preference datasets only.
    #[arg(long, value_name = "BETA", help = beta_help())]
    pub beta: Option<f32>,
    /// The peak learning rate (brain's default if not given).
    #[arg(long, value_name = "LR")]
    pub lr: Option<f32>,
    /// Records averaged into one optimizer step (default: one when the steps
    /// are named, else from the size of the dataset).
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(u32).range(1..))]
    pub records_per_step: Option<u32>,
    /// The seed of a fresh adapter's initialisation and the batch order
    /// (brain's default if not given).
    #[arg(long, value_name = "N")]
    pub seed: Option<u64>,
    /// A rehearsal dataset (`rehearse`, or any chat dataset of the base's
    /// own answers) mixed into training at --rehearsal-share of the draws,
    /// never held out, a share of it in the monitoring set.
    #[arg(long, value_name = "DATASET-ID")]
    pub rehearsal: Option<String>,
    /// The share of the training draws the rehearsal takes, in (0, 1).
    #[arg(long, value_name = "F", value_parser = rehearsal_share, requires = "rehearsal",
        help = format!("The share of the training draws the rehearsal takes, in (0, 1) [default: \
                        {DEFAULT_REHEARSAL_SHARE}]"))]
    pub rehearsal_share: Option<f64>,
    /// How the training is watched.
    #[command(flatten)]
    pub monitoring: MonitoringArgs,
}

/// `rehearse`.
#[derive(Debug, Args)]
pub struct RehearseArgs {
    /// How many tasks the set holds: as many as make the rehearsal share of
    /// the examples it will be trained beside (`learn` sizes it so).
    #[arg(long, value_name = "N", value_parser = positive_count)]
    pub records: usize,
}

/// `--beta`'s help, naming the default it falls back to.
fn beta_help() -> String {
    format!("The DPO temperature, for preference datasets only [default: {DEFAULT_DPO_BETA}]")
}
