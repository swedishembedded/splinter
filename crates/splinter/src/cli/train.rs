// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The grammar of `train` and `rehearse`: what a training run is given
//! beyond its datasets, and how a rehearsal set is asked for.

use clap::Args;
use splinter_sdk::learn::DEFAULT_REHEARSAL_SHARE;
use splinter_sdk::train::{
    Tuning, DEFAULT_DPO_BETA, DEFAULT_LEARNING_RATE, DEFAULT_LORA_ALPHA_PER_RANK,
    DEFAULT_LORA_RANK, DEFAULT_MONITOR_SHARE, DEFAULT_PATIENCE, DEFAULT_REPLAY_FRACTION,
    DEFAULT_WEIGHT_DECAY, EVALUATIONS_PER_BUDGET, MAX_MONITOR_SHARE, MAX_PASSES,
};
use splinter_sdk::vocabulary::model_ref::ModelRef;

use super::model_ref;

fn rehearsal_share(text: &str) -> Result<f64, String> {
    let share: f64 = text.parse().map_err(|e| format!("{text:?}: {e}"))?;
    if share > 0.0 && share < 1.0 {
        Ok(share)
    } else {
        Err(format!("{text:?} is not a share in (0, 1)"))
    }
}

fn monitor_share(text: &str) -> Result<f64, String> {
    let share: f64 = text.parse().map_err(|e| format!("{text:?}: {e}"))?;
    if share > 0.0 && share <= MAX_MONITOR_SHARE {
        Ok(share)
    } else {
        Err(format!(
            "{text:?} is not a share in (0, {MAX_MONITOR_SHARE}]"
        ))
    }
}

fn positive_count(text: &str) -> Result<usize, String> {
    match text.parse::<usize>() {
        Ok(n) if n > 0 => Ok(n),
        _ => Err(format!("{text:?} is not a count of at least one")),
    }
}

fn positive_rate(text: &str) -> Result<f32, String> {
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

/// How a supervised training run is watched as it trains, for the commands
/// that train: a share of its training families is set aside and scored
/// every few steps, the adapter of the evaluation with the lowest loss on
/// them is the candidate's, and the run stops once that loss has gone a
/// patience of evaluations without improving.
#[derive(Debug, Default, Args)]
pub struct MonitoringArgs {
    /// Steps between evaluations of the monitoring records; 0 monitors
    /// nothing and the candidate carries its last step.
    #[arg(long, value_name = "N", help = format!(
        "Steps between evaluations of the monitoring records; 0 monitors nothing and the \
         candidate carries its last step [default: {EVALUATIONS_PER_BUDGET} evaluations over the \
         step budget]"
    ))]
    pub eval_every: Option<u32>,
    /// Evaluations without improvement before the training stops; 0 runs
    /// the whole budget (the best evaluation is carried either way).
    #[arg(long, value_name = "N", help = format!(
        "Evaluations without improvement before the training stops; 0 runs the whole budget \
         (the best evaluation is carried either way) [default: {DEFAULT_PATIENCE}]"
    ))]
    pub patience: Option<u32>,
    /// The share of the training families set aside as the monitoring
    /// records, in (0, 1/2]: whole families, never the held-out ones the
    /// gate and the exam decide on.
    #[arg(long, value_name = "SHARE", value_parser = monitor_share, help = format!(
        "The share of the training families set aside as the monitoring records, in (0, \
         {MAX_MONITOR_SHARE}]: whole families, never the held-out ones the gate and the exam \
         decide on [default: {DEFAULT_MONITOR_SHARE}]"
    ))]
    pub monitor_share: Option<f64>,
    /// Keep the adapter of every evaluation with the candidate, so a step
    /// can be chosen on the dev suite (`select`) instead of the monitoring
    /// loss.
    #[arg(long)]
    pub keep_evaluations: bool,
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
    /// The learning rate, alpha and weight decay of the training.
    #[command(flatten)]
    pub optimiser: OptimiserArgs,
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
