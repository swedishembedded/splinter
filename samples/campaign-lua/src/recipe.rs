// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that put an informative on-policy
// reinforcement-learning signal through a language model against verifiers it
// cannot influence, for its clients. If your team needs expertise in
// recipes for stable policy-gradient training of small language models you
// can procure our services by sending an email to info@swedishembedded.com.

//! The recipe of a campaign: every setting that shapes what an arm trains on
//! and how, as command-line flags with one definition, and the controller and trainer
//! configuration they make.

use anyhow::Result;
use clap::Args;
use splinter_sdk::model::policy_gradient::{Behaviour, PolicyGradientOptions, Reference};
use splinter_sdk::model::precision::BasePrecision;
use splinter_sdk::served::brain::{ModelSettings, TrainerSettings};

/// The settings of an arm.
#[derive(Args, Clone, Debug)]
pub struct Recipe {
    /// Optimiser steps per arm.
    #[arg(long, default_value_t = 16)]
    pub steps: u32,
    /// Learning rate.
    #[arg(long, default_value_t = 1e-5)]
    pub lr: f32,
    /// Adapter rank.
    #[arg(long, default_value_t = 16)]
    pub rank: u32,
    /// Records averaged into one optimiser step.
    #[arg(long, default_value_t = 8)]
    pub records_per_step: u32,
    /// The lead (share of dev programs won) an arm is meant to produce.
    #[arg(long, default_value_t = 0.1)]
    pub target_effect: f64,
    /// Expected share of dev programs a candidate and its champion differ on.
    #[arg(long, default_value_t = 0.2)]
    pub discordance: f64,
    /// Requests in flight at once.
    #[arg(long, default_value_t = 16)]
    pub concurrency: usize,
    /// Tokens a completion may take.
    #[arg(long, default_value_t = 24)]
    pub max_tokens: u32,
    /// Retention loss, as a share of the retention set, a candidate may show
    /// (a non-inferiority margin); also the loss the retention set must be
    /// large enough to detect.
    #[arg(long, default_value_t = 0.0)]
    pub retention_margin: f64,
    /// Precision of the frozen base during training: `bf16` or `fp32`.
    #[arg(long, default_value = "bf16")]
    pub base_precision: String,
    /// Half-width of the clip band of the importance ratio.
    #[arg(long)]
    pub clip_eps: Option<f32>,
    /// The upper edge of the clip band, `1 + this` (clip-higher); symmetric
    /// when absent.
    #[arg(long)]
    pub clip_eps_high: Option<f32>,
    /// Skip an optimiser step whose mean KL to the reference exceeds this.
    #[arg(long)]
    pub max_step_kl: Option<f32>,
    /// Skip an optimiser step whose gradient norm exceeds this.
    #[arg(long)]
    pub max_step_grad_norm: Option<f32>,
}

impl Recipe {
    /// How the model is loaded and sampled for the base at `base`.
    #[must_use]
    pub fn model_settings(&self, base: &std::path::Path) -> ModelSettings {
        let mut model = ModelSettings::new(base);
        model.max_tokens = self.max_tokens;
        model.concurrency = self.concurrency;
        model
    }

    /// How brain trains an arm: a clipped policy gradient on the policy's own
    /// attempts, scored against the policy the arm started from (exact for
    /// on-policy rollouts), with no KL term in the loss and the guard bounds
    /// named here.
    ///
    /// # Errors
    /// The base precision is not `bf16` or `fp32`.
    pub fn trainer_settings(&self, base: &std::path::Path) -> Result<TrainerSettings> {
        Ok(TrainerSettings {
            base: base.to_path_buf(),
            steps: self.steps,
            rank: self.rank,
            alpha: (self.rank * 2) as f32,
            learning_rate: self.lr,
            records_per_step: self.records_per_step,
            max_block: Some(1024),
            options: PolicyGradientOptions {
                clip_eps: self.clip_eps,
                clip_eps_high: self.clip_eps_high,
                max_step_kl: self.max_step_kl,
                max_step_grad_norm: self.max_step_grad_norm,
                kl_beta: Some(0.0),
                reference: Some(Reference::StartOfRun),
                behaviour: Some(Behaviour::StartOfRun),
                temperature: Some(1.0),
                ..PolicyGradientOptions::default()
            },
            base_precision: match self.base_precision.as_str() {
                "bf16" => BasePrecision::Bf16,
                "fp32" => BasePrecision::Fp32,
                other => anyhow::bail!("{other:?} is not a base precision: bf16 or fp32"),
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        recipe: Recipe,
    }

    #[test]
    fn the_guard_and_the_clip_edges_reach_the_trainer_as_named() {
        let cli = Cli::parse_from([
            "x",
            "--clip-eps",
            "0.2",
            "--clip-eps-high",
            "0.28",
            "--max-step-kl",
            "0.3",
            "--max-step-grad-norm",
            "5",
        ]);
        let trainer = cli
            .recipe
            .trainer_settings(std::path::Path::new("m"))
            .unwrap();
        let o = trainer.options;
        assert_eq!(
            (
                o.clip_eps,
                o.clip_eps_high,
                o.max_step_kl,
                o.max_step_grad_norm
            ),
            (Some(0.2), Some(0.28), Some(0.3), Some(5.0))
        );
        assert_eq!(o.behaviour, Some(Behaviour::StartOfRun));
    }

    #[test]
    fn an_unknown_precision_is_refused() {
        let cli = Cli::parse_from(["x", "--base-precision", "int3"]);
        assert!(cli
            .recipe
            .trainer_settings(std::path::Path::new("m"))
            .is_err());
    }
}
