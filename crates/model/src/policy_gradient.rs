// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reinforcement learning from the verified
// rewards of agent rollouts for its clients. If your team needs expertise in
// training agents against verifiers that do not trust them, you can procure
// our services by sending an email to info@swedishembedded.com.

//! One LoRA fine-tune by a clipped policy gradient on rollouts collected
//! outside brain: the `policy_gradient` objective of brain's training job
//! ([`crate::served_train::train_policy_gradient_served`]).
//!
//! Each record of the dataset is one model call of an agent whose supervised
//! assistant turn is the action, carrying the `"reward"` of the rollout it
//! belongs to and the `"group"` of rollouts it is compared with (brain's
//! training-record contract v2; see `splinter-data`'s policy-gradient view).
//! brain turns each group's rewards into advantages by the normalisation
//! [`PolicyGradientOptions::advantage_norm`] names. A record whose advantage
//! is negative lowers the probability of its action, a positive one raises
//! it. A run holds nothing out and monitors
//! nothing - brain refuses both beside this objective - so what the adapter
//! earned is judged by the exam, not by this call. brain trains it on
//! `qwen35` bases only and refuses any other before it claims a device.

use std::path::{Path, PathBuf};

use serde::Serialize;

use splinter_core::training::PolicyGradientSummary;
use splinter_data::{AdvantageFloor, AdvantageNorm};

use crate::precision::BasePrecision;

/// The policy the KL term of the objective keeps the adapter close to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Reference {
    /// The frozen base alone.
    Base,
    /// The adapter the run starts from (the base, for a fresh adapter): a
    /// run that continues a release stays close to that release.
    StartOfRun,
}

impl std::str::FromStr for Reference {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, String> {
        match text {
            "base" => Ok(Self::Base),
            "start" => Ok(Self::StartOfRun),
            other => Err(format!("{other:?} is not a reference: base or start")),
        }
    }
}

/// A named policy-gradient algorithm: brain expands it into its settings,
/// which are the run's defaults; every setting named beside it overrides it.
/// The training record names the algorithm and every setting it trained with.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Algorithm {
    /// GRPO: z-score advantages, per-sequence mean, a symmetric clip band and
    /// the KL term.
    Grpo,
    /// Dr. GRPO: mean-centred advantages, a constant-length normaliser and no
    /// KL term.
    DrGrpo,
    /// DAPO: z-score advantages, clip-higher, token sum, truncated rollouts
    /// masked and no KL term.
    Dapo,
    /// RLOO: leave-one-out advantages and token sum.
    Rloo,
}

impl Algorithm {
    /// The name brain's training job takes.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Grpo => "grpo",
            Self::DrGrpo => "dr-grpo",
            Self::Dapo => "dapo",
            Self::Rloo => "rloo",
        }
    }
}

impl std::str::FromStr for Algorithm {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, String> {
        [Self::Grpo, Self::DrGrpo, Self::Dapo, Self::Rloo]
            .into_iter()
            .find(|a| a.name() == text)
            .ok_or_else(|| format!("{text:?} is not an algorithm: grpo, dr-grpo, dapo or rloo"))
    }
}

/// Where the behaviour log-probabilities of the importance ratio come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Behaviour {
    /// The sampler's own, recorded with each rollout: every supervised token
    /// must have one, and the ratio is the true importance ratio.
    Recorded,
    /// Where none was recorded, the log-probability under the policy the run
    /// starts from. Exact only when the rollouts were sampled from that policy
    /// (on-policy rounds are); otherwise a proxy, which the run's record says.
    StartOfRun,
}

impl std::str::FromStr for Behaviour {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, String> {
        match text {
            "recorded" => Ok(Self::Recorded),
            "start" => Ok(Self::StartOfRun),
            other => Err(format!(
                "{other:?} is not a behaviour source: recorded or start"
            )),
        }
    }
}

/// How the token terms of the loss are normalised.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum LossAggregation {
    /// Each sequence's terms are divided by its own supervised-token count.
    SeqMean,
    /// Every token of a step weighs the same whatever the length of its
    /// sequence.
    TokenSum,
}

impl std::str::FromStr for LossAggregation {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, String> {
        match text {
            "seq-mean" => Ok(Self::SeqMean),
            "token-sum" => Ok(Self::TokenSum),
            other => Err(format!(
                "{other:?} is not a loss aggregation: seq-mean or token-sum"
            )),
        }
    }
}

/// The settings of the policy-gradient objective a caller names; whatever it
/// leaves out is brain's default, which the run's training record states.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct PolicyGradientOptions {
    /// The named algorithm whose settings are the run's defaults; unset:
    /// brain's own defaults, setting by setting.
    pub algorithm: Option<Algorithm>,
    /// Half-width of the clip band `[1 - eps, 1 + eps]` of the importance
    /// ratio, in (0, 1).
    pub clip_eps: Option<f32>,
    /// The upper edge `1 + eps_high` of the clip band when it is wider than
    /// the lower one (clip-higher): a rare token that turns out well keeps
    /// gaining mass instead of being clipped early. Unset: symmetric.
    pub clip_eps_high: Option<f32>,
    /// Skip, and record, the optimiser step of a step whose mean KL to the
    /// reference exceeds this; unset: no bound.
    pub max_step_kl: Option<f32>,
    /// Skip, and record, the optimiser step of a step whose global gradient
    /// norm exceeds this; unset: no bound.
    pub max_step_grad_norm: Option<f32>,
    /// Coefficient of the KL term to the reference policy; 0 puts none in the
    /// loss.
    pub kl_beta: Option<f32>,
    /// The policy the KL term is taken against.
    pub reference: Option<Reference>,
    /// Where the behaviour log-probabilities come from.
    pub behaviour: Option<Behaviour>,
    /// How the token terms are normalised.
    pub loss_agg: Option<LossAggregation>,
    /// The temperature the rollouts were sampled at.
    pub temperature: Option<f32>,
    /// For records with a teacher context (self-distillation): the frozen
    /// policy that scores the student's tokens in the privileged context.
    pub teacher: Option<Reference>,
    /// For records with a teacher context: each token's advantage is clipped
    /// to `[-bound, bound]`.
    pub advantage_bound: Option<f32>,
    /// For records with a teacher context: the most the magnitudes of a
    /// sequence's token advantages may sum to; a sequence over it is scaled
    /// down, so a few pivot tokens cannot dominate a step.
    pub sequence_advantage_cap: Option<f32>,
    /// How brain turns each group's rewards into advantages: the
    /// normalisation the dataset was built for (its manifest names it), never
    /// a train flag. Unset: brain's default.
    pub advantage_norm: Option<AdvantageNorm>,
    /// Whether brain clamps each advantage at zero after its group's baseline
    /// is taken: the floor the dataset was built for (a positive-only one
    /// floors at zero), never a train flag. Unset: brain's default (none).
    pub advantage_floor: Option<AdvantageFloor>,
}

impl PolicyGradientOptions {
    /// The flags among them that were named, for a command to report when the
    /// run they were given to cannot use them.
    #[must_use]
    pub fn named(&self) -> Vec<&'static str> {
        [
            (self.algorithm.is_some(), "--algorithm"),
            (self.clip_eps.is_some(), "--clip-eps"),
            (self.clip_eps_high.is_some(), "--clip-eps-high"),
            (self.max_step_kl.is_some(), "--max-step-kl"),
            (self.max_step_grad_norm.is_some(), "--max-step-grad-norm"),
            (self.kl_beta.is_some(), "--kl-beta"),
            (self.reference.is_some(), "--reference"),
            (self.behaviour.is_some(), "--behaviour"),
            (self.loss_agg.is_some(), "--loss-agg"),
            (self.temperature.is_some(), "--temperature"),
            (self.teacher.is_some(), "--teacher-policy"),
            (self.advantage_bound.is_some(), "--advantage-bound"),
            (
                self.sequence_advantage_cap.is_some(),
                "--sequence-advantage-cap",
            ),
        ]
        .into_iter()
        .filter_map(|(set, flag)| set.then_some(flag))
        .collect()
    }
}

/// One policy-gradient fine-tune: what to train, on what, and where its files
/// go.
#[derive(Clone, Debug)]
pub struct PolicyGradientTune<'a> {
    /// Base checkpoint directory (the model the agent serves).
    pub model_dir: &'a Path,
    /// Chat-format JSONL rollout records, each carrying a reward and a group.
    pub train: &'a Path,
    /// This attempt's own directory: the adapter and its training record
    /// land here.
    pub attempt_dir: &'a Path,
    /// Optimizer steps to train for.
    pub steps: u32,
    /// LoRA rank of a fresh adapter.
    pub rank: u32,
    /// LoRA alpha of a fresh adapter.
    pub alpha: f32,
    /// The AdamW weight decay on the adapter's matrices.
    pub weight_decay: f32,
    /// Records averaged into one optimizer step.
    pub grad_accum: u32,
    /// An adapter to continue instead of starting a fresh one; its own rank
    /// and alpha then apply.
    pub continue_from: Option<&'a Path>,
    /// Stops training at the next optimizer step once cancelled.
    pub cancel: Option<&'a sven_sdk::CancelToken>,
    /// How the frozen base is held while the adapter trains.
    pub base: BasePrecision,
    /// The peak learning rate; brain's default when `None`.
    pub learning_rate: Option<f32>,
    /// The seed of a fresh adapter's initialisation and the batch order.
    pub seed: u64,
    /// The longest training row in tokens; a longer record is named and
    /// refused before any device is claimed.
    pub max_block: Option<u32>,
    /// Whether the model is trained to reason before it acts.
    pub thinking: bool,
    /// The objective's settings.
    pub options: PolicyGradientOptions,
}

/// What one policy-gradient fine-tune produced.
#[derive(Clone, Debug)]
pub struct TrainedPolicyGradient {
    /// The adapter file, carrying its ModelCard.
    pub adapter: PathBuf,
    /// `sha256:<hex>` of the adapter file.
    pub adapter_digest: String,
    /// `sha256:<hex>` of the base checkpoint the adapter was trained on.
    pub base_digest: String,
    /// brain's training record for the adapter, beside it: the settings, the
    /// make-up of the dataset and the loss, clip fraction, KL and importance
    /// ratio of every step.
    pub training_record: PathBuf,
    /// Records in the dataset.
    pub records: usize,
    /// Training row length, sized to the longest example.
    pub block: u32,
    /// The per-step statistics brain measured (KL to the reference, clip
    /// fraction, importance ratio) and their summary.
    pub statistics: PolicyGradientSummary,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_reference_and_the_loss_aggregation_are_named_as_brain_names_them() {
        assert_eq!("base".parse(), Ok(Reference::Base));
        assert_eq!("start".parse(), Ok(Reference::StartOfRun));
        assert!("StartOfRun".parse::<Reference>().is_err());
        assert_eq!("seq-mean".parse(), Ok(LossAggregation::SeqMean));
        assert_eq!("token-sum".parse(), Ok(LossAggregation::TokenSum));
        assert!("mean".parse::<LossAggregation>().is_err());
        assert_eq!("dr-grpo".parse(), Ok(Algorithm::DrGrpo));
        assert_eq!(Algorithm::Grpo.name(), "grpo");
        assert!("ppo".parse::<Algorithm>().is_err());
    }
}
