// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reinforcement learning from the verified
// rewards of agent rollouts for its clients. If your team needs expertise in
// training agents against verifiers that do not trust them, you can procure
// our services by sending an email to info@swedishembedded.com.

//! One LoRA fine-tune by a clipped policy gradient on rollouts collected
//! outside brain: brain's own trainer ([`brain::ChatFineTune::policy_gradient`])
//! called through its SDK.
//!
//! Each record of the dataset is one model call of an agent whose supervised
//! assistant turn is the action, carrying the `"advantage"` of the rollout it
//! belongs to (see `splinter-data`'s policy-gradient view). A record with a
//! negative advantage lowers the probability of its action, one with a
//! positive advantage raises it. A run holds nothing out and monitors
//! nothing - brain refuses both beside this objective - so what the adapter
//! earned is judged by the exam, not by this call. brain trains it on
//! `qwen35` bases only and refuses any other before it claims a device.

use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::capabilities::overlong_records;
use crate::error::PolicyError;
use crate::pictures::validate_dataset_on;
use crate::precision::BasePrecision;
use crate::progress::poll;
use crate::train::{base_weights, candidate_adapter_id};

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
/// leaves out is brain's default ([`PolicyGradientOptions::resolved`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct PolicyGradientOptions {
    /// Half-width of the clip band `[1 - eps, 1 + eps]` of the importance
    /// ratio, in (0, 1).
    pub clip_eps: Option<f32>,
    /// Coefficient of the KL term to the reference policy; 0 puts none in the
    /// loss.
    pub kl_beta: Option<f32>,
    /// The policy the KL term is taken against.
    pub reference: Option<Reference>,
    /// How the token terms are normalised.
    pub loss_agg: Option<LossAggregation>,
    /// The temperature the rollouts were sampled at.
    pub temperature: Option<f32>,
}

/// The settings with every default made explicit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResolvedPolicyGradient {
    /// See [`PolicyGradientOptions::clip_eps`].
    pub clip_eps: f32,
    /// See [`PolicyGradientOptions::kl_beta`].
    pub kl_beta: f32,
    /// See [`PolicyGradientOptions::reference`].
    pub reference: Reference,
    /// See [`PolicyGradientOptions::loss_agg`].
    pub loss_agg: LossAggregation,
    /// See [`PolicyGradientOptions::temperature`].
    pub temperature: f32,
}

impl PolicyGradientOptions {
    /// The flags among them that were named, for a command to report when the
    /// run they were given to cannot use them.
    #[must_use]
    pub fn named(&self) -> Vec<&'static str> {
        [
            (self.clip_eps.is_some(), "--clip-eps"),
            (self.kl_beta.is_some(), "--kl-beta"),
            (self.reference.is_some(), "--reference"),
            (self.loss_agg.is_some(), "--loss-agg"),
            (self.temperature.is_some(), "--temperature"),
        ]
        .into_iter()
        .filter_map(|(set, flag)| set.then_some(flag))
        .collect()
    }

    /// The settings with brain's defaults where none was named.
    #[must_use]
    pub fn resolved(&self) -> ResolvedPolicyGradient {
        let defaults = Self::defaults();
        ResolvedPolicyGradient {
            clip_eps: self.clip_eps.unwrap_or(defaults.clip_eps),
            kl_beta: self.kl_beta.unwrap_or(defaults.kl_beta),
            reference: self.reference.unwrap_or(defaults.reference),
            loss_agg: self.loss_agg.unwrap_or(defaults.loss_agg),
            temperature: self.temperature.unwrap_or(defaults.temperature),
        }
    }

    /// brain's defaults.
    #[must_use]
    pub fn defaults() -> ResolvedPolicyGradient {
        let config = brain::PolicyGradientConfig::default();
        ResolvedPolicyGradient {
            clip_eps: config.clip_eps,
            kl_beta: config.kl_beta,
            reference: match config.reference {
                brain::ReferencePolicy::Base => Reference::Base,
                brain::ReferencePolicy::StartOfRun => Reference::StartOfRun,
            },
            loss_agg: match config.loss_agg {
                brain::LossAgg::SeqMean => LossAggregation::SeqMean,
                brain::LossAgg::TokenSum => LossAggregation::TokenSum,
            },
            temperature: config.temperature,
        }
    }

    fn config(&self) -> brain::PolicyGradientConfig {
        let resolved = self.resolved();
        brain::PolicyGradientConfig {
            clip_eps: resolved.clip_eps,
            kl_beta: resolved.kl_beta,
            reference: match resolved.reference {
                Reference::Base => brain::ReferencePolicy::Base,
                Reference::StartOfRun => brain::ReferencePolicy::StartOfRun,
            },
            loss_agg: match resolved.loss_agg {
                LossAggregation::SeqMean => brain::LossAgg::SeqMean,
                LossAggregation::TokenSum => brain::LossAgg::TokenSum,
            },
            temperature: resolved.temperature,
            ..brain::PolicyGradientConfig::default()
        }
    }
}

/// One policy-gradient fine-tune: what to train, on what, and where its files
/// go.
#[derive(Clone, Debug)]
pub struct PolicyGradientTune<'a> {
    /// Base checkpoint directory (the model the agent serves).
    pub model_dir: &'a Path,
    /// Chat-format JSONL rollout records, each carrying an advantage.
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
}

/// Fine-tunes a LoRA on the rollout records of `request.train` by a clipped
/// policy gradient.
pub fn train_policy_gradient(
    request: &PolicyGradientTune<'_>,
) -> Result<TrainedPolicyGradient, PolicyError> {
    let failed = |reason: String| PolicyError::Train {
        dir: request.attempt_dir.to_path_buf(),
        reason,
    };
    let trained_on = validate_dataset_on(request.train, request.model_dir)?;
    let (weights, base_id) = base_weights(request.model_dir)?;
    if let Some(max_block) = request.max_block {
        let long = overlong_records(
            request.train,
            request.model_dir,
            request.thinking,
            false,
            max_block,
        )?;
        if !long.is_empty() {
            return Err(failed(format!(
                "{} record(s) longer than max_block {max_block}; shorten or drop them: {}",
                long.len(),
                long.iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("; ")
            )));
        }
    }
    let mut fine_tune = brain::ChatFineTune::from_pretrained(weights.as_str())
        .device(crate::device::selected().spec())
        .dataset(request.train)
        .thinking(request.thinking)
        .out_dir(request.attempt_dir)
        .adapter_id(candidate_adapter_id(&base_id, request.attempt_dir)?)
        .steps(request.steps)
        .rank(request.rank)
        .alpha(request.alpha)
        .seed(request.seed)
        .weight_decay(request.weight_decay)
        .grad_accum(request.grad_accum)
        .policy_gradient(request.options.config());
    if let Some(format) = request.base.brain_format() {
        fine_tune = fine_tune.base_format(format);
    }
    if let Some(lr) = request.learning_rate {
        fine_tune = fine_tune.lr(lr);
    }
    if let Some(max_block) = request.max_block {
        fine_tune = fine_tune.max_block(max_block);
    }
    if let Some(adapter) = request.continue_from {
        fine_tune = fine_tune.continue_from(adapter);
    }
    let brain_cancel = brain::CancelToken::armed();
    let outcome = fine_tune
        .run_with(
            &brain_cancel,
            poll(request.cancel, &brain_cancel, None, None),
        )
        .map_err(|e| {
            failed(format!(
                "policy-gradient training on {}: {e}",
                request.train.display()
            ))
        })?;
    if brain_cancel.is_cancelled() {
        return Err(PolicyError::Cancelled {
            dir: request.attempt_dir.to_path_buf(),
        });
    }
    if outcome.status != brain::FineTuneStatus::Completed {
        return Err(failed("it did not complete".into()));
    }
    if outcome.policy_gradient.is_none() {
        return Err(failed(
            "brain trained the records by imitation and reported no policy-gradient run".into(),
        ));
    }
    let incomplete = |what: &str| failed(format!("it reported no {what}"));
    Ok(TrainedPolicyGradient {
        adapter: outcome.adapter.ok_or_else(|| incomplete("adapter"))?,
        adapter_digest: outcome
            .adapter_digest
            .ok_or_else(|| incomplete("adapter digest"))?,
        base_digest: outcome
            .base_digest
            .ok_or_else(|| incomplete("base digest"))?,
        training_record: outcome
            .record
            .ok_or_else(|| incomplete("training record"))?,
        records: trained_on.records,
        block: outcome.block,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Settings left out are brain's defaults, so a run says plainly what it
    /// ran with; settings named are the ones brain is handed.
    #[test]
    fn what_is_not_named_is_brains_default_and_what_is_named_is_kept() {
        let none = PolicyGradientOptions::default();
        assert!(none.named().is_empty());
        assert_eq!(none.resolved(), PolicyGradientOptions::defaults());
        let config = none.config();
        let brain_default = brain::PolicyGradientConfig::default();
        assert_eq!(
            (config.clip_eps, config.kl_beta, config.temperature),
            (
                brain_default.clip_eps,
                brain_default.kl_beta,
                brain_default.temperature
            )
        );

        let named = PolicyGradientOptions {
            clip_eps: Some(0.3),
            kl_beta: Some(0.0),
            reference: Some(Reference::StartOfRun),
            loss_agg: Some(LossAggregation::TokenSum),
            temperature: Some(0.6),
        };
        assert_eq!(named.named().len(), 5);
        let config = named.config();
        assert_eq!(
            (config.clip_eps, config.kl_beta, config.temperature),
            (0.3, 0.0, 0.6)
        );
        assert_eq!(config.reference, brain::ReferencePolicy::StartOfRun);
        assert_eq!(config.loss_agg, brain::LossAgg::TokenSum);
    }

    #[test]
    fn the_reference_and_the_loss_aggregation_are_named_as_brain_names_them() {
        assert_eq!("base".parse(), Ok(Reference::Base));
        assert_eq!("start".parse(), Ok(Reference::StartOfRun));
        assert!("StartOfRun".parse::<Reference>().is_err());
        assert_eq!("seq-mean".parse(), Ok(LossAggregation::SeqMean));
        assert_eq!("token-sum".parse(), Ok(LossAggregation::TokenSum));
        assert!("mean".parse::<LossAggregation>().is_err());
    }
}
