// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reinforcement learning from the verified
// rewards of agent rollouts for its clients. If your team needs expertise in
// training agents against verifiers that do not trust them, you can procure
// our services by sending an email to info@swedishembedded.com.

//! How a dataset build rewards an attempt and reads its advantage (the
//! policy-gradient and self-distill views).

use serde::Serialize;
use splinter_data::{AdvantageKind, Overlong, RewardShape};
use splinter_orchestrator::error::OrchestratorError;

/// How the policy-gradient view rewards an attempt and reads its advantage;
/// the other views ignore it.
#[derive(Clone, Debug, Default, Serialize)]
pub struct RlBuild {
    /// The metrics a failed attempt's reward is made of, with their weights
    /// (non-negative, summing to one); none leaves a failure at zero.
    pub reward_metrics: Vec<(String, f64)>,
    /// The most a failed attempt can earn, below a pass's one; needs
    /// `reward_metrics`. `None` is [`DEFAULT_FAIL_CEILING`].
    pub fail_ceiling: Option<f64>,
    /// How the advantage is read from the rewards of a group.
    pub advantage: AdvantageKind,
    /// For the policy-gradient view: distil the groups whose attempts all
    /// failed alike (the cliff, where GRPO's advantage is zero) with a
    /// teacher context, and keep the scalar advantage for the other groups.
    /// The self-distill view distils every attempt and needs no flag.
    pub cliff_distill: bool,
    /// DAPO's soft over-length penalty: the completion tokens at and past
    /// which an attempt loses a whole reward (`L`); `None` penalises no
    /// length.
    pub overlong_max_tokens: Option<u64>,
    /// The tokens before `overlong_max_tokens` over which the penalty grows
    /// from nothing to a whole reward (`B`); needs `overlong_max_tokens`.
    /// `None` is a quarter of the budget, DAPO's ratio.
    pub overlong_buffer: Option<u64>,
}

/// The share of the over-length budget the penalty grows over when no buffer
/// is named, as its divisor: a quarter (DAPO's 4096 tokens of 16384).
pub const DEFAULT_OVERLONG_BUFFER_DIVISOR: u64 = 4;

/// The most a failed attempt earns when metrics shape its reward and no
/// ceiling is named: half of a pass's.
pub const DEFAULT_FAIL_CEILING: f64 = 0.5;

impl RlBuild {
    /// The reward shape this asks for.
    pub(crate) fn shape(&self) -> Result<RewardShape, OrchestratorError> {
        let refused = |e: splinter_data::ViewError| OrchestratorError::Refused(e.to_string());
        let shape = self.verdict_shape()?;
        match (self.overlong_max_tokens, self.overlong_buffer) {
            (None, None) => Ok(shape),
            (None, Some(_)) => Err(OrchestratorError::Refused(
                "--overlong-buffer is the stretch before the --overlong-max-tokens budget over \
                 which the over-length penalty grows; name the budget"
                    .into(),
            )),
            (Some(max_tokens), buffer) => Ok(shape.with_overlong(
                Overlong::new(
                    max_tokens,
                    buffer.unwrap_or(max_tokens / DEFAULT_OVERLONG_BUFFER_DIVISOR),
                )
                .map_err(refused)?,
            )),
        }
    }

    /// The reward shape of the verdict and metrics, before any length penalty.
    fn verdict_shape(&self) -> Result<RewardShape, OrchestratorError> {
        let refused = |e: splinter_data::ViewError| OrchestratorError::Refused(e.to_string());
        match (self.reward_metrics.is_empty(), self.fail_ceiling) {
            (true, None) => Ok(RewardShape::unshaped()),
            (true, Some(_)) => Err(OrchestratorError::Refused(
                "--fail-ceiling bounds what the --reward-metric terms earn a failed attempt; \
                 name at least one metric"
                    .into(),
            )),
            (false, ceiling) => RewardShape::shaped(
                ceiling.unwrap_or(DEFAULT_FAIL_CEILING),
                self.reward_metrics.clone(),
            )
            .map_err(refused),
        }
    }

    pub(crate) fn is_default(&self) -> bool {
        self.reward_metrics.is_empty()
            && self.fail_ceiling.is_none()
            && self.advantage == AdvantageKind::default()
            && !self.cliff_distill
            && self.overlong_max_tokens.is_none()
            && self.overlong_buffer.is_none()
    }
}
