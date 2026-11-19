// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reinforcement learning from the verified
// rewards of agent rollouts for its clients. If your team needs expertise in
// training agents against verifiers that do not trust them, you can procure
// our services by sending an email to info@swedishembedded.com.

//! How a dataset build rewards an attempt and reads its advantage (the
//! policy-gradient and self-distill views), and how many pairs of attempts
//! it makes (the preference view).

use serde::{Deserialize, Serialize};
use splinter_data::{AdvantageKind, Overlong, RewardShape};
use splinter_orchestrator::error::OrchestratorError;

use crate::bench::practice::PracticeFacts;

/// A named reward: how a failed attempt earns part of a pass's reward.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RewardPreset {
    /// A pass earns one, a failure what its `reward_metrics` earn under the
    /// ceiling (nothing without any).
    #[default]
    Verdict,
    /// A pass earns one, a failure `c * max(0, (f - f_naive) / (1 - f_naive))`
    /// of its [`PARTIAL_METRIC`] `f` above its task's naive floor `f_naive`
    /// (the examiner's, from the practice material; zero without one), `c`
    /// the fail ceiling ([`splinter_data::PartialCredit`]). It changes what a
    /// failure teaches, so it is named, never the default.
    ShapedPartial,
}

impl RewardPreset {
    /// The preset's name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Verdict => "verdict",
            Self::ShapedPartial => "shaped-partial",
        }
    }
}

impl std::str::FromStr for RewardPreset {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, String> {
        [Self::Verdict, Self::ShapedPartial]
            .into_iter()
            .find(|p| p.name() == text)
            .ok_or_else(|| format!("{text:?} is not a reward: verdict or shaped-partial"))
    }
}

/// The metric `shaped-partial` gives partial credit for: the share of a
/// task's weighted checks an attempt passed, as the examiner's verifiers
/// measure it.
pub const PARTIAL_METRIC: &str = "functional_correctness";

/// How the policy-gradient view rewards an attempt and reads its advantage;
/// the other views ignore it.
#[derive(Clone, Debug, Default, Serialize)]
pub struct RlBuild {
    /// The named reward.
    pub reward: RewardPreset,
    /// The metrics a failed attempt's reward is made of, with their weights
    /// (non-negative, summing to one); none leaves a failure at zero. Only
    /// under [`RewardPreset::Verdict`].
    pub reward_metrics: Vec<(String, f64)>,
    /// The most a failed attempt can earn, below a pass's one; needs
    /// `reward_metrics` or [`RewardPreset::ShapedPartial`]. `None` is
    /// [`DEFAULT_FAIL_CEILING`].
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
    /// For the preference view: the most pairs of one group of attempts (one
    /// task version by one policy) or one repaired task; `None` is
    /// [`DEFAULT_PAIRS_PER_GROUP`].
    pub pairs_per_group: Option<usize>,
    /// For the preference view: the seed the pairs of a group are chosen by;
    /// `None` derives it from the experiences, so building again from them
    /// pairs the same.
    pub pair_seed: Option<u64>,
}

/// The most pairs the preference view makes of one group of attempts when
/// none is named: a group of eight attempts that split evenly offers sixteen,
/// and a task with many attempts would otherwise outweigh the others.
pub const DEFAULT_PAIRS_PER_GROUP: usize = 4;

/// The share of the over-length budget the penalty grows over when no buffer
/// is named, as its divisor: a quarter (DAPO's 4096 tokens of 16384).
pub const DEFAULT_OVERLONG_BUFFER_DIVISOR: u64 = 4;

/// The most a failed attempt earns when metrics shape its reward and no
/// ceiling is named: half of a pass's.
pub const DEFAULT_FAIL_CEILING: f64 = 0.5;

impl RlBuild {
    /// The reward shape this asks for; `facts` holds the naive floors a
    /// `shaped-partial` reward measures from.
    pub(crate) fn shape(&self, facts: &PracticeFacts) -> Result<RewardShape, OrchestratorError> {
        let refused = |e: splinter_data::ViewError| OrchestratorError::Refused(e.to_string());
        let shape = self.verdict_shape(facts)?;
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
    fn verdict_shape(&self, facts: &PracticeFacts) -> Result<RewardShape, OrchestratorError> {
        let refused = |e: splinter_data::ViewError| OrchestratorError::Refused(e.to_string());
        if self.reward == RewardPreset::ShapedPartial {
            if !self.reward_metrics.is_empty() {
                return Err(OrchestratorError::Refused(format!(
                    "--reward shaped-partial credits {PARTIAL_METRIC} alone; it takes no \
                     --reward-metric"
                )));
            }
            return RewardShape::partial(
                self.fail_ceiling.unwrap_or(DEFAULT_FAIL_CEILING),
                PARTIAL_METRIC,
                facts.naive_floors(),
                facts.sources.clone(),
            )
            .map_err(refused);
        }
        match (self.reward_metrics.is_empty(), self.fail_ceiling) {
            (true, None) => Ok(RewardShape::unshaped()),
            (true, Some(_)) => Err(OrchestratorError::Refused(
                "--fail-ceiling bounds what the --reward-metric terms (or --reward \
                 shaped-partial) earn a failed attempt; name at least one metric"
                    .into(),
            )),
            (false, ceiling) => RewardShape::shaped(
                ceiling.unwrap_or(DEFAULT_FAIL_CEILING),
                self.reward_metrics.clone(),
            )
            .map_err(refused),
        }
    }

    /// Whether no pairing setting was named.
    pub(crate) fn pairing_is_default(&self) -> bool {
        self.pairs_per_group.is_none() && self.pair_seed.is_none()
    }

    /// Whether no reward or advantage setting was named.
    pub(crate) fn rewards_are_default(&self) -> bool {
        self.reward == RewardPreset::Verdict
            && self.reward_metrics.is_empty()
            && self.fail_ceiling.is_none()
            && self.advantage == AdvantageKind::default()
            && !self.cliff_distill
            && self.overlong_max_tokens.is_none()
            && self.overlong_buffer.is_none()
    }
}
