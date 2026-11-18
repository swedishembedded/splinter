// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reinforcement learning from the verified
// rewards of agent rollouts for its clients. If your team needs expertise in
// training agents against verifiers that do not trust them, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Rewards: the numbers a policy-gradient run learns from.
//!
//! An attempt (trial) earns a scalar [`RewardShape::reward`] from the verdict
//! of an independent verifier and the metrics it measured. Attempts of one
//! task by one policy form a group. The records carry each attempt's reward
//! and its group; brain turns a group's rewards into the attempts' advantages
//! (its `advantage_norm`), so how much better an attempt did is computed in
//! one place. [`AdvantageKind`] selects that normalisation and, for
//! [`AdvantageKind::Positive`], which attempts are written at all. Splinter's
//! core knows no domain: which metrics shape a failed attempt's reward, and
//! how, is the caller's configuration.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ViewError;

/// How far apart the best and the worst reward of a group must be for the
/// group to say anything: closer, they are the same reward.
const MIN_SPREAD: f64 = 1e-12;
/// How far the weights of a reward shape may be from summing to one.
const WEIGHT_SUM_TOLERANCE: f64 = 1e-9;

fn parameter(name: &'static str, reason: String) -> ViewError {
    ViewError::Parameter { name, reason }
}

/// How an attempt's verdict and metrics become its reward.
///
/// A pass earns `1.0`. A failure earns `fail_ceiling * sum(w_i * m_i)` over
/// the configured metrics, each in `[0, 1]`, the weights summing to one; as
/// the ceiling is below one, a failure never earns as much as a pass. With no
/// metrics configured a failure earns nothing. An [`Overlong`] penalty, when
/// set, is added to either.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RewardShape {
    fail_ceiling: f64,
    weights: Vec<(String, f64)>,
    overlong: Option<Overlong>,
}

/// DAPO's soft over-length penalty: an attempt that wrote more completion
/// tokens than `max_tokens - buffer` loses reward linearly, down to `-1` at
/// `max_tokens`, and `-1` beyond it, so the policy learns to finish within its
/// budget instead of being cut off by it.
///
/// With `L = max_tokens`, `B = buffer` and `n` the attempt's completion
/// tokens, the penalty is `0` for `n <= L - B`, `(L - B - n) / B` for
/// `L - B < n <= L`, and `-1` for `n > L`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Overlong {
    /// `L`: the completion tokens at and past which the full penalty applies.
    pub max_tokens: u64,
    /// `B`: the tokens before `max_tokens` over which the penalty grows.
    pub buffer: u64,
}

impl Overlong {
    /// The penalty, refused unless `0 < buffer <= max_tokens`.
    ///
    /// # Errors
    /// The buffer is zero or longer than the budget.
    pub fn new(max_tokens: u64, buffer: u64) -> Result<Self, ViewError> {
        if buffer == 0 || buffer > max_tokens {
            return Err(parameter(
                "over-length penalty",
                format!(
                    "the buffer {buffer} is not in (0, {max_tokens}]: the penalty grows over \
                     the buffer's tokens before the budget of {max_tokens}"
                ),
            ));
        }
        Ok(Self { max_tokens, buffer })
    }

    /// The penalty, in `[-1, 0]`, of an attempt that wrote `tokens`
    /// completion tokens.
    #[must_use]
    pub fn penalty(&self, tokens: u64) -> f64 {
        let soft = self.max_tokens - self.buffer;
        if tokens <= soft {
            0.0
        } else if tokens <= self.max_tokens {
            (soft as f64 - tokens as f64) / self.buffer as f64
        } else {
            -1.0
        }
    }
}

impl RewardShape {
    /// The reward a pass earns and a failure does not.
    #[must_use]
    pub fn unshaped() -> Self {
        Self::default()
    }

    /// A shape in which a failure earns up to `fail_ceiling` by `weights`.
    ///
    /// # Errors
    /// The ceiling is not in `[0, 1)`; no weight is given, one is not a
    /// finite non-negative number, a metric is named twice or empty, or the
    /// weights do not sum to one.
    pub fn shaped(fail_ceiling: f64, weights: Vec<(String, f64)>) -> Result<Self, ViewError> {
        if !(fail_ceiling.is_finite() && (0.0..1.0).contains(&fail_ceiling)) {
            return Err(parameter(
                "fail ceiling",
                format!(
                    "{fail_ceiling} is not in [0, 1): a failure must never earn as much as a pass"
                ),
            ));
        }
        if weights.is_empty() {
            return Err(parameter(
                "reward metric",
                "a shaped reward needs at least one metric weight".into(),
            ));
        }
        for (at, (name, weight)) in weights.iter().enumerate() {
            if name.is_empty() {
                return Err(parameter("reward metric", "a metric has no name".into()));
            }
            if !(weight.is_finite() && *weight >= 0.0) {
                return Err(parameter(
                    "reward metric",
                    format!("the weight of {name} is {weight}, not a finite non-negative number"),
                ));
            }
            if weights[..at].iter().any(|(other, _)| other == name) {
                return Err(parameter(
                    "reward metric",
                    format!("{name} is weighted twice"),
                ));
            }
        }
        let sum: f64 = weights.iter().map(|(_, w)| w).sum();
        if (sum - 1.0).abs() > WEIGHT_SUM_TOLERANCE {
            return Err(parameter(
                "reward metric",
                format!("the weights sum to {sum}, not 1"),
            ));
        }
        Ok(Self {
            fail_ceiling,
            weights,
            overlong: None,
        })
    }

    /// The same shape with `overlong`'s penalty added to every reward.
    #[must_use]
    pub fn with_overlong(self, overlong: Overlong) -> Self {
        Self {
            overlong: Some(overlong),
            ..self
        }
    }

    /// The over-length penalty the shape adds, if any.
    #[must_use]
    pub fn overlong(&self) -> Option<Overlong> {
        self.overlong
    }

    /// The reward of the attempt `trial`: `1.0` if it `passed`, else the
    /// shaped share of the ceiling its `metrics` (the verifier's object of
    /// named measurements) earn; plus the over-length penalty of its
    /// `completion_tokens` when the shape has one.
    ///
    /// # Errors
    /// A failed attempt lacks a configured metric, or one is not a number in
    /// `[0, 1]`; the shape penalises length and the attempt recorded no
    /// completion tokens. None is read as zero: an unmeasured attempt is not a
    /// bad one, nor a short one.
    pub(crate) fn reward(
        &self,
        passed: bool,
        metrics: Option<&Value>,
        completion_tokens: Option<u64>,
        trial: &str,
    ) -> Result<f64, ViewError> {
        let penalty = match self.overlong {
            None => 0.0,
            Some(overlong) => overlong.penalty(completion_tokens.ok_or_else(|| {
                parameter(
                    "over-length penalty",
                    format!("trial {trial} records no completion token count to penalise"),
                )
            })?),
        };
        Ok(self.verdict_reward(passed, metrics, trial)? + penalty)
    }

    /// [`RewardShape::reward`] before the over-length penalty.
    fn verdict_reward(
        &self,
        passed: bool,
        metrics: Option<&Value>,
        trial: &str,
    ) -> Result<f64, ViewError> {
        if passed {
            return Ok(1.0);
        }
        let mut shaped = 0.0;
        for (name, weight) in &self.weights {
            let metric = metrics
                .and_then(|m| m.get(name))
                .ok_or_else(|| {
                    parameter(
                        "reward metric",
                        format!("trial {trial} failed and records no metric {name}"),
                    )
                })?
                .as_f64()
                .filter(|m| m.is_finite() && (0.0..=1.0).contains(m))
                .ok_or_else(|| {
                    parameter(
                        "reward metric",
                        format!("metric {name} of trial {trial} is not a number in [0, 1]"),
                    )
                })?;
            shaped += weight * metric;
        }
        Ok(self.fail_ceiling * shaped)
    }
}

/// How brain turns the rewards of a group into advantages: the training
/// job's `advantage_norm`, by its spelling on the wire.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AdvantageNorm {
    /// The reward less the group mean.
    MeanCentred,
    /// The reward less the group mean, over the group's standard deviation
    /// (population, plus a small constant): GRPO's.
    ZScore,
    /// The reward less the mean of the group's other rewards (RLOO).
    LeaveOneOut,
}

impl AdvantageNorm {
    /// The normalisation's name as brain's training job takes it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::MeanCentred => "mean-centred",
            Self::ZScore => "z-score",
            Self::LeaveOneOut => "leave-one-out",
        }
    }
}

/// Which attempts of a group are written, and how brain reads their
/// advantages from the group's rewards.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AdvantageKind {
    /// Every attempt; brain standardises the group's rewards (GRPO's).
    #[default]
    Standardised,
    /// Every attempt; brain subtracts the group mean.
    MeanCentred,
    /// Every attempt; brain subtracts the mean of the others (RLOO).
    LeaveOneOut,
    /// Only the attempts whose reward is above their group's mean, and no
    /// record at all for the rest: nothing is pushed down. Wrong answers
    /// share most of their tokens with right ones, so pushing a wrong answer
    /// down also pushes down the tokens that were right in it; this keeps
    /// the better attempts without that damage, and loses the signal of what
    /// to avoid. brain standardises the rewards of the attempts written, so
    /// the kept ones of a group still need rewards that differ.
    Positive,
}

impl AdvantageKind {
    /// The normalisation brain applies to the records this kind writes.
    #[must_use]
    pub fn norm(self) -> AdvantageNorm {
        match self {
            Self::Standardised | Self::Positive => AdvantageNorm::ZScore,
            Self::MeanCentred => AdvantageNorm::MeanCentred,
            Self::LeaveOneOut => AdvantageNorm::LeaveOneOut,
        }
    }

    /// Which attempts of a group with `rewards` are written, in order; `None`
    /// when the group says nothing to brain: a single attempt, all rewards
    /// the same, or (under [`Self::Positive`]) fewer than two kept attempts
    /// whose rewards differ.
    #[must_use]
    pub fn keeps(self, rewards: &[f64]) -> Option<Vec<bool>> {
        if !signals(rewards) {
            return None;
        }
        if self != Self::Positive {
            return Some(vec![true; rewards.len()]);
        }
        let mean = rewards.iter().sum::<f64>() / rewards.len() as f64;
        let kept: Vec<bool> = rewards.iter().map(|r| *r > mean).collect();
        let survivors: Vec<f64> = rewards
            .iter()
            .zip(&kept)
            .filter_map(|(r, k)| k.then_some(*r))
            .collect();
        signals(&survivors).then_some(kept)
    }
}

/// Whether a group's rewards say anything: more than one, and not all alike.
#[must_use]
pub fn signals(rewards: &[f64]) -> bool {
    let (min, max) = rewards
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), r| {
            (lo.min(*r), hi.max(*r))
        });
    rewards.len() >= 2 && max - min >= MIN_SPREAD
}

/// One attempt of a group.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrialReport {
    /// The trial's name.
    pub trial: String,
    /// The reward the attempt earned.
    pub reward: f64,
    /// Whether its records were written (with its reward and group): its
    /// group says something, and [`AdvantageKind::Positive`] kept it. A
    /// report written before records carried rewards reads as not written.
    #[serde(default)]
    pub written: bool,
    /// How many of its conversations (experiences) the view saw.
    pub conversations: usize,
}

/// The attempts of one task by one policy that were compared.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GroupReport {
    /// The task's id as the verifier recorded it.
    pub task: String,
    /// The task's version.
    pub version: String,
    /// The policy that made the attempts: the model recorded on them.
    pub policy: String,
    /// The attempts, by trial name.
    pub trials: Vec<TrialReport>,
    /// Whether the group has rewards to compare: more than one attempt, and
    /// not all of them alike.
    pub signal: bool,
    /// What kind of group it is: the campaign counts groups by kind.
    #[serde(default)]
    pub kind: GroupKind,
    /// Whether its attempts became self-distillation records (a teacher's
    /// per-token advantage) and not scalar-advantage ones.
    #[serde(default)]
    pub distilled: bool,
}

/// What a group of attempts at one task can teach.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GroupKind {
    /// The attempts earned different rewards: the group-relative advantage
    /// says which did better (GRPO's signal).
    #[default]
    Graded,
    /// Every attempt failed and none outranks another: the advantage is zero.
    /// The cliff, where only privileged information can say what to do.
    AllFail,
    /// Every attempt passed alike (or the only attempt did): nothing to
    /// compare, nothing to correct.
    AllPass,
}

impl GroupKind {
    /// The kind as it is printed.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Graded => "graded",
            Self::AllFail => "all-fail",
            Self::AllPass => "all-pass",
        }
    }
}
