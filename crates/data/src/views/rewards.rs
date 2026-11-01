// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reinforcement learning from the verified
// rewards of agent rollouts for its clients. If your team needs expertise in
// training agents against verifiers that do not trust them, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Rewards and advantages: the numbers a policy-gradient run learns from.
//!
//! An attempt (trial) earns a scalar [`RewardShape::reward`] from the verdict
//! of an independent verifier and the metrics it measured. Attempts of one
//! task by one policy form a group, and each attempt's advantage is how much
//! better it did than the group ([`AdvantageKind`]). Splinter's core knows no
//! domain: which metrics shape a failed attempt's reward, and how, is the
//! caller's configuration.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ViewError;

/// How far apart the best and the worst reward of a group must be for the
/// group to say anything: closer, they are the same reward.
const MIN_SPREAD: f64 = 1e-12;
/// Added to the spread before dividing by it, so a nearly flat group does not
/// blow its advantages up.
const SPREAD_EPSILON: f64 = 1e-6;
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
/// metrics configured a failure earns nothing.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RewardShape {
    fail_ceiling: f64,
    weights: Vec<(String, f64)>,
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
        })
    }

    /// The reward of the attempt `trial`: `1.0` if it `passed`, else the
    /// shaped share of the ceiling its `metrics` (the verifier's object of
    /// named measurements) earn.
    ///
    /// # Errors
    /// A failed attempt lacks a configured metric, or one is not a number in
    /// `[0, 1]`. Neither is read as zero: an unmeasured attempt is not a bad
    /// one.
    pub(crate) fn reward(
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

/// How the advantage of an attempt is read from the rewards of its group.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AdvantageKind {
    /// The reward less the group mean, over the group's standard deviation
    /// (population, plus a small constant): GRPO's.
    #[default]
    Standardised,
    /// The reward less the group mean.
    MeanCentred,
}

impl AdvantageKind {
    /// The advantage of each reward of a group, in order; `None` when the
    /// group says nothing: a single attempt, or all rewards the same.
    pub(crate) fn advantages(self, rewards: &[f64]) -> Option<Vec<f64>> {
        if rewards.len() < 2 {
            return None;
        }
        let (min, max) = rewards
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), r| {
                (lo.min(*r), hi.max(*r))
            });
        if max - min < MIN_SPREAD {
            return None;
        }
        let n = rewards.len() as f64;
        let mean = rewards.iter().sum::<f64>() / n;
        let spread = match self {
            Self::Standardised => {
                (rewards.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n).sqrt()
                    + SPREAD_EPSILON
            }
            Self::MeanCentred => 1.0,
        };
        Some(rewards.iter().map(|r| (r - mean) / spread).collect())
    }
}

/// One attempt of a group.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrialReport {
    /// The trial's name.
    pub trial: String,
    /// The reward the attempt earned.
    pub reward: f64,
    /// Its advantage over the group; zero for a group that says nothing.
    pub advantage: f64,
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
    /// Whether the group has advantages to train on: more than one attempt,
    /// and not all of them alike.
    pub signal: bool,
}
