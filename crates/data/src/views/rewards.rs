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
//! [`AdvantageKind::Positive`], the floor at zero brain clamps them to. Splinter's
//! core knows no domain: which metrics shape a failed attempt's reward, and
//! how, is the caller's configuration.

use std::collections::BTreeMap;

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
/// metrics configured a failure earns nothing. Under [`RewardShape::partial`]
/// a failure earns its credit above its task's naive floor instead
/// ([`PartialCredit`]). An [`Overlong`] penalty, when set, is added to either.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RewardShape {
    fail_ceiling: f64,
    weights: Vec<(String, f64)>,
    partial: Option<PartialCredit>,
    overlong: Option<Overlong>,
}

/// The `shaped-partial` reward: a failed attempt earns
/// `credit * max(0, (f - f_naive) / (1 - f_naive))`, where `f` is the metric
/// the verifier measured (the share of its weighted checks the attempt
/// passed) and `f_naive` the best share a naive answer to the same task
/// earns, as the examiner measured it when it admitted the task. Credit a
/// naive answer would earn is no credit: an attempt has to do better than
/// one to earn any. A task without a floor is measured from zero, and the
/// projection counts the groups measured that way. The strict pass stays the
/// gate: no failure earns `credit` or more, and `credit` is below a pass's 1.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PartialCredit {
    /// The metric `f`, as the verifier names it.
    pub metric: String,
    /// `credit`: the most a failure earns, in `[0, 1)`.
    pub credit: f64,
    /// Each task's naive floor `f_naive`, in `[0, 1]`, by the task's content
    /// digest (the verdict's `task.digest`).
    pub floors: BTreeMap<String, f64>,
    /// Where the floors came from (the sha256 of each examiner manifest
    /// read); empty when none was read, so every task was measured from zero.
    #[serde(default)]
    pub sources: Vec<String>,
    /// The groups whose task had no floor, so `f_naive` was zero; set by the
    /// projection.
    #[serde(default)]
    pub groups_without_floor: usize,
}

impl PartialCredit {
    /// The floor of the task whose content digest is `digest`, if it has one.
    #[must_use]
    pub fn floor(&self, digest: Option<&str>) -> Option<f64> {
        digest.and_then(|d| self.floors.get(d)).copied()
    }

    /// What a failure measuring `f` on a task with `floor` (`None`: zero)
    /// earns.
    #[must_use]
    pub fn earned(&self, f: f64, floor: Option<f64>) -> f64 {
        let floor = floor.unwrap_or(0.0);
        if floor >= 1.0 {
            // A naive answer earns every check: no failure does better.
            return 0.0;
        }
        self.credit * ((f - floor) / (1.0 - floor)).max(0.0)
    }
}

/// DAPO's soft over-length penalty: an attempt that wrote more completion
/// tokens than `max_tokens - buffer` loses reward linearly, down to `-1` at
/// `max_tokens`, and `-1` beyond it, so the policy learns to finish within its
/// budget instead of being cut off by it.
///
/// With `L = max_tokens`, `B = buffer` and `n` the attempt's completion
/// tokens, the penalty is `0` for `n <= L - B`, `(L - B - n) / B` for
/// `L - B < n <= L`, and `-1` for `n > L`.
///
/// The invariant `0 < B <= L` holds for every value: [`Overlong::new`] checks
/// it, and so does reading one from a manifest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "OverlongThresholds")]
pub struct Overlong {
    /// `L`: the completion tokens at and past which the full penalty applies.
    max_tokens: u64,
    /// `B`: the tokens before `max_tokens` over which the penalty grows.
    buffer: u64,
}

/// An [`Overlong`] as a manifest spells it, before the invariant is checked.
#[derive(Deserialize)]
struct OverlongThresholds {
    max_tokens: u64,
    buffer: u64,
}

impl TryFrom<OverlongThresholds> for Overlong {
    type Error = ViewError;

    fn try_from(thresholds: OverlongThresholds) -> Result<Self, ViewError> {
        Self::new(thresholds.max_tokens, thresholds.buffer)
    }
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
            partial: None,
            overlong: None,
        })
    }

    /// The `shaped-partial` shape ([`PartialCredit`]): a failure earns up to
    /// `credit` of `metric` above its task's naive floor in `floors` (by
    /// task content digest; zero for a task without one), read from
    /// `sources`.
    ///
    /// # Errors
    /// The credit is not in `[0, 1)`, the metric has no name, or a floor is
    /// not in `[0, 1]`.
    pub fn partial(
        credit: f64,
        metric: &str,
        floors: BTreeMap<String, f64>,
        sources: Vec<String>,
    ) -> Result<Self, ViewError> {
        if !(credit.is_finite() && (0.0..1.0).contains(&credit)) {
            return Err(parameter(
                "partial credit",
                format!("{credit} is not in [0, 1): a failure must never earn as much as a pass"),
            ));
        }
        if metric.is_empty() {
            return Err(parameter("partial credit", "the metric has no name".into()));
        }
        if let Some((task, floor)) = floors
            .iter()
            .find(|(_, f)| !(f.is_finite() && (0.0..=1.0).contains(*f)))
        {
            return Err(parameter(
                "naive floor",
                format!("task {task}'s floor {floor} is not in [0, 1]"),
            ));
        }
        Ok(Self {
            partial: Some(PartialCredit {
                metric: metric.to_string(),
                credit,
                floors,
                sources,
                groups_without_floor: 0,
            }),
            ..Self::default()
        })
    }

    /// The partial credit the shape gives a failure, under `shaped-partial`.
    #[must_use]
    pub fn partial_credit(&self) -> Option<&PartialCredit> {
        self.partial.as_ref()
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

    /// The reward of the attempt `trial` at the task whose content digest is
    /// `task_digest`: `1.0` if it `passed`, else the shaped share of the
    /// ceiling its `metrics` (the verifier's object of named measurements)
    /// earn, or its partial credit above the task's floor; plus the
    /// over-length penalty of its `completion_tokens` when the shape has one.
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
        task_digest: Option<&str>,
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
        Ok(self.verdict_reward(passed, metrics, task_digest, trial)? + penalty)
    }

    /// [`RewardShape::reward`] before the over-length penalty.
    fn verdict_reward(
        &self,
        passed: bool,
        metrics: Option<&Value>,
        task_digest: Option<&str>,
        trial: &str,
    ) -> Result<f64, ViewError> {
        if passed {
            return Ok(1.0);
        }
        if let Some(partial) = &self.partial {
            let f = metric(metrics, &partial.metric, trial)?;
            return Ok(partial.earned(f, partial.floor(task_digest)));
        }
        let mut shaped = 0.0;
        for (name, weight) in &self.weights {
            shaped += weight * metric(metrics, name, trial)?;
        }
        Ok(self.fail_ceiling * shaped)
    }
}

/// The metric `name` of a failed attempt's `metrics`, in `[0, 1]`.
fn metric(metrics: Option<&Value>, name: &str, trial: &str) -> Result<f64, ViewError> {
    metrics
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
        })
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

/// What brain does with an advantage below zero: the training job's
/// `advantage_floor`, by its spelling on the wire.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AdvantageFloor {
    /// Kept: an attempt below its group's baseline is pushed down.
    None,
    /// Clamped at zero after the whole group set the baseline: an attempt
    /// below it takes part in the group's statistics and trains nothing.
    Zero,
}

impl AdvantageFloor {
    /// The floor's name as brain's training job takes it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Zero => "zero",
        }
    }
}

/// How brain reads the advantages of a group's attempts from its rewards:
/// every attempt of a group whose rewards differ is written, and every one
/// takes part in the group's baseline.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AdvantageKind {
    /// Brain standardises the group's rewards (GRPO's).
    #[default]
    Standardised,
    /// Brain subtracts the group mean.
    MeanCentred,
    /// Brain subtracts the mean of the others (RLOO).
    LeaveOneOut,
    /// Brain standardises the group's rewards and clamps the advantages at
    /// zero ([`AdvantageFloor::Zero`]): the attempts below the baseline set it
    /// and are not pushed down. Wrong answers share most of their tokens with
    /// right ones, so pushing a wrong answer down also pushes down the tokens
    /// that were right in it; this keeps the better attempts without that
    /// damage, and loses the signal of what to avoid.
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

    /// The floor brain applies to the advantages this kind writes.
    #[must_use]
    pub fn floor(self) -> AdvantageFloor {
        match self {
            Self::Positive => AdvantageFloor::Zero,
            Self::Standardised | Self::MeanCentred | Self::LeaveOneOut => AdvantageFloor::None,
        }
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
    /// group says something. A report written before records carried rewards
    /// reads as not written.
    #[serde(default)]
    pub written: bool,
    /// Whether it was written only to set its group's baseline: no
    /// conversation of it could be rendered, so one record carries its reward
    /// and group and trains nothing (`baseline_only`).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub baseline_only: bool,
    /// How many of its conversations (experiences) the view saw.
    pub conversations: usize,
    /// What its model requests cost: completion plus uncached prompt tokens,
    /// as its verdict's score usage counts them; `None` when it counts none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<u64>,
    /// The completion tokens it generated, as counted; `None` when not.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u64>,
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
    /// The naive floor its failures' partial credit was measured above,
    /// under `shaped-partial`; `None` under any other shape, or when its task
    /// had none (measured from zero).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub naive_floor: Option<f64>,
    /// The task's content digest, as its verdicts record it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub digest: Option<String>,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn credit(c: f64) -> PartialCredit {
        PartialCredit {
            metric: "functional_correctness".into(),
            credit: c,
            floors: BTreeMap::new(),
            sources: Vec::new(),
            groups_without_floor: 0,
        }
    }

    /// `c * max(0, (f - f_naive) / (1 - f_naive))` by hand at its edges: no
    /// floor is a floor of zero, a share at or under the floor earns nothing,
    /// a naive answer that earns every check leaves nothing to earn, and a
    /// failure that passed every weighted check still earns only `c`.
    #[test]
    fn partial_credit_is_the_share_above_the_naive_floor_scaled_to_the_credit() {
        let half = credit(0.5);
        for (f, floor, expected) in [
            (0.6, None, 0.3),
            (0.75, Some(0.5), 0.25),
            (0.2, Some(0.2), 0.0),
            (0.1, Some(0.2), 0.0),
            (1.0, Some(0.2), 0.5),
            (1.0, Some(1.0), 0.0),
            (0.0, None, 0.0),
        ] {
            let earned = half.earned(f, floor);
            assert!(
                (earned - expected).abs() < 1e-12,
                "f {f} floor {floor:?}: {earned}, by hand {expected}"
            );
        }
        assert_eq!(credit(0.0).earned(0.9, Some(0.1)), 0.0);
    }

    /// A pass earns one whatever its metrics say; a group is a signal only
    /// with two rewards that differ.
    #[test]
    fn a_pass_earns_one_and_only_a_spread_of_two_or_more_rewards_signals() {
        let shape = RewardShape::partial(0.5, "m", BTreeMap::new(), Vec::new()).unwrap();
        let reward = |passed, m: f64| {
            shape
                .reward(
                    passed,
                    Some(&serde_json::json!({ "m": m })),
                    None,
                    None,
                    "t",
                )
                .unwrap()
        };
        assert_eq!(reward(true, 0.0), 1.0);
        assert!((reward(false, 0.8) - 0.4).abs() < 1e-12);
        assert!(!signals(&[1.0]));
        assert!(!signals(&[0.3, 0.3, 0.3]));
        assert!(signals(&[0.3, 0.3, 0.0]));
        assert!(!signals(&[]));
    }
}
