// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training loops that notice a model moving the
// wrong way before it is released, for its clients. If your team needs
// expertise in monitoring continuously trained models, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The watchdog's first two layers.
//!
//! * **Invariants** ([`Bounds`]): fixed, registered bounds on the health of
//!   the training itself (a non-finite step figure, the share of steps the
//!   trainer's guard skipped, the largest importance ratio, a version's
//!   truncation and infrastructure-failure rates). A breach pauses the
//!   intake of new groups and raises an alarm; it never rolls back by
//!   itself, since it says the machinery is unsafe, not that the policy got
//!   worse.
//! * **Drift** ([`Drift`]): one-sided cumulative-sum detectors on the
//!   change of a version's behaviour from the trailing mean of the
//!   promoted versions' (the reference): the policy's entropy falling, its
//!   false-completion or repeated-call rate rising, its reward falling. A
//!   detector that crosses raises an alarm and asks for an evaluation now;
//!   only the paired evaluation decides.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use super::super::health::{StepHealth, VersionHealth};

/// The registered bounds of the invariants.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Bounds {
    /// The largest share of the last `window` steps the trainer's guard may
    /// skip.
    pub guard_skip_rate: f64,
    /// The largest importance ratio a step may report.
    pub ratio_max: f64,
    /// The largest share of a version's rollouts a limit may end.
    pub truncation_rate: f64,
    /// The largest share of a version's rollout attempts the infrastructure
    /// may fail.
    pub infra_failure_rate: f64,
    /// Steps the guard-skip rate is taken over.
    pub window: u32,
    /// Rollouts a version needs before its rates are read.
    pub min_rollouts: u32,
}

/// A breached invariant.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Breach {
    /// What was breached.
    pub invariant: &'static str,
    /// The value seen.
    pub value: f64,
    /// Its bound.
    pub bound: f64,
    /// The version or step it was seen at.
    pub at: u64,
}

impl Bounds {
    /// The breaches among the learner's `steps` and `skipped` steps.
    #[must_use]
    pub fn steps(&self, steps: &[StepHealth], skipped: usize) -> Vec<Breach> {
        let mut out = Vec::new();
        for step in steps {
            if !step.finite() {
                out.push(Breach {
                    invariant: "non_finite_step",
                    value: f64::NAN,
                    bound: 0.0,
                    at: step.step,
                });
            }
            if let Some(ratio) = step.ratio_max.filter(|r| *r > self.ratio_max) {
                out.push(Breach {
                    invariant: "ratio_max",
                    value: ratio,
                    bound: self.ratio_max,
                    at: step.step,
                });
            }
        }
        let window = usize::try_from(self.window).unwrap_or(usize::MAX).max(1);
        if steps.len() >= window {
            #[allow(clippy::cast_precision_loss)]
            let rate = skipped.min(window) as f64 / window as f64;
            if rate > self.guard_skip_rate {
                out.push(Breach {
                    invariant: "guard_skip_rate",
                    value: rate,
                    bound: self.guard_skip_rate,
                    at: steps.last().map_or(0, |s| s.step),
                });
            }
        }
        out
    }

    /// The breaches of version `version`'s rollouts.
    #[must_use]
    pub fn version(&self, version: u64, health: &VersionHealth) -> Vec<Breach> {
        if health.rollouts < u64::from(self.min_rollouts) {
            return Vec::new();
        }
        [
            (
                "truncation_rate",
                health.truncation_rate(),
                self.truncation_rate,
            ),
            (
                "infra_failure_rate",
                health.infra_failure_rate(),
                self.infra_failure_rate,
            ),
        ]
        .into_iter()
        .filter_map(|(invariant, value, bound)| {
            value.filter(|v| *v > bound).map(|value| Breach {
                invariant,
                value,
                bound,
                at: version,
            })
        })
        .collect()
    }
}

/// A behaviour a drift detector follows, and the direction that is worse.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Signal {
    /// The policy's entropy (from the steps that trained toward a version):
    /// a fall is worse.
    Entropy,
    /// The false-completion rate: a rise is worse.
    FalseCompletion,
    /// The repeated-call rate: a rise is worse.
    RepeatedCalls,
    /// The reward: a fall is worse.
    Reward,
}

impl Signal {
    /// Every signal.
    pub const ALL: [Self; 4] = [
        Self::Entropy,
        Self::FalseCompletion,
        Self::RepeatedCalls,
        Self::Reward,
    ];

    /// Whether a rise of the signal is the worse direction.
    fn rise_is_worse(self) -> bool {
        matches!(self, Self::FalseCompletion | Self::RepeatedCalls)
    }

    /// The signal's value for a version: its rollouts', or the mean entropy
    /// of `entropy` (the steps that trained toward it).
    #[must_use]
    pub fn value(self, health: &VersionHealth, entropy: Option<f64>) -> Option<f64> {
        match self {
            Self::Entropy => entropy,
            Self::FalseCompletion => health.false_completion_rate(),
            Self::RepeatedCalls => health.repeated_call_rate(),
            Self::Reward => health.reward_mean(),
        }
    }
}

/// The registered parameters of the drift detectors.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DriftSettings {
    /// The change per version, in the signal's own unit, that a detector
    /// absorbs (its slack).
    pub slack: f64,
    /// The accumulated change past the slack that raises an alarm.
    pub threshold: f64,
    /// Promoted versions the reference is the mean of.
    pub reference_window: u32,
}

/// One one-sided cumulative-sum detector.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Cusum {
    /// The signal.
    pub signal: Signal,
    /// The accumulated change in the worse direction.
    pub sum: f64,
    reference: VecDeque<f64>,
}

impl Cusum {
    fn new(signal: Signal) -> Self {
        Self {
            signal,
            sum: 0.0,
            reference: VecDeque::new(),
        }
    }

    /// The reference: the mean of the promoted versions' values.
    #[must_use]
    pub fn reference(&self) -> Option<f64> {
        #[allow(clippy::cast_precision_loss)]
        (!self.reference.is_empty())
            .then(|| self.reference.iter().sum::<f64>() / self.reference.len() as f64)
    }
}

/// The drift detectors of a run.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Drift {
    settings: DriftSettings,
    /// One detector per signal.
    pub detectors: Vec<Cusum>,
}

/// A detector that crossed.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DriftAlarm {
    /// Its signal.
    pub signal: Signal,
    /// Its accumulated change.
    pub sum: f64,
    /// The reference it is measured from.
    pub reference: f64,
    /// The version that crossed.
    pub version: u64,
}

impl Drift {
    /// Detectors on every signal, held to `settings`.
    #[must_use]
    pub fn new(settings: DriftSettings) -> Self {
        Self {
            settings,
            detectors: Signal::ALL.into_iter().map(Cusum::new).collect(),
        }
    }

    /// A promoted version's values join the reference (the newest
    /// `reference_window` of them), and every detector starts afresh from it.
    pub fn promoted(&mut self, values: &[(Signal, Option<f64>)]) {
        let window = usize::try_from(self.settings.reference_window)
            .unwrap_or(usize::MAX)
            .max(1);
        for detector in &mut self.detectors {
            if let Some(Some(v)) = values
                .iter()
                .find(|(s, _)| *s == detector.signal)
                .map(|(_, v)| *v)
            {
                detector.reference.push_back(v);
                while detector.reference.len() > window {
                    detector.reference.pop_front();
                }
            }
            detector.sum = 0.0;
        }
    }

    /// A candidate version's values: each detector adds its change from the
    /// reference in the worse direction, less the slack; the ones that
    /// cross are returned.
    pub fn observe(&mut self, version: u64, values: &[(Signal, Option<f64>)]) -> Vec<DriftAlarm> {
        let mut alarms = Vec::new();
        for detector in &mut self.detectors {
            let (Some(reference), Some(Some(value))) = (
                detector.reference(),
                values
                    .iter()
                    .find(|(s, _)| *s == detector.signal)
                    .map(|(_, v)| *v),
            ) else {
                continue;
            };
            let change = if detector.signal.rise_is_worse() {
                value - reference
            } else {
                reference - value
            };
            detector.sum = (detector.sum + change - self.settings.slack).max(0.0);
            if detector.sum > self.settings.threshold {
                alarms.push(DriftAlarm {
                    signal: detector.signal,
                    sum: detector.sum,
                    reference,
                    version,
                });
            }
        }
        alarms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn health(rollouts: u64, truncated: u64, failures: u64) -> VersionHealth {
        VersionHealth {
            rollouts,
            truncated,
            infra_failures: failures,
            ..VersionHealth::default()
        }
    }

    fn bounds() -> Bounds {
        Bounds {
            guard_skip_rate: 0.25,
            ratio_max: 10.0,
            truncation_rate: 0.5,
            infra_failure_rate: 0.3,
            window: 4,
            min_rollouts: 8,
        }
    }

    #[test]
    fn invariants_are_breached_only_past_their_registered_bounds() {
        let step = |n, ratio: f64, kl: f64| StepHealth {
            step: n,
            ratio_max: Some(ratio),
            kl: Some(kl),
            ..StepHealth::default()
        };
        let steps = [
            step(1, 2.0, 0.1),
            step(2, 12.0, 0.1),
            step(3, 1.0, f64::NAN),
            step(4, 1.0, 0.1),
        ];
        let breaches: Vec<&str> = bounds()
            .steps(&steps, 2)
            .iter()
            .map(|b| b.invariant)
            .collect();
        assert_eq!(
            breaches,
            ["ratio_max", "non_finite_step", "guard_skip_rate"]
        );
        assert!(bounds()
            .steps(&steps[..3], 3)
            .iter()
            .all(|b| b.invariant != "guard_skip_rate"));
        assert!(
            bounds().version(1, &health(4, 4, 4)).is_empty(),
            "too few rollouts to read"
        );
        let breached: Vec<&str> = bounds()
            .version(1, &health(10, 6, 5))
            .iter()
            .map(|b| b.invariant)
            .collect();
        assert_eq!(breached, ["truncation_rate", "infra_failure_rate"]);
        assert!(
            bounds().version(1, &health(10, 5, 4)).is_empty(),
            "at a bound is within it"
        );
    }

    #[test]
    fn a_detector_follows_change_from_the_promoted_reference_not_level() {
        let mut drift = Drift::new(DriftSettings {
            slack: 0.05,
            threshold: 0.25,
            reference_window: 2,
        });
        let rate = |v: f64| {
            vec![
                (Signal::FalseCompletion, Some(v)),
                (Signal::Reward, Some(0.5)),
            ]
        };
        // A high but steady false-completion rate is the reference, not an alarm.
        drift.promoted(&rate(0.6));
        for version in 1..20 {
            assert!(
                drift.observe(version, &rate(0.62)).is_empty(),
                "version {version}"
            );
        }
        // A rise from it accumulates until it crosses.
        let mut crossed = None;
        for version in 20..30 {
            if let Some(alarm) = drift.observe(version, &rate(0.75)).first() {
                crossed = Some((version, alarm.signal));
                break;
            }
        }
        assert_eq!(crossed, Some((22, Signal::FalseCompletion)));
        // A promotion moves the reference and clears the sums.
        drift.promoted(&rate(0.75));
        assert!(drift.observe(31, &rate(0.75)).is_empty());
        assert!(drift.detectors.iter().all(|d| d.sum < 0.25));
    }
}
