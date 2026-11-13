// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for telling, step by step, whether a
// reinforcement-learning run is moving the policy or breaking it, for its
// clients. If your team needs expertise in the training diagnostics of
// policy-gradient runs you can procure our services by sending an email to
// info@swedishembedded.com.

//! The numbers of one training step a gate reads.
//!
//! Real-weight runs that ended with 10.7 to 14.7 percent of their groups
//! carrying any advantage, a KL to the reference of 0.0008 to 0.016 and a
//! few percent of tokens clipped could not distinguish "reinforcement
//! learning does not work here" from "the experiment could not have seen an
//! effect". The controller therefore logs, for every step, the share of
//! informative groups, the KL to the reference, the clip fraction and the
//! policy's entropy, and [`SignalBounds::assess`] turns them into findings.
//! A number that was not measured is absent and is itself a finding: it is
//! never read as `0`.

use serde::{Deserialize, Serialize};

/// What one training step measured.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct StepSignals {
    /// The step, counted from 1 over the campaign.
    pub step: u32,
    /// The policy version the step trained from.
    pub policy_version: u64,
    /// Informative groups among those offered while building the batch.
    pub informative_fraction: Option<f64>,
    /// Mean KL to the reference policy.
    pub kl: Option<f64>,
    /// Share of tokens with a non-zero advantage whose ratio was clipped.
    pub clip_fraction: Option<f64>,
    /// The policy's entropy per generated token, estimated from the
    /// log-probabilities of the tokens it sampled at temperature 1.
    pub entropy: Option<f64>,
    /// Mean reward of the batch's attempts.
    pub mean_reward: Option<f64>,
}

/// The bounds a step's signals are held to; a bound left `None` is not
/// checked.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SignalBounds {
    /// Fewest informative groups, as a share of those offered. Below it the
    /// pool is off the frontier.
    pub min_informative_fraction: Option<f64>,
    /// Largest mean KL to the reference before the run is halted.
    pub max_kl: Option<f64>,
    /// Largest clip fraction: past it the updates are mostly cut off.
    pub max_clip_fraction: Option<f64>,
    /// Smallest entropy: below it exploration has collapsed.
    pub min_entropy: Option<f64>,
}

impl Default for SignalBounds {
    fn default() -> Self {
        Self {
            min_informative_fraction: Some(0.25),
            max_kl: Some(0.5),
            max_clip_fraction: Some(0.5),
            min_entropy: None,
        }
    }
}

/// One thing a step's signals showed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "finding", rename_all = "snake_case")]
pub enum Finding {
    /// Too few groups carried an advantage.
    Starved {
        /// What was measured.
        observed: f64,
        /// The bound.
        min: f64,
    },
    /// The policy moved too far from the reference.
    KlTooHigh {
        /// What was measured.
        observed: f64,
        /// The bound.
        max: f64,
    },
    /// Too many tokens were clipped.
    ClipTooHigh {
        /// What was measured.
        observed: f64,
        /// The bound.
        max: f64,
    },
    /// The policy's entropy fell below the floor.
    EntropyCollapsed {
        /// What was measured.
        observed: f64,
        /// The bound.
        min: f64,
    },
    /// A signal a bound names was not measured, or was not finite.
    Unmeasured {
        /// Which one.
        signal: String,
    },
}

impl Finding {
    /// Whether the finding stops the run: an unstable policy is stopped, a
    /// starved or unmeasured step is reported and handled by the controller.
    #[must_use]
    pub fn halts(&self) -> bool {
        matches!(
            self,
            Self::KlTooHigh { .. } | Self::ClipTooHigh { .. } | Self::EntropyCollapsed { .. }
        )
    }
}

impl SignalBounds {
    /// What `signals` show against the bounds.
    #[must_use]
    pub fn assess(&self, signals: &StepSignals) -> Vec<Finding> {
        let mut findings = Vec::new();
        let mut check = |name: &str,
                         bound: Option<f64>,
                         value: Option<f64>,
                         breach: &dyn Fn(f64, f64) -> Option<Finding>| {
            let Some(bound) = bound else { return };
            match value.filter(|v| v.is_finite()) {
                None => findings.push(Finding::Unmeasured {
                    signal: name.to_string(),
                }),
                Some(value) => findings.extend(breach(value, bound)),
            }
        };
        check(
            "informative_fraction",
            self.min_informative_fraction,
            signals.informative_fraction,
            &|observed, min| (observed < min).then_some(Finding::Starved { observed, min }),
        );
        check("kl", self.max_kl, signals.kl, &|observed, max| {
            (observed > max).then_some(Finding::KlTooHigh { observed, max })
        });
        check(
            "clip_fraction",
            self.max_clip_fraction,
            signals.clip_fraction,
            &|observed, max| (observed > max).then_some(Finding::ClipTooHigh { observed, max }),
        );
        check(
            "entropy",
            self.min_entropy,
            signals.entropy,
            &|observed, min| {
                (observed < min).then_some(Finding::EntropyCollapsed { observed, min })
            },
        );
        findings
    }
}

/// The entropy of a policy estimated from the log-probabilities it gave the
/// tokens it sampled: the mean of `-log p`. Exact in expectation for tokens
/// sampled at temperature 1 with no truncation; `None` with no token.
#[must_use]
pub fn sampled_entropy(log_probs: impl IntoIterator<Item = f64>) -> Option<f64> {
    let (sum, count) = log_probs
        .into_iter()
        .fold((0.0, 0usize), |(sum, count), lp| (sum - lp, count + 1));
    (count > 0).then(|| sum / count as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn healthy() -> StepSignals {
        StepSignals {
            step: 1,
            informative_fraction: Some(0.6),
            kl: Some(0.02),
            clip_fraction: Some(0.1),
            entropy: Some(0.9),
            ..StepSignals::default()
        }
    }

    #[test]
    fn a_healthy_step_has_no_finding() {
        assert!(SignalBounds::default().assess(&healthy()).is_empty());
    }

    #[test]
    fn each_breached_bound_is_a_finding_and_only_instability_halts() {
        let bounds = SignalBounds {
            min_entropy: Some(0.5),
            ..SignalBounds::default()
        };
        let starved = StepSignals {
            informative_fraction: Some(0.107),
            ..healthy()
        };
        let findings = bounds.assess(&starved);
        assert_eq!(findings.len(), 1);
        assert!(matches!(findings[0], Finding::Starved { .. }));
        assert!(
            !findings[0].halts(),
            "a starved batch is routed, not a crash"
        );

        let unstable = StepSignals {
            kl: Some(0.9),
            clip_fraction: Some(0.8),
            entropy: Some(0.1),
            ..healthy()
        };
        let findings = bounds.assess(&unstable);
        assert_eq!(findings.len(), 3);
        assert!(findings.iter().all(Finding::halts));
    }

    #[test]
    fn an_unmeasured_signal_a_bound_names_is_a_finding_not_a_zero() {
        let signals = StepSignals {
            kl: None,
            entropy: Some(f64::NAN),
            ..healthy()
        };
        let bounds = SignalBounds {
            min_entropy: Some(0.1),
            ..SignalBounds::default()
        };
        let findings = bounds.assess(&signals);
        assert_eq!(
            findings,
            vec![
                Finding::Unmeasured {
                    signal: "kl".into()
                },
                Finding::Unmeasured {
                    signal: "entropy".into()
                }
            ]
        );
        assert!(findings.iter().all(|f| !f.halts()));
        let unbounded = SignalBounds {
            min_informative_fraction: None,
            max_kl: None,
            max_clip_fraction: None,
            min_entropy: None,
        };
        assert!(unbounded.assess(&StepSignals::default()).is_empty());
    }

    #[test]
    fn entropy_is_the_mean_negative_log_probability_of_sampled_tokens() {
        let entropy = sampled_entropy([-0.5, -1.5, -1.0]).unwrap();
        assert!((entropy - 1.0).abs() < 1e-12);
        assert_eq!(sampled_entropy(std::iter::empty()), None);
    }
}
