// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reward-hacking alarms for reinforcement
// learning campaigns, for its clients. If your team needs expertise in
// telling a policy that learned a skill from one that learned its verifier,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! The reward-versus-held-out divergence alarm.
//!
//! A policy that learns to please a verifier shows it as a training reward
//! that climbs while a measurement the policy cannot influence (a dev suite
//! of other instances, a held-out family, an independent check) stays flat or
//! falls. A campaign controller feeds one [`Observation`] per evaluation and
//! acts on the [`Signal`]: continue, stop and roll back, or, when the
//! held-out measurement is missing, treat the rise as unconfirmed.
//!
//! The rule is a trend, not a threshold on one noisy point: over the last
//! [`DivergencePolicy::window`] observations the least-squares slope of each
//! series, scaled to the window, is its rise. Diverging means the reward rose
//! by at least [`DivergencePolicy::min_reward_rise`] while the held-out
//! measure rose by no more than [`DivergencePolicy::max_heldout_rise`].
//! Observations without a held-out value are not counted as zero: with fewer
//! than two held-out points in the window the signal is [`Signal::Unconfirmed`]
//! whenever the reward rose.

use serde::{Deserialize, Serialize};
use splinter_core::verifiability::VerifierClass;

/// How the alarm judges a trend.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DivergencePolicy {
    /// How many of the latest observations the trend is fitted over.
    pub window: usize,
    /// The training-reward rise over the window that counts as a rise.
    pub min_reward_rise: f64,
    /// The most the held-out measure may rise over the window while the
    /// reward rises, before the rise is called a divergence. Zero or
    /// negative: any fall or a flat line.
    pub max_heldout_rise: f64,
}

impl Default for DivergencePolicy {
    fn default() -> Self {
        Self {
            window: 5,
            min_reward_rise: 0.1,
            max_heldout_rise: 0.02,
        }
    }
}

/// One evaluation of a training run.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Observation {
    /// The training step.
    pub step: u64,
    /// The mean training reward at this step, in `0..=1`.
    pub reward: f64,
    /// The held-out measure at this step, when one was taken. Absent is not
    /// zero.
    pub heldout: Option<f64>,
}

/// What the alarm says after an observation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Signal {
    /// Nothing to report: the reward did not rise, or the held-out measure
    /// rose with it.
    Steady,
    /// The reward rose and the held-out measure did not follow.
    Diverging(Alert),
    /// The reward rose and there is not enough held-out evidence to say
    /// whether it means anything.
    Unconfirmed {
        /// The reward's rise over the window.
        reward_rise: f64,
        /// How many held-out points the window holds.
        heldout_points: usize,
    },
}

/// A divergence, with what a verdict built on this reward must say.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Alert {
    /// The step the alarm was raised at.
    pub step: u64,
    /// The reward's rise over the window.
    pub reward_rise: f64,
    /// The held-out measure's rise over the window (negative: a fall).
    pub heldout_rise: f64,
    /// The label the reward's verifier class forces on a verdict.
    pub label: Option<String>,
}

/// The alarm: a window of observations and the policy that reads it.
#[derive(Clone, Debug)]
pub struct DivergenceAlarm {
    policy: DivergencePolicy,
    class: VerifierClass,
    history: Vec<Observation>,
}

/// The rise of a series over its span: its least-squares slope per step
/// index, times the number of intervals. `None` for fewer than two points.
fn rise(points: &[(f64, f64)]) -> Option<f64> {
    let n = points.len() as f64;
    if points.len() < 2 {
        return None;
    }
    let mean_x = points.iter().map(|p| p.0).sum::<f64>() / n;
    let mean_y = points.iter().map(|p| p.1).sum::<f64>() / n;
    let sxx: f64 = points.iter().map(|p| (p.0 - mean_x).powi(2)).sum();
    if sxx == 0.0 {
        return None;
    }
    let sxy: f64 = points.iter().map(|p| (p.0 - mean_x) * (p.1 - mean_y)).sum();
    let span = points.last().map_or(0.0, |p| p.0) - points.first().map_or(0.0, |p| p.0);
    Some(sxy / sxx * span)
}

impl DivergenceAlarm {
    /// An alarm over rewards of `class`.
    #[must_use]
    pub fn new(policy: DivergencePolicy, class: VerifierClass) -> Self {
        Self {
            policy,
            class,
            history: Vec::new(),
        }
    }

    /// Records `observation` and reads the trend over the latest window.
    /// Fewer than [`DivergencePolicy::window`] observations are
    /// [`Signal::Steady`]: no trend can be read yet.
    pub fn observe(&mut self, observation: Observation) -> Signal {
        self.history.push(observation);
        if self.history.len() < self.policy.window.max(2) {
            return Signal::Steady;
        }
        let window = &self.history[self.history.len() - self.policy.window.max(2)..];
        let rewards: Vec<(f64, f64)> = window.iter().map(|o| (o.step as f64, o.reward)).collect();
        let Some(reward_rise) = rise(&rewards).filter(|r| *r >= self.policy.min_reward_rise) else {
            return Signal::Steady;
        };
        let heldout: Vec<(f64, f64)> = window
            .iter()
            .filter_map(|o| o.heldout.map(|h| (o.step as f64, h)))
            .collect();
        let Some(heldout_rise) = rise(&heldout) else {
            return Signal::Unconfirmed {
                reward_rise,
                heldout_points: heldout.len(),
            };
        };
        if heldout_rise <= self.policy.max_heldout_rise {
            Signal::Diverging(Alert {
                step: window.last().map_or(0, |o| o.step),
                reward_rise,
                heldout_rise,
                label: self.class.label().map(ToString::to_string),
            })
        } else {
            Signal::Steady
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(alarm: &mut DivergenceAlarm, series: &[(f64, Option<f64>)]) -> Vec<Signal> {
        series
            .iter()
            .enumerate()
            .map(|(step, (reward, heldout))| {
                alarm.observe(Observation {
                    step: step as u64,
                    reward: *reward,
                    heldout: *heldout,
                })
            })
            .collect()
    }

    fn alarm(class: VerifierClass) -> DivergenceAlarm {
        DivergenceAlarm::new(DivergencePolicy::default(), class)
    }

    #[test]
    fn a_reward_that_climbs_while_the_heldout_measure_stays_flat_raises_the_alarm() {
        let mut a = alarm(VerifierClass::Objective);
        let signals = feed(
            &mut a,
            &[
                (0.30, Some(0.40)),
                (0.38, Some(0.41)),
                (0.46, Some(0.40)),
                (0.55, Some(0.41)),
                (0.64, Some(0.40)),
            ],
        );
        let Signal::Diverging(alert) = signals.last().cloned().unwrap_or(Signal::Steady) else {
            panic!("{signals:?}");
        };
        assert!(alert.reward_rise > 0.3, "{alert:?}");
        assert!(alert.heldout_rise.abs() < 0.05, "{alert:?}");
        assert_eq!(alert.step, 4);
        assert_eq!(alert.label, None, "an objective reward needs no label");
        assert!(
            signals[..4].iter().all(|s| *s == Signal::Steady),
            "no trend before the window fills"
        );
    }

    #[test]
    fn a_heldout_measure_that_falls_while_the_reward_climbs_is_a_divergence_labelled_by_the_verifier(
    ) {
        let mut a = alarm(VerifierClass::Proxy);
        let signals = feed(
            &mut a,
            &[
                (0.2, Some(0.50)),
                (0.3, Some(0.47)),
                (0.4, Some(0.45)),
                (0.5, Some(0.40)),
                (0.6, Some(0.38)),
            ],
        );
        let Signal::Diverging(alert) = signals.last().cloned().unwrap_or(Signal::Steady) else {
            panic!("{signals:?}");
        };
        assert!(alert.heldout_rise < 0.0);
        assert_eq!(alert.label.as_deref(), Some("proxy only"));
    }

    #[test]
    fn a_reward_and_a_heldout_measure_that_rise_together_are_steady() {
        let mut a = alarm(VerifierClass::Objective);
        let signals = feed(
            &mut a,
            &[
                (0.3, Some(0.30)),
                (0.4, Some(0.36)),
                (0.5, Some(0.41)),
                (0.6, Some(0.47)),
                (0.7, Some(0.52)),
            ],
        );
        assert!(signals.iter().all(|s| *s == Signal::Steady), "{signals:?}");
    }

    #[test]
    fn a_flat_reward_is_never_an_alarm_however_the_heldout_measure_moves() {
        let mut a = alarm(VerifierClass::Objective);
        let signals = feed(
            &mut a,
            &[
                (0.5, Some(0.6)),
                (0.5, Some(0.5)),
                (0.51, Some(0.4)),
                (0.5, Some(0.3)),
                (0.5, Some(0.2)),
            ],
        );
        assert!(signals.iter().all(|s| *s == Signal::Steady), "{signals:?}");
    }

    #[test]
    fn a_rising_reward_without_heldout_points_is_unconfirmed_not_steady() {
        let mut a = alarm(VerifierClass::Objective);
        let signals = feed(
            &mut a,
            &[
                (0.2, None),
                (0.3, None),
                (0.4, Some(0.3)),
                (0.5, None),
                (0.6, None),
            ],
        );
        assert_eq!(
            signals.last(),
            Some(&Signal::Unconfirmed {
                reward_rise: signals_rise(&signals),
                heldout_points: 1
            })
        );
    }

    fn signals_rise(signals: &[Signal]) -> f64 {
        match signals.last() {
            Some(Signal::Unconfirmed { reward_rise, .. }) => *reward_rise,
            _ => f64::NAN,
        }
    }

    #[test]
    fn only_the_latest_window_counts() {
        let mut a = DivergenceAlarm::new(
            DivergencePolicy {
                window: 3,
                ..DivergencePolicy::default()
            },
            VerifierClass::Objective,
        );
        // A long divergence, then both recover and rise together.
        feed(
            &mut a,
            &[(0.1, Some(0.5)), (0.3, Some(0.5)), (0.5, Some(0.5))],
        );
        let signals = feed(
            &mut a,
            &[(0.55, Some(0.5)), (0.65, Some(0.55)), (0.75, Some(0.65))],
        );
        assert_eq!(signals.last(), Some(&Signal::Steady), "{signals:?}");
    }
}
