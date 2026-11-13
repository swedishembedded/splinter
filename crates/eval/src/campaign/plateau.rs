// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for knowing when a model-improvement
// campaign has stopped paying for its GPU-hours, for its clients. If your team
// needs expertise in compute-efficient experiment programmes you can procure
// our services by sending an email to info@swedishembedded.com.

//! The plateau switch: when discovery per GPU-hour falls, change what the
//! campaign does instead of running the same arm again.
//!
//! A discovery is whatever the campaign counts as new capability found (a
//! task the policy solved for the first time, a dev pass won), credited to
//! the arm that found it, so a rejected arm that cost eight GPU-hours and
//! found nothing counts against the rate. The rate of the most recent
//! `window` arms is compared with the best rate any window of that size
//! reached; below [`PlateauRule::fraction_of_peak`] of it, the campaign has
//! plateaued.

use serde::{Deserialize, Serialize};

/// What one arm cost and found.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArmCost {
    /// The arm.
    pub arm: String,
    /// GPU-hours it used.
    pub gpu_hours: f64,
    /// What it discovered, in the campaign's unit; zero for an arm that
    /// found nothing.
    pub discoveries: f64,
}

/// When a falling rate counts as a plateau.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlateauRule {
    /// Arms in a window; the trend needs two windows of history.
    pub window: usize,
    /// The recent rate, as a share of the peak rate, at or below which the
    /// campaign has plateaued.
    pub fraction_of_peak: f64,
}

impl Default for PlateauRule {
    fn default() -> Self {
        Self {
            window: 3,
            fraction_of_peak: 0.25,
        }
    }
}

/// What the history says.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "trend", rename_all = "snake_case")]
pub enum Trend {
    /// Fewer than two windows of arms, or a window that used no GPU-hours:
    /// no trend can be stated.
    TooEarly,
    /// The recent rate holds above the plateau line.
    Improving {
        /// Discoveries per GPU-hour over the last window.
        recent_rate: f64,
        /// The best window's rate.
        peak_rate: f64,
    },
    /// The recent rate fell to the plateau line or below.
    Plateau {
        /// Discoveries per GPU-hour over the last window.
        recent_rate: f64,
        /// The best window's rate.
        peak_rate: f64,
    },
}

/// Discoveries per GPU-hour over `arms`; `None` when they used none.
#[must_use]
pub fn discovery_rate(arms: &[ArmCost]) -> Option<f64> {
    let hours: f64 = arms.iter().map(|a| a.gpu_hours).sum();
    (hours > 0.0).then(|| arms.iter().map(|a| a.discoveries).sum::<f64>() / hours)
}

/// The trend of `history`, oldest arm first.
#[must_use]
pub fn assess(history: &[ArmCost], rule: &PlateauRule) -> Trend {
    let window = rule.window.max(1);
    if history.len() < 2 * window {
        return Trend::TooEarly;
    }
    let rates: Option<Vec<f64>> = history.windows(window).map(discovery_rate).collect();
    let Some(rates) = rates else {
        return Trend::TooEarly;
    };
    let recent_rate = rates[rates.len() - 1];
    let peak_rate = rates.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if recent_rate <= rule.fraction_of_peak * peak_rate {
        Trend::Plateau {
            recent_rate,
            peak_rate,
        }
    } else {
        Trend::Improving {
            recent_rate,
            peak_rate,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arm(hours: f64, discoveries: f64) -> ArmCost {
        ArmCost {
            arm: "a".into(),
            gpu_hours: hours,
            discoveries,
        }
    }

    #[test]
    fn the_rate_is_discoveries_per_gpu_hour_and_absent_without_hours() {
        assert_eq!(discovery_rate(&[arm(2.0, 3.0), arm(2.0, 1.0)]), Some(1.0));
        assert_eq!(discovery_rate(&[arm(0.0, 5.0)]), None);
        assert_eq!(discovery_rate(&[]), None);
    }

    #[test]
    fn too_little_history_states_no_trend() {
        let rule = PlateauRule::default();
        assert_eq!(assess(&vec![arm(1.0, 1.0); 5], &rule), Trend::TooEarly);
        let idle = vec![arm(0.0, 0.0); 6];
        assert_eq!(assess(&idle, &rule), Trend::TooEarly);
    }

    #[test]
    fn a_steady_rate_is_improving_and_a_collapsed_one_is_a_plateau() {
        let rule = PlateauRule::default();
        let steady = vec![arm(1.0, 2.0); 6];
        assert!(matches!(assess(&steady, &rule), Trend::Improving { .. }));

        let mut stalled = vec![arm(1.0, 4.0); 3];
        stalled.extend(vec![arm(1.0, 0.0); 3]);
        match assess(&stalled, &rule) {
            Trend::Plateau {
                recent_rate,
                peak_rate,
            } => {
                assert_eq!(recent_rate, 0.0);
                assert_eq!(peak_rate, 4.0);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_campaign_that_never_found_anything_has_plateaued() {
        let none = vec![arm(2.0, 0.0); 6];
        assert!(matches!(
            assess(&none, &PlateauRule::default()),
            Trend::Plateau { .. }
        ));
    }

    #[test]
    fn a_slower_but_not_collapsed_rate_is_still_improving() {
        let mut history = vec![arm(1.0, 4.0); 3];
        history.extend(vec![arm(1.0, 2.0); 3]);
        assert!(matches!(
            assess(&history, &PlateauRule::default()),
            Trend::Improving { .. }
        ));
    }
}
