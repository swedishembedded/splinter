// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements admission gates that decide whether a
// training environment may be trained on, for its clients. If your team
// needs expertise in calibrating task difficulty to a model so that
// reinforcement learning has signal, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The measured base pass-rate band: an environment a base policy always
//! solves, or never solves, gives a group-relative trainer nothing to learn
//! from, so no environment enters a campaign before its base pass rate was
//! measured and found inside a band.

use serde::{Deserialize, Serialize};

use super::admission::GateResult;

/// The pass rates of a base policy at which an environment is worth
/// training on: always failing gives no group a gradient, always passing
/// teaches nothing.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BandPolicy {
    /// The lowest base pass rate admitted.
    pub low: f64,
    /// The highest base pass rate admitted.
    pub high: f64,
    /// The fewest attempts a measurement must rest on.
    pub min_attempts: u32,
}

impl Default for BandPolicy {
    fn default() -> Self {
        Self {
            low: 0.1,
            high: 0.9,
            min_attempts: 16,
        }
    }
}

/// A base policy's attempts at the environment's dev instances.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BandMeasurement {
    /// Who was measured (a model and its serving settings).
    pub model: String,
    /// Attempts made.
    pub attempts: u32,
    /// Attempts graded solved.
    pub passes: u32,
}

/// A measured pass rate with its two-sided Wilson score interval
/// ([`splinter_eval::pilot::Share::wilson`]).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BandResult {
    /// Who was measured.
    pub model: String,
    /// Attempts made.
    pub attempts: u32,
    /// The point estimate.
    pub rate: f64,
    /// The interval's lower end.
    pub lower: f64,
    /// The interval's upper end.
    pub upper: f64,
}

impl BandMeasurement {
    /// The rate and its Wilson interval; `None` for no attempts or more
    /// passes than attempts.
    #[must_use]
    pub fn result(&self) -> Option<BandResult> {
        let share =
            splinter_eval::pilot::Share::wilson(self.passes as usize, self.attempts as usize)?;
        Some(BandResult {
            model: self.model.clone(),
            attempts: self.attempts,
            rate: share.value,
            lower: share.low,
            upper: share.high,
        })
    }
}

/// Where a measurement leaves an environment.
pub enum BandStatus {
    /// Not measured, or on too few attempts to say.
    Unmeasured,
    /// In the band.
    Inside,
    /// Outside it.
    Outside,
}

/// Holds `measurement` (with its `result`) to `policy`: the gate it earns,
/// when there is one, and where it leaves the environment.
#[must_use]
pub fn judge_band(
    measurement: Option<&BandMeasurement>,
    result: Option<&BandResult>,
    policy: BandPolicy,
) -> (Option<GateResult>, BandStatus) {
    let (Some(measurement), Some(result)) = (measurement, result) else {
        return (None, BandStatus::Unmeasured);
    };
    let gate = |passed: bool, detail: String| GateResult {
        name: "base-band".into(),
        passed,
        detail,
    };
    if measurement.attempts < policy.min_attempts {
        return (
            Some(gate(
                false,
                format!(
                    "only {} attempts; a band needs {}",
                    measurement.attempts, policy.min_attempts
                ),
            )),
            BandStatus::Unmeasured,
        );
    }
    let text = format!(
        "{} solves {:.0}% (Wilson interval {:.0}% to {:.0}%) over {} attempts; band {:.0}% to {:.0}%",
        result.model,
        result.rate * 100.0,
        result.lower * 100.0,
        result.upper * 100.0,
        result.attempts,
        policy.low * 100.0,
        policy.high * 100.0
    );
    if result.rate < policy.low {
        (
            Some(gate(
                false,
                format!("{text}: too hard, lower a difficulty knob"),
            )),
            BandStatus::Outside,
        )
    } else if result.rate > policy.high {
        (
            Some(gate(
                false,
                format!("{text}: too easy, raise a difficulty knob"),
            )),
            BandStatus::Outside,
        )
    } else {
        (Some(gate(true, text)), BandStatus::Inside)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] // a panic is the failure report
mod tests {
    use super::*;

    fn measurement(passes: u32, attempts: u32) -> BandMeasurement {
        BandMeasurement {
            model: "base".into(),
            attempts,
            passes,
        }
    }

    #[test]
    fn the_wilson_interval_brackets_the_rate_and_stays_inside_zero_to_one() {
        let result = measurement(8, 16).result().unwrap();
        assert!((result.rate - 0.5).abs() < 1e-12);
        assert!(result.lower < 0.5 && 0.5 < result.upper);
        let none = measurement(0, 16).result().unwrap();
        assert!(none.lower == 0.0 && none.upper > 0.0 && none.upper < 0.3);
        assert!(measurement(1, 0).result().is_none());
        assert!(measurement(5, 4).result().is_none());
    }

    #[test]
    fn a_band_is_unmeasured_without_enough_attempts_and_outside_when_too_easy_or_hard() {
        let policy = BandPolicy::default();
        let judged = |m: &BandMeasurement| judge_band(Some(m), m.result().as_ref(), policy).1;
        assert!(matches!(judged(&measurement(8, 8)), BandStatus::Unmeasured));
        assert!(matches!(judged(&measurement(0, 32)), BandStatus::Outside));
        assert!(matches!(judged(&measurement(32, 32)), BandStatus::Outside));
        assert!(matches!(judged(&measurement(16, 32)), BandStatus::Inside));
        assert!(matches!(
            judge_band(None, None, policy).1,
            BandStatus::Unmeasured
        ));
    }
}
