// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements small-sample statistics for model
// evaluations that are honest about how little they rest on, for its
// clients. If your team needs expertise in reporting results from a few
// families or a single training run, you can procure our services by
// sending an email to info@swedishembedded.com.

//! What a result on few independent units or a single training run may be
//! called, and the interval that fits a share counted over few units.
//!
//! * A result is a **pilot** when it rests on fewer than [`PILOT_UNITS`]
//!   independent units (families) or fewer than [`MIN_TRAINING_RUNS`]
//!   training runs; it is labelled so, with the reason, wherever it is
//!   reported.
//! * A share counted over units (the candidate's wins among the discordant
//!   families) gets a Wilson score interval, which keeps its ends inside
//!   `[0, 1]` and does not collapse to a point at 0 or all, where a
//!   percentile bootstrap over a handful of units does.

use serde::{Deserialize, Serialize};

/// Fewer independent units than this make a result a pilot.
pub const PILOT_UNITS: usize = 100;

/// Fewer training runs than this make a result a pilot.
pub const MIN_TRAINING_RUNS: usize = 3;

/// The two-sided 95% normal quantile the intervals are made at.
const Z_95: f64 = 1.959_963_984_540_054;

/// A share counted over independent units, with its Wilson interval.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Share {
    /// Units counted in favour.
    pub successes: usize,
    /// Units counted.
    pub total: usize,
    /// The share, `successes / total`.
    pub value: f64,
    /// The lower end of the 95% Wilson interval.
    pub low: f64,
    /// The upper end of the 95% Wilson interval.
    pub high: f64,
}

impl Share {
    /// `successes` of `total` with its 95% Wilson interval; `None` when
    /// nothing was counted (no share is not a share of zero) or `successes`
    /// exceeds `total`.
    #[must_use]
    pub fn wilson(successes: usize, total: usize) -> Option<Self> {
        if total == 0 || successes > total {
            return None;
        }
        let n = total as f64;
        let p = successes as f64 / n;
        let z2 = Z_95 * Z_95;
        let centre = (successes as f64 + z2 / 2.0) / (n + z2);
        let half = Z_95 * (n * p * (1.0 - p) + z2 / 4.0).sqrt() / (n + z2);
        Some(Self {
            successes,
            total,
            value: p,
            low: (centre - half).max(0.0),
            high: (centre + half).min(1.0),
        })
    }
}

/// Why a result on `units` independent units and `training_runs` training
/// seeds is a pilot, one sentence per reason; empty when it is not one.
#[must_use]
pub fn pilot_reasons(units: usize, training_runs: usize) -> Vec<String> {
    let mut reasons = Vec::new();
    if units < PILOT_UNITS {
        reasons.push(format!(
            "{units} independent unit(s), fewer than {PILOT_UNITS}"
        ));
    }
    if training_runs < MIN_TRAINING_RUNS {
        reasons.push(format!(
            "{training_runs} training run(s), fewer than {MIN_TRAINING_RUNS}"
        ));
    }
    reasons
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-3
    }

    #[test]
    fn the_wilson_interval_matches_hand_computed_values_and_stays_inside_zero_and_one() {
        // 5 of 5: centre 0.7827, half 0.2172.
        let all = Share::wilson(5, 5).unwrap();
        assert!(close(all.low, 0.5655) && close(all.high, 1.0), "{all:?}");
        // 0 of 10: the upper end is 0.2775.
        let none = Share::wilson(0, 10).unwrap();
        assert!(close(none.low, 0.0) && close(none.high, 0.2775), "{none:?}");
        // 10 of 20 is symmetric about a half.
        let half = Share::wilson(10, 20).unwrap();
        assert!(
            close(half.low, 0.2993) && close(half.high, 0.7007),
            "{half:?}"
        );
        assert!(close(half.value, 0.5));
    }

    #[test]
    fn nothing_counted_is_no_share_and_an_impossible_count_is_refused() {
        assert_eq!(Share::wilson(0, 0), None);
        assert_eq!(Share::wilson(3, 2), None);
    }

    #[test]
    fn a_result_is_a_pilot_below_a_hundred_units_or_three_seeds_and_says_which() {
        assert!(pilot_reasons(100, 3).is_empty());
        assert_eq!(pilot_reasons(99, 3).len(), 1);
        assert!(pilot_reasons(150, 2)[0].contains("training run"));
        assert_eq!(pilot_reasons(8, 1).len(), 2);
    }
}
