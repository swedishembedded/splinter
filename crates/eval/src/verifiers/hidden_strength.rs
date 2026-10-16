// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements executable evaluation of code changes
// against a project's own test suite, for its clients. If your team needs
// expertise in telling whether the hidden tests of a benchmark task would
// notice a wrong patch, you can procure our services by sending an email to
// info@swedishembedded.com.

//! How strong a workspace task's hidden tests are: of the other faults
//! seeded near the one the task is about, how many do the hidden tests
//! catch?
//!
//! A hidden suite that fails on the seeded fault and on nothing else lets a
//! plausible wrong patch through as a pass. The measurement is recorded on
//! the task's check, so that admission can refuse a weak suite and a release
//! gate can weigh a pass by the strength of the tests behind it.

use serde::{Deserialize, Serialize};

/// What the hidden tests did on the variants of a fault.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HiddenStrength {
    /// Variants of the fault that some test of the whole suite catches: the
    /// ones that are demonstrably wrong. A variant no test catches may be
    /// equivalent to the original and is not counted.
    pub variants: usize,
    /// Of those, the ones the hidden tests alone catch.
    pub caught: usize,
}

impl HiddenStrength {
    /// The share of variants the hidden tests catch; `None` when no variant
    /// was demonstrably wrong, which is no measurement and never `0`.
    #[must_use]
    pub fn kill_rate(&self) -> Option<f64> {
        (self.variants > 0).then(|| self.caught as f64 / self.variants as f64)
    }
}

/// What the hidden suites of a set of workspace tasks are worth, as a gate
/// report states it beside the pass rates they produced: a pass on a task
/// whose suite catches few of the plausible wrong patches is weaker evidence
/// than one on a task whose suite catches most. The gate reports this and
/// does not weigh passes by it: the floor below which a suite is too weak
/// has not been calibrated against wrong patches that were really tried, and
/// a weight chosen without that would move the verdict by an arbitrary
/// amount.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct StrengthSummary {
    /// Workspace tasks in the suite.
    pub tasks: usize,
    /// Of those, the ones whose hidden suite has a measured kill rate;
    /// the rest were made without the measurement or had no demonstrably
    /// wrong variant.
    pub measured: usize,
    /// The mean kill rate over the measured tasks; `None` when none was
    /// measured, never `0`.
    pub mean_kill_rate: Option<f64>,
    /// The lowest kill rate among the measured tasks.
    pub min_kill_rate: Option<f64>,
}

/// The summary of `strengths`, one entry per workspace task (`None` for a
/// task made without the measurement); `None` when there is no task.
#[must_use]
pub fn summarise(strengths: &[Option<HiddenStrength>]) -> Option<StrengthSummary> {
    if strengths.is_empty() {
        return None;
    }
    let rates: Vec<f64> = strengths
        .iter()
        .filter_map(|s| s.as_ref().and_then(HiddenStrength::kill_rate))
        .collect();
    Some(StrengthSummary {
        tasks: strengths.len(),
        measured: rates.len(),
        mean_kill_rate: (!rates.is_empty()).then(|| rates.iter().sum::<f64>() / rates.len() as f64),
        min_kill_rate: rates.iter().copied().reduce(f64::min),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strength(variants: usize, caught: usize) -> Option<HiddenStrength> {
        Some(HiddenStrength { variants, caught })
    }

    #[test]
    fn the_summary_counts_unmeasured_tasks_and_never_reads_them_as_zero() {
        assert_eq!(summarise(&[]), None);
        let none = summarise(&[None, strength(0, 0)]).unwrap();
        assert_eq!((none.tasks, none.measured), (2, 0));
        assert_eq!((none.mean_kill_rate, none.min_kill_rate), (None, None));
        let mixed = summarise(&[strength(4, 4), strength(2, 1), None]).unwrap();
        assert_eq!((mixed.tasks, mixed.measured), (3, 2));
        assert_eq!(mixed.mean_kill_rate, Some(0.75));
        assert_eq!(mixed.min_kill_rate, Some(0.5));
    }

    #[test]
    fn no_demonstrably_wrong_variant_is_unmeasured_not_zero() {
        assert_eq!(
            HiddenStrength {
                variants: 0,
                caught: 0
            }
            .kill_rate(),
            None
        );
        assert_eq!(
            HiddenStrength {
                variants: 4,
                caught: 3
            }
            .kill_rate(),
            Some(0.75)
        );
        assert_eq!(
            HiddenStrength {
                variants: 4,
                caught: 0
            }
            .kill_rate(),
            Some(0.0)
        );
    }
}
