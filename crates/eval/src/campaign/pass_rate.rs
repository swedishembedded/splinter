// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for spending a reinforcement-learning
// budget only on prompts that teach, for its clients. If your team needs
// expertise in curricula for policy-gradient training you can procure our
// services by sending an email to info@swedishembedded.com.

//! Per-task pass rate under the current policy.
//!
//! A task the policy always solves gives a group with no spread, so no
//! gradient; one it never solves gives none either. The campaign therefore
//! measures every task before it trains (an offline calibration of
//! [`CALIBRATION_K`] to 16 attempts under the current policy) and keeps the
//! rate up to date from every group it later collects. Measurements age: when
//! the policy moves on, what was counted under an older version weighs
//! [`STALE_WEIGHT`] of what it did, so a rate follows the policy without
//! forgetting it all at each refit.
//!
//! A task whose rate reaches [`RETIRE_AT`] on at least `min_graded` graded
//! attempts is retired and never scheduled again: the policy has learned it.
//! Retirement is permanent by design; a retired task that later regresses is
//! the retention probe's to catch, not a reason to spend rollouts on it.
//! A task with no graded attempt has no rate, never `0`.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Attempts per task of the offline calibration, at least: enough for a task
/// solved half the time to show both outcomes in all but one group in 128, and
/// for a rate of 0.9 to mean eight of eight.
pub const CALIBRATION_K: usize = 8;

/// The pass rate at which a task is retired for good.
pub const RETIRE_AT: f64 = 0.9;

/// How much of what was counted under an older policy version still counts
/// once a newer version is measured.
pub const STALE_WEIGHT: f64 = 0.5;

/// Where a task's rate places it for scheduling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Band {
    /// No graded attempt yet: it must be calibrated before it is trained on.
    Unmeasured,
    /// Retired: solved reliably, never scheduled again.
    Retired,
    /// Never solved at the calibration depth: no group can show a spread
    /// until the policy improves, so it is revisited rarely.
    Cliff,
    /// Solved some of the time: the tasks a group can teach from.
    Frontier,
}

/// What the campaign knows of one task's pass rate.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TaskRate {
    /// Weighted passes among the graded attempts.
    passes: f64,
    /// Weighted graded attempts.
    graded: f64,
    /// The policy version of the newest attempts counted.
    policy_version: u64,
    /// Groups collected for it so far.
    visits: u64,
    /// Whether it was retired.
    retired: bool,
}

/// A refused configuration.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PassRateError {
    /// The retirement rate is outside `(0, 1]`.
    #[error("retire_at is {0}; it must lie in (0, 1]")]
    RetireAt(String),
    /// A task is retired on at least one graded attempt.
    #[error("min_graded is 0; a task cannot be retired on no evidence")]
    MinGraded,
}

/// The pass rate of every task a campaign has measured.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PassRates {
    tasks: BTreeMap<String, TaskRate>,
    retire_at: f64,
    min_graded: usize,
}

impl Default for PassRates {
    fn default() -> Self {
        Self {
            tasks: BTreeMap::new(),
            retire_at: RETIRE_AT,
            min_graded: CALIBRATION_K,
        }
    }
}

/// How many tasks fall in each band.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BandCounts {
    /// Tasks without a graded attempt.
    pub unmeasured: usize,
    /// Tasks retired.
    pub retired: usize,
    /// Tasks never solved.
    pub cliff: usize,
    /// Tasks solved some of the time.
    pub frontier: usize,
}

impl PassRates {
    /// Rates that retire a task at `retire_at` once `min_graded` attempts at
    /// it were graded.
    ///
    /// # Errors
    /// `retire_at` outside `(0, 1]`, or `min_graded` of zero.
    pub fn new(retire_at: f64, min_graded: usize) -> Result<Self, PassRateError> {
        if !(retire_at.is_finite() && retire_at > 0.0 && retire_at <= 1.0) {
            return Err(PassRateError::RetireAt(retire_at.to_string()));
        }
        if min_graded == 0 {
            return Err(PassRateError::MinGraded);
        }
        Ok(Self {
            tasks: BTreeMap::new(),
            retire_at,
            min_graded,
        })
    }

    /// Counts a group of attempts at `task` made by `policy_version`: each
    /// outcome is `Some(passed)`, or `None` when no verifier decided it.
    /// Retires the task when its rate reaches the retirement rate on enough
    /// graded attempts.
    pub fn observe(&mut self, task: &str, policy_version: u64, outcomes: &[Option<bool>]) {
        let state = self.tasks.entry(task.to_string()).or_default();
        if policy_version > state.policy_version && state.graded > 0.0 {
            state.passes *= STALE_WEIGHT;
            state.graded *= STALE_WEIGHT;
        }
        state.policy_version = state.policy_version.max(policy_version);
        state.visits += 1;
        for outcome in outcomes.iter().flatten() {
            state.graded += 1.0;
            state.passes += f64::from(u8::from(*outcome));
        }
        if state.graded >= self.min_graded as f64 && state.passes / state.graded >= self.retire_at {
            state.retired = true;
        }
    }

    /// The task's pass rate; `None` before any graded attempt.
    #[must_use]
    pub fn rate(&self, task: &str) -> Option<f64> {
        let state = self.tasks.get(task)?;
        (state.graded > 0.0).then(|| state.passes / state.graded)
    }

    /// Groups collected for the task so far.
    #[must_use]
    pub fn visits(&self, task: &str) -> u64 {
        self.tasks.get(task).map_or(0, |state| state.visits)
    }

    /// Whether the task was retired.
    #[must_use]
    pub fn is_retired(&self, task: &str) -> bool {
        self.tasks.get(task).is_some_and(|state| state.retired)
    }

    /// Where the task's rate places it.
    #[must_use]
    pub fn band(&self, task: &str) -> Band {
        let Some(state) = self.tasks.get(task) else {
            return Band::Unmeasured;
        };
        if state.retired {
            Band::Retired
        } else if state.graded <= 0.0 {
            Band::Unmeasured
        } else if state.passes <= 0.0 && state.graded >= self.min_graded as f64 {
            Band::Cliff
        } else {
            Band::Frontier
        }
    }

    /// How the tasks of `pool` fall in the bands.
    #[must_use]
    pub fn counts<'a>(&self, pool: impl IntoIterator<Item = &'a String>) -> BandCounts {
        let mut counts = BandCounts::default();
        for task in pool {
            match self.band(task) {
                Band::Unmeasured => counts.unmeasured += 1,
                Band::Retired => counts.retired += 1,
                Band::Cliff => counts.cliff += 1,
                Band::Frontier => counts.frontier += 1,
            }
        }
        counts
    }

    /// The tasks of `pool` that are not retired and have no graded attempt:
    /// what the calibration measures next.
    #[must_use]
    pub fn uncalibrated<'a>(&self, pool: &'a [String]) -> Vec<&'a String> {
        pool.iter()
            .filter(|task| self.band(task) == Band::Unmeasured)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outcomes(passes: usize, fails: usize) -> Vec<Option<bool>> {
        std::iter::repeat_n(Some(true), passes)
            .chain(std::iter::repeat_n(Some(false), fails))
            .collect()
    }

    #[test]
    fn a_task_nobody_graded_has_no_rate_and_is_unmeasured() {
        let mut rates = PassRates::default();
        assert_eq!((rates.rate("t"), rates.band("t")), (None, Band::Unmeasured));
        rates.observe("t", 0, &[None, None]);
        assert_eq!(
            (rates.rate("t"), rates.band("t")),
            (None, Band::Unmeasured),
            "attempts with no verdict are not zeros"
        );
    }

    #[test]
    fn a_calibration_places_tasks_in_bands() {
        let mut rates = PassRates::default();
        rates.observe("solved", 0, &outcomes(8, 0));
        rates.observe("never", 0, &outcomes(0, 8));
        rates.observe("half", 0, &outcomes(4, 4));
        rates.observe("nearly", 0, &outcomes(7, 1));
        assert_eq!(rates.band("solved"), Band::Retired);
        assert_eq!(rates.band("never"), Band::Cliff);
        assert_eq!(rates.band("half"), Band::Frontier);
        assert_eq!(
            rates.band("nearly"),
            Band::Frontier,
            "7 of 8 is 0.875, below the retirement rate"
        );
        assert_eq!(rates.rate("half"), Some(0.5));
        let pool: Vec<String> = ["solved", "never", "half", "nearly", "fresh"]
            .map(String::from)
            .to_vec();
        assert_eq!(
            rates.counts(&pool),
            BandCounts {
                unmeasured: 1,
                retired: 1,
                cliff: 1,
                frontier: 2
            }
        );
        assert_eq!(rates.uncalibrated(&pool), vec![&"fresh".to_string()]);
    }

    #[test]
    fn too_few_attempts_neither_retire_nor_condemn_a_task() {
        let mut rates = PassRates::default();
        rates.observe("lucky", 0, &outcomes(3, 0));
        rates.observe("unlucky", 0, &outcomes(0, 3));
        assert_eq!(rates.band("lucky"), Band::Frontier);
        assert_eq!(rates.band("unlucky"), Band::Frontier);
    }

    #[test]
    fn retirement_is_permanent() {
        let mut rates = PassRates::default();
        rates.observe("t", 0, &outcomes(8, 0));
        assert!(rates.is_retired("t"));
        rates.observe("t", 1, &outcomes(0, 16));
        assert!(
            rates.is_retired("t"),
            "a retired task is never scheduled again"
        );
        assert_eq!(rates.band("t"), Band::Retired);
    }

    #[test]
    fn a_newer_policy_version_outweighs_older_counts() {
        let mut rates = PassRates::default();
        rates.observe("t", 0, &outcomes(0, 8));
        rates.observe("t", 1, &outcomes(4, 4));
        // 0/8 aged to 0/4, then 4/8: 4 of 12.
        let rate = rates.rate("t").unwrap();
        assert!((rate - 4.0 / 12.0).abs() < 1e-12, "{rate}");
        assert_eq!(rates.visits("t"), 2);
        rates.observe("t", 1, &outcomes(2, 0));
        let same_version = rates.rate("t").unwrap();
        assert!(
            (same_version - 6.0 / 14.0).abs() < 1e-12,
            "attempts under the same version are not aged: {same_version}"
        );
    }

    #[test]
    fn configuration_is_validated() {
        assert!(PassRates::new(0.0, 8).is_err());
        assert!(PassRates::new(1.5, 8).is_err());
        assert!(PassRates::new(f64::NAN, 8).is_err());
        assert!(PassRates::new(0.9, 0).is_err());
        let mut strict = PassRates::new(1.0, 4).unwrap();
        strict.observe("t", 0, &outcomes(3, 1));
        assert_eq!(strict.band("t"), Band::Frontier);
        strict.observe("t", 0, &outcomes(4, 0));
        assert_eq!(strict.band("t"), Band::Frontier, "7 of 8 is below 1.0");
        strict.observe("u", 0, &outcomes(4, 0));
        assert_eq!(strict.band("u"), Band::Retired);
    }

    #[test]
    fn rates_survive_a_round_trip_through_json() {
        let mut rates = PassRates::default();
        rates.observe("a", 3, &outcomes(5, 3));
        rates.observe("b", 3, &outcomes(8, 0));
        let text = serde_json::to_string(&rates).unwrap();
        assert_eq!(serde_json::from_str::<PassRates>(&text).unwrap(), rates);
    }
}
