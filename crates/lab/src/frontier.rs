// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements curricula that spend a learner's solver and
// training budget where it still learns, for its clients. If your team needs
// expertise in curriculum design or agent evaluation, you can procure our
// services by sending an email to info@swedishembedded.com.

//! pass@k: how often a policy solves one task in k attempts, and where that
//! puts the task. Pure: no model, no files.
//!
//! Each attempt's outcome is `Some(true)` (graded right), `Some(false)`
//! (graded wrong) or `None` (no verdict: the verifiers abstained or
//! disagreed). The pass rate is over the graded attempts only; a task with
//! no graded attempt has no rate, never `0`. The rate places the task:
//!
//! | Rate | Class | Why |
//! |---|---|---|
//! | 1 | [`FrontierClass::Always`] | the policy already solves it: no signal to learn from |
//! | 0 | [`FrontierClass::Never`] | nothing it produced passes: no usable data |
//! | strictly between | [`FrontierClass::Frontier`] | sometimes right: where learning happens |
//! | none | [`FrontierClass::Unmeasured`] | no attempt was graded |

use serde::{Deserialize, Serialize};

/// Where a task's pass rate places it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FrontierClass {
    /// Every graded attempt passed.
    Always,
    /// No graded attempt passed.
    Never,
    /// Some graded attempts passed and some failed.
    Frontier,
    /// No attempt was graded.
    Unmeasured,
}

/// One task's attempts and what they were graded.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PassCount {
    /// Attempts made.
    pub attempts: usize,
    /// Of those, attempts with a verdict.
    pub graded: usize,
    /// Of those, attempts graded right.
    pub passes: usize,
}

impl PassCount {
    /// Counts one more attempt, graded `outcome`.
    pub fn record(&mut self, outcome: Option<bool>) {
        self.attempts += 1;
        if let Some(passed) = outcome {
            self.graded += 1;
            self.passes += usize::from(passed);
        }
    }

    /// The pass rate over the graded attempts; `None` with none.
    #[must_use]
    pub fn rate(&self) -> Option<f64> {
        (self.graded > 0).then(|| self.passes as f64 / self.graded as f64)
    }

    /// Where the rate places the task; see the module documentation.
    #[must_use]
    pub fn class(&self) -> FrontierClass {
        if self.graded == 0 {
            FrontierClass::Unmeasured
        } else if self.passes == self.graded {
            FrontierClass::Always
        } else if self.passes == 0 {
            FrontierClass::Never
        } else {
            FrontierClass::Frontier
        }
    }
}

/// How many tasks fell in each class.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Distribution {
    /// Tasks every graded attempt solved: dropped, no signal.
    pub always: usize,
    /// Tasks no graded attempt solved: dropped, no usable data.
    pub never: usize,
    /// Tasks sometimes solved: kept.
    pub frontier: usize,
    /// Tasks with no graded attempt: dropped, nothing was measured.
    pub unmeasured: usize,
}

impl Distribution {
    /// Counts one task of `class`.
    pub fn add(&mut self, class: FrontierClass) {
        match class {
            FrontierClass::Always => self.always += 1,
            FrontierClass::Never => self.never += 1,
            FrontierClass::Frontier => self.frontier += 1,
            FrontierClass::Unmeasured => self.unmeasured += 1,
        }
    }

    /// Tasks counted.
    #[must_use]
    pub fn tasks(&self) -> usize {
        self.always + self.never + self.frontier + self.unmeasured
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn count(outcomes: &[Option<bool>]) -> PassCount {
        let mut count = PassCount::default();
        for outcome in outcomes {
            count.record(*outcome);
        }
        count
    }

    #[test]
    fn the_rate_places_a_task_and_unmeasured_is_not_zero() {
        let always = count(&[Some(true), Some(true), None]);
        assert_eq!(
            (always.rate(), always.class()),
            (Some(1.0), FrontierClass::Always)
        );
        let never = count(&[Some(false), Some(false)]);
        assert_eq!(
            (never.rate(), never.class()),
            (Some(0.0), FrontierClass::Never)
        );
        let frontier = count(&[Some(true), Some(false), Some(false), None]);
        assert_eq!(
            (frontier.attempts, frontier.graded, frontier.passes),
            (4, 3, 1)
        );
        assert_eq!(frontier.class(), FrontierClass::Frontier);
        let unmeasured = count(&[None, None]);
        assert_eq!(
            (unmeasured.rate(), unmeasured.class()),
            (None, FrontierClass::Unmeasured)
        );
        let mut distribution = Distribution::default();
        for c in [&always, &never, &frontier, &unmeasured, &frontier] {
            distribution.add(c.class());
        }
        assert_eq!(
            (
                distribution.always,
                distribution.never,
                distribution.frontier
            ),
            (1, 1, 2)
        );
        assert_eq!((distribution.unmeasured, distribution.tasks()), (1, 5));
    }
}
