// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements curricula that spend a learner's solver and
// training budget where it still learns, for its clients. If your team needs
// expertise in curriculum design or agent evaluation, you can procure our
// services by sending an email to info@swedishembedded.com.

//! pass@k: how often a policy solves one task in k attempts, and whether
//! that makes the task worth training on. Pure: no model, no files.
//!
//! Each attempt's outcome is `Some(true)` (graded right), `Some(false)`
//! (graded wrong) or `None` (no verdict: the verifiers abstained or
//! disagreed). The pass rate is over the graded attempts only; a task with
//! no graded attempt has no rate, never `0`.
//!
//! A task is worth training on when the closed-book student fails it at
//! least sometimes and a verified answer to it exists: a passing student
//! attempt, or a teacher's - the task solved open-book, with its grounding
//! material shown - that its verifiers passed. The student's rate, and
//! whether a teacher's answer was verified, place the task:
//!
//! | Student's rate | Teacher's answer | Class | Why |
//! |---|---|---|---|
//! | 1 | - | [`FrontierClass::Always`] | the student already solves it: no signal to learn from |
//! | strictly between | - | [`FrontierClass::Frontier`] | sometimes right: its passing attempts are the answer |
//! | 0 | verified | [`FrontierClass::Taught`] | never right closed-book: the teacher's answer is new knowledge |
//! | 0 | none verified | [`FrontierClass::Never`] | no verified answer: nothing to learn from |
//! | none | - | [`FrontierClass::Unmeasured`] | no attempt was graded |
//!
//! [`FrontierClass::Frontier`] and [`FrontierClass::Taught`] are kept
//! ([`FrontierClass::kept`]); the rest are dropped.

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
    /// No graded attempt passed, and a teacher's answer was verified.
    Taught,
    /// No attempt was graded.
    Unmeasured,
}

impl FrontierClass {
    /// Whether a task of this class is worth training on: the student
    /// fails it at least sometimes and a verified answer exists.
    #[must_use]
    pub fn kept(self) -> bool {
        matches!(self, Self::Frontier | Self::Taught)
    }
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

    /// Where this, the student's count, places the task when a teacher's
    /// answer to it was verified (`taught`) or not; see the module
    /// documentation.
    #[must_use]
    pub fn class(&self, taught: bool) -> FrontierClass {
        if self.graded == 0 {
            FrontierClass::Unmeasured
        } else if self.passes == self.graded {
            FrontierClass::Always
        } else if self.passes > 0 {
            FrontierClass::Frontier
        } else if taught {
            FrontierClass::Taught
        } else {
            FrontierClass::Never
        }
    }
}

/// How many tasks fell in each class.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Distribution {
    /// Tasks every graded attempt solved: dropped, no signal.
    pub always: usize,
    /// Tasks no graded attempt solved and no teacher answered verifiably:
    /// dropped, nothing to learn from.
    pub never: usize,
    /// Tasks sometimes solved: kept.
    pub frontier: usize,
    /// Tasks never solved closed-book that a teacher answered verifiably:
    /// kept, new knowledge.
    pub taught: usize,
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
            FrontierClass::Taught => self.taught += 1,
            FrontierClass::Unmeasured => self.unmeasured += 1,
        }
    }

    /// Tasks counted.
    #[must_use]
    pub fn tasks(&self) -> usize {
        self.always + self.never + self.frontier + self.taught + self.unmeasured
    }

    /// Tasks kept: on the frontier, or taught.
    #[must_use]
    pub fn kept(&self) -> usize {
        self.frontier + self.taught
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
    fn the_rate_and_a_verified_teacher_answer_place_a_task_and_unmeasured_is_not_zero() {
        let always = count(&[Some(true), Some(true), None]);
        assert_eq!(
            (always.rate(), always.class(false), always.class(true)),
            (Some(1.0), FrontierClass::Always, FrontierClass::Always)
        );
        let never = count(&[Some(false), Some(false)]);
        assert_eq!(
            (never.rate(), never.class(false)),
            (Some(0.0), FrontierClass::Never)
        );
        assert_eq!(
            never.class(true),
            FrontierClass::Taught,
            "never solved closed-book, but a verified answer exists"
        );
        let frontier = count(&[Some(true), Some(false), Some(false), None]);
        assert_eq!(
            (frontier.attempts, frontier.graded, frontier.passes),
            (4, 3, 1)
        );
        assert_eq!(frontier.class(false), FrontierClass::Frontier);
        assert_eq!(frontier.class(true), FrontierClass::Frontier);
        let unmeasured = count(&[None, None]);
        assert_eq!(
            (unmeasured.rate(), unmeasured.class(true)),
            (None, FrontierClass::Unmeasured)
        );
        let kept: Vec<bool> = [
            FrontierClass::Always,
            FrontierClass::Never,
            FrontierClass::Frontier,
            FrontierClass::Taught,
            FrontierClass::Unmeasured,
        ]
        .into_iter()
        .map(FrontierClass::kept)
        .collect();
        assert_eq!(kept, [false, false, true, true, false]);
        let mut distribution = Distribution::default();
        for class in [
            always.class(false),
            never.class(false),
            never.class(true),
            frontier.class(false),
            unmeasured.class(false),
            frontier.class(false),
        ] {
            distribution.add(class);
        }
        assert_eq!(
            (
                distribution.always,
                distribution.never,
                distribution.frontier,
                distribution.taught
            ),
            (1, 1, 2, 1)
        );
        assert_eq!(
            (
                distribution.unmeasured,
                distribution.tasks(),
                distribution.kept()
            ),
            (1, 6, 3)
        );
    }
}
