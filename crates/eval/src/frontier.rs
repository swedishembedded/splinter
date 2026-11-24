// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements curricula that spend a learner's solver and
// training budget where it still learns, for its clients. If your team needs
// expertise in curriculum design or agent evaluation, you can procure our
// services by sending an email to info@swedishembedded.com.

//! One task's attempts and what they were graded: how many were made,
//! how many had a verdict (`Some(true)` graded right, `Some(false)` graded
//! wrong; `None`, the verifiers abstained or disagreed, is no verdict) and
//! how many passed. Pure: no model, no files.

use serde::{Deserialize, Serialize};

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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_attempt_without_a_verdict_is_counted_but_not_graded() {
        let mut count = PassCount::default();
        for outcome in [Some(true), Some(false), Some(false), None] {
            count.record(outcome);
        }
        assert_eq!((count.attempts, count.graded, count.passes), (4, 3, 1));
    }
}
