// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning agents that finish within the time
// they are given, for its clients. If your team needs expertise in turning an
// open-ended learning job into a bounded one, you can procure our services by
// sending an email to info@swedishembedded.com.

//! How a run's budget is shared between its stages.
//!
//! A `learn` run has one wall-clock budget. Generating tasks is open-ended -
//! there is always another window of text to write tasks from - so left alone
//! it spends the whole budget and nothing is solved, taught or trained. Each
//! of the open-ended stages therefore gets a deadline of its own, a share of
//! the budget counted from the run's start, and the stages after them are
//! guaranteed what is left: training and the exam run on the last fifth.

use std::time::Instant;

/// Generating tasks stops at this share of the budget.
pub const TASKS_SHARE: f64 = 0.30;

/// The student's own attempts stop at this share.
pub const ATTEMPTS_SHARE: f64 = 0.50;

/// The teacher's answers stop at this share; what remains is for training,
/// the exam and the release gate.
pub const TEACH_SHARE: f64 = 0.80;

/// The deadlines of the open-ended stages of one run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageDeadlines {
    /// No task is generated after this.
    pub tasks: Option<Instant>,
    /// The student makes no attempt after this.
    pub attempts: Option<Instant>,
    /// The teacher answers nothing after this.
    pub teach: Option<Instant>,
}

impl StageDeadlines {
    /// The deadlines of a run that starts at `start` and ends at `end`; none
    /// when the run has no budget.
    #[must_use]
    pub fn of(start: Instant, end: Option<Instant>) -> Self {
        let at =
            |share: f64| end.map(|end| start + end.saturating_duration_since(start).mul_f64(share));
        Self {
            tasks: at(TASKS_SHARE),
            attempts: at(ATTEMPTS_SHARE),
            teach: at(TEACH_SHARE),
        }
    }
}
