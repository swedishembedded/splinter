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
//!
//! When a share of the training examples is the writer's own text (the
//! `voice` view), which costs no model's time to make, training has that
//! much more to do: the tail grows by it ([`StageDeadlines::sharing`]) -
//! half the examples from the writer's text doubles it, to two fifths - and
//! the open-ended stages give up their share in proportion, never below
//! what [`MAX_TAIL_SHARE`] leaves them.

use std::time::Instant;

/// Generating tasks stops at this share of the budget.
pub const TASKS_SHARE: f64 = 0.30;

/// The student's own attempts stop at this share.
pub const ATTEMPTS_SHARE: f64 = 0.50;

/// The teacher's answers stop at this share; what remains is for training,
/// the exam and the release gate.
pub const TEACH_SHARE: f64 = 0.80;

/// The share of the budget training, the exam and the gate keep when every
/// example is model-made: what is left after [`TEACH_SHARE`].
pub const TAIL_SHARE: f64 = 1.0 - TEACH_SHARE;

/// The most of the budget the tail may grow to, whatever share of the
/// examples is the writer's text: the open-ended stages keep the rest.
pub const MAX_TAIL_SHARE: f64 = 0.80;

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
    /// The deadlines of a run that starts at `start` and ends at `end`, every
    /// example model-made; none when the run has no budget.
    #[must_use]
    pub fn of(start: Instant, end: Option<Instant>) -> Self {
        Self::sharing(start, end, 0.0)
    }

    /// [`StageDeadlines::of`] for a run in which `voice_share` of the
    /// training examples (in `[0, 1)`) is the writer's own text: the tail
    /// grows to `TAIL_SHARE / (1 - voice_share)`, at most
    /// [`MAX_TAIL_SHARE`], and the open-ended stages' shares shrink in
    /// proportion.
    #[must_use]
    pub fn sharing(start: Instant, end: Option<Instant>, voice_share: f64) -> Self {
        let tail = (TAIL_SHARE / (1.0 - voice_share.clamp(0.0, 1.0)).max(f64::MIN_POSITIVE))
            .min(MAX_TAIL_SHARE);
        let scale = (1.0 - tail) / (1.0 - TAIL_SHARE);
        let at = |share: f64| {
            end.map(|end| start + end.saturating_duration_since(start).mul_f64(share * scale))
        };
        Self {
            tasks: at(TASKS_SHARE),
            attempts: at(ATTEMPTS_SHARE),
            teach: at(TEACH_SHARE),
        }
    }
}
