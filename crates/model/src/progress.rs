// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for measuring what a training run
// costs on a card for its clients. If your team needs expertise in training
// throughput on GPUs then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The time a fine-tune's optimizer steps take, apart from the model load and
//! the scoring around them.

use std::cell::Cell;
use std::time::Instant;

use splinter_core::training::TrainingSpeed;

/// When a fine-tune started and when its steps were reported, to time the
/// steps apart from the model load and scoring around them.
pub(crate) struct StepClock {
    start: Instant,
    /// The step and instant of the first report, and of the latest one.
    first: Cell<Option<(u32, Instant)>>,
    last: Cell<Option<(u32, Instant)>>,
}

impl StepClock {
    /// A clock that starts now.
    pub(crate) fn start() -> Self {
        Self::started(Instant::now())
    }

    /// Notes that step `step` was just reported.
    pub(crate) fn tick(&self, step: u32) {
        self.reported(step, Instant::now());
    }

    /// The speed, with the fine-tune ending now.
    pub(crate) fn finish(&self, rows_per_step: u32, block: u32) -> Option<TrainingSpeed> {
        self.speed(rows_per_step, block, Instant::now())
    }

    fn started(start: Instant) -> Self {
        Self {
            start,
            first: Cell::new(None),
            last: Cell::new(None),
        }
    }

    fn reported(&self, step: u32, at: Instant) {
        if self.first.get().is_none() {
            self.first.set(Some((step, at)));
        }
        self.last.set(Some((step, at)));
    }

    /// The speed of the steps after the first report, `rows_per_step` rows of
    /// `block` tokens each, if the run reported two steps or more; `end` is
    /// when the fine-tune returned.
    fn speed(&self, rows_per_step: u32, block: u32, end: Instant) -> Option<TrainingSpeed> {
        let ((first_step, first_at), (last_step, last_at)) = (self.first.get()?, self.last.get()?);
        TrainingSpeed::measured(
            last_step.checked_sub(first_step)?,
            rows_per_step,
            block,
            last_at.duration_since(first_at).as_secs_f64(),
            first_at.duration_since(self.start).as_secs_f64(),
            end.duration_since(last_at).as_secs_f64(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The steps are timed from the first report to the last, apart from the
    /// lead before the first and the tail after the last.
    #[test]
    fn the_speed_times_the_steps_between_the_first_and_last_report() {
        let t0 = Instant::now();
        let at = |s: u64| t0 + std::time::Duration::from_secs(s);
        let clock = StepClock::started(t0);
        assert!(
            clock.speed(4, 1000, at(60)).is_none(),
            "no report, no speed"
        );
        clock.reported(1, at(30));
        assert!(
            clock.speed(4, 1000, at(60)).is_none(),
            "one report has no interval"
        );
        clock.reported(5, at(40));
        clock.reported(10, at(50));
        let speed = clock
            .speed(4, 1000, at(57))
            .expect("two reports time an interval");
        assert_eq!(
            speed.steps_timed, 9,
            "the steps after the first reported one"
        );
        assert_eq!(speed.row_tokens, 36_000);
        assert!((speed.seconds - 20.0).abs() < 1e-9);
        assert!((speed.tokens_per_second - 1800.0).abs() < 1e-9);
        assert!(
            (speed.lead_seconds - 30.0).abs() < 1e-9 && (speed.tail_seconds - 7.0).abs() < 1e-9
        );
    }
}
