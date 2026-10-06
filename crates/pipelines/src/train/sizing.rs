// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training schedules that stop a fine-tune
// where its held-out loss turns, for its clients. If your team needs
// expertise in sizing and monitoring small-data fine-tunes, you can procure
// our services by sending an email to info@swedishembedded.com.

//! How long a supervised run trains and how often it looks: the step budget
//! as a ceiling of passes over the data, what a step averages, the cadence
//! of the monitoring evaluations and the patience within it, and how much of
//! the training data is set aside to monitor on.
//!
//! The budget is a ceiling, not a target. A run scores its monitoring set as
//! it trains and exports the adapter of the evaluation with the lowest
//! monitoring loss; it stops once that loss has gone [`DEFAULT_PATIENCE`]
//! evaluations without improving, and otherwise runs the budget out. What the
//! budget decides is how far past the optimum the curve may be followed, and
//! what the learning-rate schedule decays over.

use super::DEFAULT_STEPS;

/// The most passes over what a run has learned its step budget allows when
/// its steps are not given: three. Instruction fine-tunes are trained for one
/// to three passes as a rule, a low-rank adapter learns slower than the whole
/// model would, and the best held-out loss of a persona fine-tune on a few
/// hundred answers came inside its first pass - the budget has to reach past
/// the optimum for the curve to show where it was, and no further than the
/// patience needs.
pub const MAX_PASSES: u32 = 3;

/// The most steps a run takes when they are not given.
pub const MAX_AUTO_STEPS: u32 = 2000;

/// The data a step averages one more record for.
pub const RECORDS_PER_STEP_UNIT: usize = 16;

/// The most records a step averages when none is given.
pub const MAX_RECORDS_PER_STEP: u32 = 8;

/// How many evaluations the step budget is divided into when the cadence is
/// not given: sixteen, so a patience of a few evaluations is a quarter of
/// the budget and a curve of sixteen points shows where the loss turned.
pub const EVALUATIONS_PER_BUDGET: u32 = 16;

/// Evaluations without improvement before a run stops when none is given:
/// four, a quarter of the budget at the default cadence. One evaluation is a
/// hair trigger on a monitoring set of a dozen records; waiting much longer
/// buys little and costs the hours the stop exists to save.
pub const DEFAULT_PATIENCE: u32 = 4;

/// The share of the training families set aside as the monitoring set when
/// none is given: a tenth, taken whole families at a time.
pub const DEFAULT_MONITOR_SHARE: f64 = 0.1;

/// The most of the training families a monitoring set may take: half; past
/// it the run would monitor more than it fits.
pub const MAX_MONITOR_SHARE: f64 = 0.5;

/// How many records a step of a run over `records` records averages when
/// none is given: one per [`RECORDS_PER_STEP_UNIT`] records there are, at
/// least one and at most [`MAX_RECORDS_PER_STEP`]. A set this small is made of
/// long, individual answers, and an update on one record at a time is noise
/// that the next record undoes.
#[must_use]
pub fn auto_records_per_step(records: usize) -> u32 {
    u32::try_from(records / RECORDS_PER_STEP_UNIT)
        .unwrap_or(u32::MAX)
        .clamp(1, MAX_RECORDS_PER_STEP)
}

/// The step budget of a run over `records` records when none is given:
/// [`MAX_PASSES`] passes at [`auto_records_per_step`] records a step, never
/// fewer than [`DEFAULT_STEPS`] and never more than [`MAX_AUTO_STEPS`].
#[must_use]
pub fn auto_steps(records: usize) -> u32 {
    steps_for(records, auto_records_per_step(records))
}

/// The step budget of [`MAX_PASSES`] passes over `records` records at
/// `records_per_step` a step, within the same bounds.
#[must_use]
pub fn steps_for(records: usize, records_per_step: u32) -> u32 {
    let reads = u32::try_from(records)
        .unwrap_or(u32::MAX)
        .saturating_mul(MAX_PASSES);
    reads
        .div_ceil(records_per_step.max(1))
        .clamp(DEFAULT_STEPS, MAX_AUTO_STEPS)
}

/// Steps between monitoring evaluations of a run of `steps` when none is
/// given: [`EVALUATIONS_PER_BUDGET`] evaluations over the budget, at least
/// one step apart.
#[must_use]
pub fn eval_every_for(steps: u32) -> u32 {
    steps.div_ceil(EVALUATIONS_PER_BUDGET).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A budget is passes over the data at what a step reads, within its
    /// bounds; the cadence divides it into the evaluations a patience counts.
    #[test]
    fn the_budget_is_passes_over_the_data_and_the_cadence_divides_it() {
        assert_eq!(
            steps_for(214, 8),
            81,
            "three passes over 214 examples, eight to a step"
        );
        assert_eq!(
            steps_for(2, 1),
            DEFAULT_STEPS,
            "never fewer than the default"
        );
        assert_eq!(
            steps_for(1_000_000, 8),
            MAX_AUTO_STEPS,
            "never more than the most"
        );
        assert_eq!(
            eval_every_for(81),
            6,
            "sixteen evaluations over the budget, rounded up"
        );
        assert_eq!(eval_every_for(5), 1, "at least one step apart");
        assert!(
            DEFAULT_PATIENCE * eval_every_for(80) <= 80 / 3,
            "the patience is at most a third of the budget"
        );
    }
}
