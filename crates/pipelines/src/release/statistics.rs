// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gated self-improvement loops where a model
// update is adopted only on held-out evidence. If your team needs expertise
// in evaluation-gated fine-tuning, you can procure our services by sending
// an email to info@swedishembedded.com.

//! What the gate reports beside what it decides on: the tasks' own evidence
//! for an improvement, and how large a drop the anchor suite could show.

use splinter_eval::gate::{self, DropInterval, TaskLevel};
use splinter_eval::paired::PairedOutcome;

use crate::powered::analysis::{interval_of, sign_p};
use splinter_model::stats::bootstrap_interval;

/// Resamples the task-level interval is made from.
const TASK_LEVEL_RESAMPLES: usize = 10_000;

/// What `outcomes` say task by task: the exact one-sided sign test over the
/// tasks only one model got right, and the difference in the share right with
/// an interval that resamples families (a task of no family is its own).
pub(super) fn task_level(outcomes: &[PairedOutcome]) -> Option<TaskLevel> {
    let mut by_family: std::collections::BTreeMap<&str, (f64, usize)> = Default::default();
    let (mut wins, mut losses, mut paired) = (0usize, 0usize, 0usize);
    for outcome in outcomes {
        let Some((candidate, baseline)) = outcome.paired() else {
            continue;
        };
        paired += 1;
        wins += usize::from(candidate && !baseline);
        losses += usize::from(!candidate && baseline);
        let family = outcome.cluster.as_deref().unwrap_or(outcome.item.as_str());
        let entry = by_family.entry(family).or_default();
        entry.0 += f64::from(candidate) - f64::from(baseline);
        entry.1 += 1;
    }
    if paired == 0 {
        return None;
    }
    let sum: f64 = by_family.values().map(|f| f.0).sum();
    let difference = sum / paired as f64;
    let interval = interval_of(by_family.into_values().collect(), TASK_LEVEL_RESAMPLES);
    Some(TaskLevel {
        p_value: sign_p(wins, losses),
        difference,
        low: interval.map_or(difference, |i| i.low),
        high: interval.map_or(difference, |i| i.high),
    })
}

/// `part` with the tasks' own evidence over `outcomes`.
pub(super) fn with_task_level(
    mut part: gate::PartResult,
    outcomes: &[PairedOutcome],
) -> gate::PartResult {
    part.task_level = task_level(outcomes);
    part
}

/// The share of resampled drops the anchor's interval holds.
const DROP_INTERVAL_LEVEL: f64 = 0.95;
/// Resamples the anchor's interval is made from.
const DROP_INTERVAL_RESAMPLES: usize = 4000;

/// The bootstrap interval of the champion's accuracy minus the candidate's
/// over the items both were graded on: an item the champion alone got right
/// is 1, the candidate alone -1, a tie 0.
pub(super) fn drop_interval(outcomes: &[PairedOutcome]) -> Option<DropInterval> {
    let drops: Vec<f64> = outcomes
        .iter()
        .filter_map(PairedOutcome::paired)
        .map(|(candidate, baseline)| f64::from(baseline) - f64::from(candidate))
        .collect();
    bootstrap_interval(&drops, DROP_INTERVAL_RESAMPLES, DROP_INTERVAL_LEVEL, 0).map(|i| {
        DropInterval {
            level: DROP_INTERVAL_LEVEL,
            low: i.low,
            high: i.high,
        }
    })
}

/// One-sided 5% and 80% power quantiles of the standard normal, summed and
/// squared: the factor of the sample size of a paired difference.
const POWER_FACTOR: f64 = 6.18;

/// The drop a suite is asked to be able to see, in share of items.
const WANTED_DROP: f64 = 0.02;

/// The paired items it takes to see a drop of [`WANTED_DROP`] with four
/// chances in five, at the discordance `comparison` found: the variance of an
/// item's difference is about the share of items on which the arms differ.
pub(super) fn items_to_see_two_points(
    comparison: &splinter_eval::paired::Comparison,
) -> Option<usize> {
    let discordant = comparison.candidate_wins + comparison.baseline_wins;
    (comparison.paired > 0 && discordant > 0).then(|| {
        let share = discordant as f64 / comparison.paired as f64;
        (POWER_FACTOR * share / (WANTED_DROP * WANTED_DROP)).ceil() as usize
    })
}
