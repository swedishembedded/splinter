// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements controlled validation of learning systems
// against ground truth that cannot have been seen in pretraining, for its
// clients. If your team needs expertise in measuring how stable a model's
// answers are under sampling, you can procure our services by sending an
// email to info@swedishembedded.com.

//! The sampled results beside the greedy ones.
//!
//! The unit is the item: an item's share right is the share of its draws that
//! were right, an arm's interval is a bootstrap over its items, and two arms
//! are compared by a bootstrap over the per-item differences of their shares,
//! so a gain that rests on a few items shows as a wide interval. The
//! bootstrap's seed is part of the text it produces.

use std::fmt::Write as _;

use splinter_sdk::model::stats::{bootstrap_interval, Interval};

use crate::arms::Outcome;
use crate::report::accuracy;
use crate::sampled::SampledOutcome;

/// How many resamples of the items an interval is made from.
const RESAMPLES: usize = 2000;
/// The share of resampled means an interval holds.
const LEVEL: f64 = 0.95;
/// What a table shows for a number that was not measured.
const NOT_MEASURED: &str = "not measured";

fn shares(outcomes: &[SampledOutcome]) -> Vec<(&str, f64)> {
    outcomes
        .iter()
        .filter_map(|o| Some((o.id.as_str(), o.share_right()?)))
        .collect()
}

fn interval_text(interval: Option<Interval>) -> String {
    interval.map_or_else(
        || NOT_MEASURED.to_string(),
        |i| format!("{:.3} [{:.3}, {:.3}]", i.mean, i.low, i.high),
    )
}

/// The table of every arm with greedy and sampled results side by side, and
/// for each pair in `compare` the mean difference of their shares right with
/// its interval, over the items both have draws for. `greedy` are the arms'
/// greedy outcomes, `sampled` their sampled ones; `bootstrap_seed` seeds every
/// interval and is named in the text.
#[must_use]
pub fn render(
    greedy: &[(&str, Vec<Outcome>)],
    sampled: &[(&str, Vec<SampledOutcome>)],
    compare: &[(&str, &str)],
    bootstrap_seed: u64,
) -> String {
    let mut text = format!(
        "Sampled decoding: the share of an item's draws that were right, averaged over items, \
         with a {:.0}% bootstrap interval over items ({RESAMPLES} resamples, bootstrap seed {bootstrap_seed}).\n\n\
         | arm | greedy | sampled |\n|---|---|---|\n",
        LEVEL * 100.0
    );
    for (name, outcomes) in sampled {
        let greedy_cell = greedy
            .iter()
            .find(|(n, _)| n == name)
            .filter(|(_, o)| !o.is_empty())
            .map_or_else(
                || NOT_MEASURED.to_string(),
                |(_, o)| {
                    let r = accuracy(o);
                    format!("{:.3} ({}/{})", r.rate, r.right, r.n)
                },
            );
        let per_item: Vec<f64> = shares(outcomes).into_iter().map(|(_, s)| s).collect();
        let draws = outcomes.first().map_or(0, |o| o.draws.len());
        let sampled_cell = if per_item.is_empty() {
            NOT_MEASURED.to_string()
        } else {
            format!(
                "{} ({draws} draws)",
                interval_text(bootstrap_interval(
                    &per_item,
                    RESAMPLES,
                    LEVEL,
                    bootstrap_seed
                ))
            )
        };
        let _ = writeln!(text, "| {name} | {greedy_cell} | {sampled_cell} |");
    }
    text.push('\n');
    for (a, b) in compare {
        let (Some((_, first)), Some((_, second))) = (
            sampled.iter().find(|(n, _)| n == a),
            sampled.iter().find(|(n, _)| n == b),
        ) else {
            let _ = writeln!(text, "{a} over {b}: {NOT_MEASURED}");
            continue;
        };
        let second = shares(second);
        let differences: Vec<f64> = shares(first)
            .into_iter()
            .filter_map(|(id, x)| {
                second
                    .iter()
                    .find(|(other, _)| *other == id)
                    .map(|(_, y)| x - y)
            })
            .collect();
        let _ = writeln!(
            text,
            "{a} over {b}: mean difference in share right {} over {} items",
            interval_text(bootstrap_interval(
                &differences,
                RESAMPLES,
                LEVEL,
                bootstrap_seed
            )),
            differences.len(),
        );
    }
    text
}
