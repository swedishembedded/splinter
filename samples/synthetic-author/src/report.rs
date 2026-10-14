// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What an examination found: how often each arm took the option the beliefs
//! determine, how sure that is, and whether one arm beat another on the same
//! items by more than chance allows.

use std::fmt::Write as _;

use splinter_sdk::model::stats::sign_test;

use crate::arms::Outcome;

/// A rate and the 95% Wilson interval around it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rate {
    /// Items right.
    pub right: usize,
    /// Items asked.
    pub n: usize,
    /// The share right.
    pub rate: f64,
    /// The interval's lower end.
    pub low: f64,
    /// The interval's upper end.
    pub high: f64,
}

/// The rate at which `outcomes` are right.
#[must_use]
pub fn accuracy(outcomes: &[Outcome]) -> Rate {
    rate(
        outcomes.iter().filter(|o| o.correct).count(),
        outcomes.len(),
    )
}

fn rate(right: usize, n: usize) -> Rate {
    if n == 0 {
        return Rate {
            right,
            n,
            rate: 0.0,
            low: 0.0,
            high: 0.0,
        };
    }
    let z = 1.96_f64;
    let (r, n_f) = (right as f64, n as f64);
    let p = r / n_f;
    let denominator = 1.0 + z * z / n_f;
    let centre = (p + z * z / (2.0 * n_f)) / denominator;
    let half = z * ((p * (1.0 - p) + z * z / (4.0 * n_f)) / n_f).sqrt() / denominator;
    Rate {
        right,
        n,
        rate: p,
        low: centre - half,
        high: centre + half,
    }
}

fn cell(outcomes: &[Outcome], keep: impl Fn(&Outcome) -> bool) -> String {
    let chosen: Vec<_> = outcomes.iter().filter(|o| keep(o)).cloned().collect();
    let r = accuracy(&chosen);
    format!("{:.3} ({}/{})", r.rate, r.right, r.n)
}

/// A table of every arm and, for each pair in `compare`, a paired test of the
/// first against the second on the items both were asked.
#[must_use]
pub fn render(arms: &[(&str, Vec<Outcome>)], compare: &[(&str, &str)]) -> String {
    let mut text = String::from(
        "| arm | all | topic seen | topic new | combination seen | combination new | tool seen | tool new |\n|---|---|---|---|---|---|---|---|\n",
    );
    for (name, outcomes) in arms {
        let r = accuracy(outcomes);
        let _ = writeln!(
            text,
            "| {name} | {:.3} [{:.3}, {:.3}] ({}/{}) | {} | {} | {} | {} | {} | {} |",
            r.rate,
            r.low,
            r.high,
            r.right,
            r.n,
            cell(outcomes, |o| o.topic_seen),
            cell(outcomes, |o| !o.topic_seen),
            cell(outcomes, |o| o.combo_seen),
            cell(outcomes, |o| !o.combo_seen),
            cell(outcomes, |o| o.tool_seen),
            cell(outcomes, |o| !o.tool_seen),
        );
    }
    text.push('\n');
    for (a, b) in compare {
        let (Some((_, first)), Some((_, second))) = (
            arms.iter().find(|(n, _)| n == a),
            arms.iter().find(|(n, _)| n == b),
        ) else {
            continue;
        };
        let pairs: Vec<(bool, bool)> = first
            .iter()
            .filter_map(|x| {
                second
                    .iter()
                    .find(|y| y.id == x.id)
                    .map(|y| (x.correct, y.correct))
            })
            .collect();
        let test = sign_test(&pairs);
        let _ = writeln!(
            text,
            "{a} over {b}: {} of {} items they differ on, p = {:.4} (one-sided exact sign test, {} items)",
            test.candidate_wins,
            test.discordant,
            test.p_value,
            pairs.len(),
        );
    }
    text
}
