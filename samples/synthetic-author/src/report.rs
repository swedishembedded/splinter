// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What an examination found: how often each arm took the option the beliefs
//! determine, how sure that is, and whether one arm beat another on the same
//! items by more than chance allows.

use std::fmt::Write as _;

use splinter_sdk::model::stats::sign_test;

use crate::arms::Outcome;
use crate::exam::Item;
use crate::persona::Persona;

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

/// What a table shows for a rate over no items: an unmeasured rate is not
/// zero.
const NOT_MEASURED: &str = "not measured";

fn cell(outcomes: &[Outcome], keep: impl Fn(&Outcome) -> bool) -> String {
    let chosen: Vec<_> = outcomes.iter().filter(|o| keep(o)).cloned().collect();
    if chosen.is_empty() {
        return NOT_MEASURED.to_string();
    }
    let r = accuracy(&chosen);
    format!("{:.3} ({}/{})", r.rate, r.right, r.n)
}

/// A table of every arm and, for each pair in `compare`, a paired test of the
/// first against the second on the items both were asked. `missing` names the
/// arms that were asked for and have no outcomes: they are listed, and a
/// comparison with one of them says it was not measured instead of vanishing.
#[must_use]
pub fn render(arms: &[(&str, Vec<Outcome>)], compare: &[(&str, &str)], missing: &[&str]) -> String {
    let mut text = String::from(
        "| arm | all | topic seen | topic new | combination seen | combination new | tool seen | tool new |\n|---|---|---|---|---|---|---|---|\n",
    );
    for (name, outcomes) in arms {
        let r = accuracy(outcomes);
        let all = if outcomes.is_empty() {
            NOT_MEASURED.to_string()
        } else {
            format!(
                "{:.3} [{:.3}, {:.3}] ({}/{})",
                r.rate, r.low, r.high, r.right, r.n
            )
        };
        let _ = writeln!(
            text,
            "| {name} | {all} | {} | {} | {} | {} | {} | {} |",
            cell(outcomes, |o| o.topic_seen),
            cell(outcomes, |o| !o.topic_seen),
            cell(outcomes, |o| o.combo_seen),
            cell(outcomes, |o| !o.combo_seen),
            cell(outcomes, |o| o.tool_seen),
            cell(outcomes, |o| !o.tool_seen),
        );
    }
    text.push('\n');
    for name in missing {
        let _ = writeln!(text, "{name}: not measured (no outcomes for this arm)");
    }
    for (a, b) in compare {
        let (Some((_, first)), Some((_, second))) = (
            arms.iter().find(|(n, _)| n == a),
            arms.iter().find(|(n, _)| n == b),
        ) else {
            if missing.contains(a) || missing.contains(b) {
                let _ = writeln!(text, "{a} over {b}: not measured");
            }
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

/// How a model's answers divide, on the situations where a second author
/// would decide otherwise than the author it was trained on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Divergence {
    /// Situations where both authors are determined and decide differently.
    pub differing: usize,
    /// Answers that take the trained author's decision.
    pub own: usize,
    /// Answers that take the second author's decision.
    pub rival: usize,
    /// Answers that take neither, or make no call.
    pub neither: usize,
}

/// Counts, over the items of `items` where `rival` is determined and decides
/// differently from the author the items were drawn for, whose decision each
/// of `outcomes` took.
#[must_use]
pub fn divergence(items: &[Item], outcomes: &[Outcome], rival: &Persona) -> Divergence {
    let mut d = Divergence {
        differing: 0,
        own: 0,
        rival: 0,
        neither: 0,
    };
    for item in items {
        let other = rival.decide(&item.scenario);
        if !other.determined || other.choice == item.expected {
            continue;
        }
        let Some(outcome) = outcomes.iter().find(|o| o.id == item.id) else {
            continue;
        };
        d.differing += 1;
        let taken = outcome.call.as_ref().and_then(|c| {
            item.schema
                .read(c["name"].as_str()?, &c["arguments"], &item.options)
        });
        match taken {
            Some(t) if t == item.expected => d.own += 1,
            Some(t) if t == other.choice => d.rival += 1,
            _ => d.neither += 1,
        }
    }
    d
}
