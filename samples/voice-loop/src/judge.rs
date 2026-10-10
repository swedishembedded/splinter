// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements judges of synthetic speech calibrated against
// the people who listen to it, for its clients. If your team needs expertise
// in learning a listener's preference from a few hundred comparisons, you can
// procure our services by sending an email to info@swedishembedded.com.

//! `voice-loop judge`: fit the naturalness judge to a listener's blind A/B
//! choices and report how well it predicts the ones it did not see.
//!
//! The listening session is a directory with `key.json` (for each item its kind,
//! `pair`, `control` or `repeat`, and which take sat on which side) and
//! `audio/<id>-a.wav`, `audio/<id>-b.wav`; the ratings are a directory of
//! `<id>.json` files holding the listener's `choice` (`a`, `b` or `same`) and
//! whether either side was marked garbled. Only choices of `pair` items train
//! the judge. Controls (a take against itself with noise added) and repeats
//! (a pair heard twice, sides swapped) are the checks on the listener and on
//! the judge, and are reported, not fitted.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Args;
use serde::{Deserialize, Serialize};
use splinter_sdk::measure::speech_features::{features, NAMES};
use splinter_sdk::measure::speech_judge::{leave_one_out, Comparison, FitOptions, Judge};

use crate::batch::{emit, read_clip};

#[derive(Args)]
pub struct JudgeArgs {
    /// The listening session: `key.json` and `audio/`.
    #[arg(long)]
    session: PathBuf,
    /// The directory of the listener's ratings, one `<id>.json` per item.
    #[arg(long)]
    ratings: PathBuf,
    /// Also keep the report here.
    #[arg(long)]
    report: Option<PathBuf>,
}

#[derive(Deserialize)]
struct Item {
    id: String,
    kind: String,
    a: String,
    b: String,
}

#[derive(Deserialize)]
struct Rating {
    choice: String,
    #[serde(default)]
    garbled_a: bool,
    #[serde(default)]
    garbled_b: bool,
}

#[derive(Serialize)]
struct Report {
    portrayal: &'static str,
    choices_used: usize,
    marked_same: usize,
    leave_one_out_agreement: f32,
    /// Exact one-sided probability of this many agreements or more by chance.
    sign_test_p: f64,
    feature_weights: Vec<(String, f32)>,
    controls: Controls,
    repeats: Repeats,
}

#[derive(Serialize, Default)]
struct Controls {
    rated: usize,
    noise_marked_garbled: usize,
    clean_chosen: usize,
    marked_same: usize,
    judge_ranks_clean_higher: usize,
}

#[derive(Serialize, Default)]
struct Repeats {
    rated_twice: usize,
    same_take_chosen: usize,
}

fn rating(dir: &Path, id: &str) -> Option<Rating> {
    serde_json::from_slice(&std::fs::read(dir.join(format!("{id}.json"))).ok()?).ok()
}

fn side_features(session: &Path, id: &str, side: &str) -> Result<Vec<f32>> {
    let clip = read_clip(&session.join("audio").join(format!("{id}-{side}.wav")))?;
    features(clip.samples(), clip.sample_rate())
        .map(|f| f.to_vec())
        .with_context(|| format!("{id}-{side} is shorter than one frame"))
}

/// The chance of `right` or more of `n` fair coin flips.
fn sign_test_p(right: usize, n: usize) -> f64 {
    let mut term = 1.0f64;
    let mut tail = 0.0;
    for k in 0..=n {
        if k >= right {
            tail += term;
        }
        term = term * (n - k) as f64 / (k + 1) as f64;
    }
    tail / 2f64.powi(n as i32)
}

pub fn run(args: &JudgeArgs) -> Result<()> {
    let items: Vec<Item> = serde_json::from_slice(
        &std::fs::read(args.session.join("key.json")).context("reading key.json")?,
    )?;
    let mut comparisons = Vec::new();
    let mut marked_same = 0;
    for item in items.iter().filter(|i| i.kind == "pair") {
        let Some(r) = rating(&args.ratings, &item.id) else {
            continue;
        };
        let (winner, loser) = match r.choice.as_str() {
            "a" => ("a", "b"),
            "b" => ("b", "a"),
            _ => {
                marked_same += 1;
                continue;
            }
        };
        comparisons.push(Comparison {
            winner: side_features(&args.session, &item.id, winner)?,
            loser: side_features(&args.session, &item.id, loser)?,
            weight: 1.0,
        });
    }
    let options = FitOptions::default();
    let judge = Judge::fit(&comparisons, &options);
    let agreement = leave_one_out(&comparisons, &options);
    let right = (agreement * comparisons.len() as f32).round() as usize;

    let mut controls = Controls::default();
    for item in items.iter().filter(|i| i.kind == "control") {
        let Some(r) = rating(&args.ratings, &item.id) else {
            continue;
        };
        let noisy_side = if item.a.ends_with("+noise") { "a" } else { "b" };
        let clean_side = if noisy_side == "a" { "b" } else { "a" };
        controls.rated += 1;
        controls.noise_marked_garbled += usize::from(if noisy_side == "a" {
            r.garbled_a
        } else {
            r.garbled_b
        });
        controls.marked_same += usize::from(r.choice == "same");
        controls.clean_chosen += usize::from(r.choice == clean_side);
        let (noisy, clean) = (
            side_features(&args.session, &item.id, noisy_side)?,
            side_features(&args.session, &item.id, clean_side)?,
        );
        controls.judge_ranks_clean_higher += usize::from(judge.score(&clean) > judge.score(&noisy));
    }

    // A repeat is the same two takes as a rated pair, heard again; the listener
    // is consistent when the same take is chosen both times.
    let mut repeats = Repeats::default();
    let take_chosen = |item: &Item, r: &Rating| match r.choice.as_str() {
        "a" => Some(item.a.clone()),
        "b" => Some(item.b.clone()),
        _ => None,
    };
    for repeat in items.iter().filter(|i| i.kind == "repeat") {
        let original = items.iter().find(|i| {
            i.kind == "pair"
                && ((i.a == repeat.a && i.b == repeat.b) || (i.a == repeat.b && i.b == repeat.a))
        });
        let (Some(original), Some(second)) = (original, rating(&args.ratings, &repeat.id)) else {
            continue;
        };
        let Some(first) = rating(&args.ratings, &original.id) else {
            continue;
        };
        if let (Some(x), Some(y)) = (take_chosen(original, &first), take_chosen(repeat, &second)) {
            repeats.rated_twice += 1;
            repeats.same_take_chosen += usize::from(x == y);
        }
    }

    emit(
        &Report {
            portrayal:
                "Synthetic theatrical portrayal; no recording of any real person's voice was used",
            choices_used: comparisons.len(),
            marked_same,
            leave_one_out_agreement: agreement,
            sign_test_p: sign_test_p(right, comparisons.len()),
            feature_weights: NAMES
                .iter()
                .map(|n| n.to_string())
                .zip(judge.weights().iter().copied())
                .collect(),
            controls,
            repeats,
        },
        args.report.as_deref(),
    )
}

#[cfg(test)]
mod tests {
    use super::sign_test_p;

    #[test]
    fn the_sign_test_is_the_exact_binomial_tail() {
        assert!((sign_test_p(0, 10) - 1.0).abs() < 1e-12);
        assert!((sign_test_p(10, 10) - 1.0 / 1024.0).abs() < 1e-12);
        assert!((sign_test_p(8, 10) - 56.0 / 1024.0).abs() < 1e-12);
    }
}
