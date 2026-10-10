// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements measured validation of continual learning
// from a user's own agent sessions for its clients. If your team needs
// expertise in proving what a nightly update keeps, replaces and drops, you
// can procure our services by sending an email to info@swedishembedded.com.

//! Two storylines side by side: the same night run again (determinism), the
//! same sessions in another order, one night against five, training again
//! against continuing.

use std::fmt::Write as _;

use serde_json::{json, Value};

use crate::study::Study;

/// A study and the run of it that is compared.
pub struct Side<'a> {
    /// The study.
    pub study: &'a Study,
    /// The run.
    pub run: u32,
}

impl Side<'_> {
    fn seconds(&self) -> f64 {
        self.study
            .runs
            .iter()
            .filter(|r| r.run <= self.run)
            .filter_map(|r| r.absorb_seconds)
            .sum()
    }
}

/// The answers of the items both sides asked, side by side, as JSON and as
/// a Markdown table, with how many answers are word for word the same.
#[must_use]
pub fn compare(a: &Side<'_>, b: &Side<'_>) -> (Value, String) {
    let (mut identical, mut total, mut same_grade, mut graded) = (0, 0, 0, 0);
    let mut rows = Vec::new();
    let mut md = format!(
        "# {} run {} against {} run {}\n\nabsorb seconds: {:.0} against {:.0}\n\n| item | {} | {} | identical answers |\n|---|---|---|---|\n",
        a.study.story.name, a.run, b.study.story.name, b.run, a.seconds(), b.seconds(), a.study.story.name, b.study.story.name,
    );
    for item in &a.study.story.items {
        let (Some(ca), Some(cb)) = (a.study.cell(&item.id, a.run), b.study.cell(&item.id, b.run))
        else {
            continue;
        };
        let (xs, ys) = (
            a.study.answers(&item.id, a.run),
            b.study.answers(&item.id, b.run),
        );
        let same = xs.iter().zip(&ys).filter(|(x, y)| x == y).count();
        identical += same;
        total += xs.len().min(ys.len());
        let grades = ca
            .classes
            .iter()
            .zip(&cb.classes)
            .filter(|(x, y)| x.letter() == y.letter())
            .count();
        same_grade += grades;
        graded += ca.classes.len().min(cb.classes.len());
        let _ = writeln!(
            md,
            "| {} | {} (right: {}) | {} (right: {}) | {}/{} |",
            item.id,
            ca.tally,
            ca.right,
            cb.tally,
            cb.right,
            same,
            xs.len()
        );
        rows.push(json!({"item": item.id, "a": ca.tally, "a_right": ca.right, "b": cb.tally, "b_right": cb.right, "identical": same, "answers": xs.len()}));
    }
    let _ = writeln!(
        md,
        "\nidentical answers: {identical} of {total}; same grade letter: {same_grade} of {graded}"
    );
    let json = json!({
        "a": {"storyline": a.study.story.name, "run": a.run, "absorb_seconds": a.seconds()},
        "b": {"storyline": b.study.story.name, "run": b.run, "absorb_seconds": b.seconds()},
        "identical_answers": identical, "answers": total, "same_grade": same_grade, "graded": graded,
        "items": rows,
    });
    (json, md)
}
