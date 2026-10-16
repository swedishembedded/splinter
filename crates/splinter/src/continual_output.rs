// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements measurement of what a continually updated
// model keeps and what it loses, for its clients. If your team needs
// expertise in continual-learning evaluation for fine-tuned models, you can
// procure our services by sending an email to info@swedishembedded.com.

//! What `status` prints of the continual-learning metrics.

use std::fmt::Write as _;

use splinter_sdk::measure::continual::{ContinualMetrics, Measure, ResultsMatrix};
use splinter_sdk::release::continual::{ContinualStatus, MatrixReport};

use crate::output::Report;

/// One metric on one line: the value, what it rests on and its interval, or
/// that it was not measured and why.
fn line(name: &str, measure: &Measure) -> String {
    match measure {
        Measure::Measured {
            value,
            tasks,
            units,
            interval,
            ..
        } => {
            let interval = interval.map_or_else(
                || ", no interval (too few units)".into(),
                |i| {
                    format!(
                        ", {:.0}% interval [{:+.3}, {:+.3}]",
                        i.level * 100.0,
                        i.low,
                        i.high
                    )
                },
            );
            format!(
                "  {name:<17} {value:+.3}  over {tasks} suite(s) in {units} unit(s){interval}\n"
            )
        }
        Measure::NotMeasured { reason } => format!("  {name:<17} not measured: {reason}\n"),
    }
}

/// The section `status` prints.
pub fn human(status: &ContinualStatus) -> String {
    match status {
        ContinualStatus::NotMeasured { reason } => {
            format!("continual: not measured: {reason}\n")
        }
        ContinualStatus::Measured { releases, metrics } => {
            let mut out = format!("continual: {} release(s) as generations\n", releases.len());
            out.push_str(&metric_lines(metrics));
            out
        }
    }
}

/// The metrics, one per line, each with what it rests on or why it was not
/// measured.
fn metric_lines(metrics: &ContinualMetrics) -> String {
    let mut out = String::new();
    out.push_str(&line("ACC", &metrics.acc));
    out.push_str(&line("BWT", &metrics.bwt));
    out.push_str(&line("forgetting", &metrics.forgetting));
    out.push_str(&line("forward transfer", &metrics.forward_transfer));
    out.push_str(&line("learning gain", &metrics.learning_gain));
    for point in &metrics.gain_curve {
        let _ = write!(out, "  gain after gen {}:", point.generation);
        out.push_str(&line("", &point.gain).replacen("  ", "", 1));
    }
    out
}

/// The matrix as a table: a row per generation, a column per task, `-` where
/// no gate measured the cell (an unmeasured cell is never printed as 0).
fn table(matrix: &ResultsMatrix) -> String {
    let mut out = String::from("  generation");
    for task in 0..matrix.tasks().len() {
        let _ = write!(out, "  task {task:<3}");
    }
    out.push('\n');
    for generation in 0..matrix.generations() {
        let _ = write!(out, "  {generation:<10}");
        for task in 0..matrix.tasks().len() {
            match matrix.cell(generation, task) {
                Some(v) => {
                    let _ = write!(out, "  {v:<8.3}");
                }
                None => out.push_str("  -       "),
            }
        }
        out.push('\n');
    }
    out
}

impl Report for MatrixReport {
    fn human(&self) -> String {
        let mut out = format!(
            "results matrix of {}: {} release(s), task k is release k's own held-out suite\n",
            self.alias,
            self.releases.len()
        );
        out.push_str(&table(&self.matrix));
        for (k, release) in self.releases.iter().enumerate() {
            let _ = writeln!(out, "  {k}: {release}");
        }
        out.push_str(&metric_lines(&self.metrics));
        out
    }
}

#[cfg(test)]
mod tests {
    use splinter_sdk::measure::continual::{ResultsMatrix, TaskColumn};

    use super::*;

    #[test]
    fn a_cell_nobody_measured_prints_a_dash_and_never_zero() {
        let column = |id: &str, learned_at| TaskColumn {
            id: id.into(),
            unit: id.into(),
            learned_at,
            base: None,
        };
        let mut matrix = ResultsMatrix::new(2, vec![column("a", 0), column("b", 1)]).unwrap();
        matrix.set(0, 0, 0.5).unwrap();
        matrix.set(1, 1, 0.25).unwrap();
        let text = table(&matrix);
        let rows: Vec<&str> = text.lines().collect();
        assert!(rows[1].contains("0.500") && rows[1].contains('-'), "{text}");
        assert!(
            rows[2].starts_with("  1") && rows[2].contains("0.250"),
            "{text}"
        );
        assert!(!text.contains("0.000"), "{text}");
    }
}
