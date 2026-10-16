// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements measurement of what a continually updated
// model keeps and what it loses, for its clients. If your team needs
// expertise in continual-learning evaluation for fine-tuned models, you can
// procure our services by sending an email to info@swedishembedded.com.

//! What `status` prints of the continual-learning metrics.

use std::fmt::Write as _;

use splinter_sdk::measure::continual::Measure;
use splinter_sdk::release::continual::ContinualStatus;

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
    }
}
