// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What the curriculum prints as text: the weakest concepts.

use std::fmt::Write as _;

use splinter_sdk::curriculum::mastery::{MasteryReport, ReleaseMastery};

use crate::output::Report;

fn rate(value: Option<f64>) -> String {
    value.map_or("not measured".into(), |v| format!("{v:.3}"))
}

/// A release as mastery names it.
fn release(mastery: &ReleaseMastery) -> String {
    mastery
        .release
        .as_ref()
        .map_or("the base".into(), ToString::to_string)
}

impl Report for MasteryReport {
    fn human(&self) -> String {
        let under = self
            .policy
            .release
            .as_ref()
            .map_or("the base".into(), |r| format!("release {r}"));
        let mut out = format!(
            "concepts: {} measured under {under} of {} seen\n",
            self.measured, self.concepts
        );
        for concept in &self.weakest {
            let current = concept.current.as_ref();
            let _ = write!(
                out,
                "  {}  {} ({} of {})",
                rate(current.and_then(|m| m.rate)),
                concept.concept,
                current.map_or(0, |m| m.passes),
                current.map_or(0, |m| m.graded)
            );
            let earlier: Vec<String> = concept
                .releases
                .iter()
                .filter(|m| m.release != self.policy.release)
                .map(|m| format!("{} under {}", rate(m.rate), release(m)))
                .collect();
            if !earlier.is_empty() {
                let _ = write!(out, "; earlier {}", earlier.join(", "));
            }
            out.push('\n');
        }
        out
    }
}
