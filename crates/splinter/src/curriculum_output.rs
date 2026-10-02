// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What the curriculum prints as text: a pass@k measurement, the teacher's
//! solves and the tasks kept, a training set's quotas, and the weakest
//! concepts.

use std::fmt::Write as _;

use splinter_campaign::curriculum::frontier::{Frontier, Measured};
use splinter_campaign::curriculum::mastery::{MasteryReport, ReleaseMastery};
use splinter_campaign::curriculum::quota::Selected;
use splinter_campaign::curriculum::teacher::Taught;

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

impl Report for Frontier {
    fn human(&self) -> String {
        let d = &self.distribution;
        let measured = match &self.policy {
            Some(policy) => match &policy.release {
                Some(release) => format!("policy:{} = release {release}", policy.alias),
                None => format!("policy:{} = the base", policy.alias),
            },
            None => self.solver.clone(),
        };
        let sampling = self.sampling.map_or("as its provider samples".into(), |s| {
            format!("temperature {}, top-k {}", s.temperature, s.top_k)
        });
        format!(
            "pass@{} of {} ({sampling}) over {} task(s), taught by {}: {} kept - {} on the \
             frontier, {} taught; {} always solved (no signal), {} never solved with no \
             verified answer, {} unmeasured\n  frontier task set {} (attempts and verified \
             answers: {})\n  measurement {}\n",
            self.k,
            measured,
            d.tasks(),
            self.teacher,
            d.kept(),
            d.frontier,
            d.taught,
            d.always,
            d.never,
            d.unmeasured,
            self.frontier_task_set,
            self.frontier_experience_set,
            self.measurement
        )
    }
}

impl Report for Taught {
    fn human(&self) -> String {
        format!(
            "open-book, the tasks never solved closed-book ({}):\n{}{}",
            self.task_set,
            self.solve.human(),
            self.verify.human()
        )
    }
}

impl Report for Measured {
    fn human(&self) -> String {
        format!(
            "{}{}{}{}",
            self.solve.human(),
            self.verify.human(),
            self.teach.human(),
            self.frontier.human()
        )
    }
}

impl Report for Selected {
    fn human(&self) -> String {
        let s = &self.selection;
        let mut out = format!(
            "{} of {} passing experience(s) kept in {}: {} repeat(s) dropped\n  quotas: concept \
             {}, kind {}, strength {}\n",
            s.selected.len(),
            s.candidates,
            self.experience_set,
            s.duplicates,
            self.quotas.max_concept_share,
            self.quotas.max_kind_share,
            self.quotas.max_strength_share
        );
        for (dimension, dropped) in &s.capped {
            let _ = writeln!(out, "  capped by {dimension}: {dropped} dropped");
        }
        out
    }
}

impl Report for MasteryReport {
    fn human(&self) -> String {
        let under = self
            .policy
            .release
            .as_ref()
            .map_or("the base".into(), |r| format!("release {r}"));
        let mut out = format!(
            "concepts: {} measured under {under} of {} seen; {} queued for new tasks\n",
            self.measured, self.concepts, self.queued
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
