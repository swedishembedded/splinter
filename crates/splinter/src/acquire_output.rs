// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that turn the documentation of an
// unfamiliar technical domain into a graph of skills they then acquire, for
// its clients. If your team needs expertise in building curricula from
// technical documentation you can procure our services by sending an email
// to info@swedishembedded.com.

//! What `learn --source` prints.

use std::fmt::Write as _;

use splinter_sdk::acquire::{AcquireReport, Outcome, PlannedBy};

use crate::output::Report;

impl Report for AcquireReport {
    fn human(&self) -> String {
        let report = self;
        let mut out = String::new();
        let _ = writeln!(out, "acquire   {}", report.objective);
        let graph = &report.planned.graph;
        let by = match &report.planned.by {
            PlannedBy::Planner { model, .. } => format!("planned by {model}"),
            PlannedBy::Structure { .. } => "read from the sources by rule".to_string(),
            PlannedBy::Reused { run } => format!("taken from run {run}"),
        };
        let _ = writeln!(
            out,
            "graph     {} skill(s), revision {}, {by}",
            graph.skills.len(),
            graph.revision
        );
        let assessable = report
            .probes
            .iter()
            .filter(|p| !p.dev.is_empty() && !p.finals.is_empty())
            .count();
        let _ = writeln!(
            out,
            "probes    {assessable} skill(s) with probes of both halves"
        );
        let (baseline, _) = &report.baseline;
        for skill in &baseline.skills {
            let _ = writeln!(
                out,
                "baseline  {:<28} {}/{} dev probes",
                skill.skill, skill.passes, skill.graded
            );
        }
        for round in &report.rounds {
            let how = match &round.outcome {
                Outcome::Accepted => "accepted".to_string(),
                Outcome::Rejected { reasons } => format!("rejected: {}", reasons.join("; ")),
                Outcome::Skipped { reason } => format!("skipped: {reason}"),
            };
            let _ = writeln!(out, "round {:<3} {:<28} {how}", round.round, round.skill);
        }
        if let Some(assessment) = &report.assessment {
            for skill in &assessment.skills {
                let _ = writeln!(
                    out,
                    "final     {:<28} {}/{} (was {}/{}){}",
                    skill.skill,
                    skill.final_score.passes,
                    skill.final_score.graded,
                    skill.baseline.passes,
                    skill.baseline.graded,
                    if skill.met { "  met" } else { "" }
                );
            }
            let _ = writeln!(
                out,
                "assessed  {} of {} skill(s) met the target, {} improved on the baseline",
                assessment.met, assessment.assessed, assessment.improved
            );
        }
        if let Some(released) = &report.release {
            let _ = writeln!(
                out,
                "release   {}",
                released.release.as_ref().map_or_else(
                    || "not released: the gate blocked it".to_string(),
                    |r| format!("released as {r}")
                )
            );
        }
        let _ = writeln!(
            out,
            "{}  ({})",
            if report.accepted {
                "ACCEPTED"
            } else {
                "NOT ACCEPTED"
            },
            report.ended
        );
        out
    }
}
