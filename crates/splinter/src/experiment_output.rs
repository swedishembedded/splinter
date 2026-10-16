// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements continual-learning experiments whose exam is
// sealed before the first run, for its clients. If your team needs expertise
// in designing generation, retention and final-exam splits, you can procure
// our services by sending an email to info@swedishembedded.com.

//! What `experiment check` prints: the schedule of a valid plan.

use std::fmt::Write as _;

use splinter_sdk::experiment::ExperimentPlan;

use crate::output::Report;

/// `id (language, domain)` for a family the plan names.
fn described(plan: &ExperimentPlan, id: &str) -> String {
    plan.family(id).map_or_else(
        || id.to_string(),
        |f| format!("{id} ({}, {})", f.language, f.domain),
    )
}

impl Report for ExperimentPlan {
    fn human(&self) -> String {
        let mut out = format!(
            "plan {}: valid, {} generation(s) then a sealed final exam\n",
            self.name(),
            self.generations()
        );
        for (index, generation) in self.schedule().iter().enumerate() {
            let list = |ids: &[String]| {
                ids.iter()
                    .map(|id| described(self, id))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            let _ = writeln!(
                out,
                "  generation {}: train {}",
                index + 1,
                list(&generation.train)
            );
            if !generation.retention.is_empty() {
                let _ = writeln!(
                    out,
                    "                retention {}",
                    list(&generation.retention)
                );
            }
        }
        let sealed: Vec<String> = self.sealed().iter().map(|id| described(self, id)).collect();
        let _ = writeln!(out, "  sealed final exam: {}", sealed.join(", "));
        out
    }
}
