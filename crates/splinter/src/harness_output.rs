// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements automated improvement of agent harnesses
// that adopts a change only on verified, regression-free evidence, for its
// clients. If your team needs expertise in measuring whether a harness
// change really helped, you can procure our services by sending an email to
// info@swedishembedded.com.

//! How `harness optimise` reads to a person.

use std::fmt::Write as _;

use splinter_sdk::harness_opt::Optimised;

use crate::output::Report;

impl Report for Optimised {
    fn human(&self) -> String {
        let mut out = format!(
            "baseline {} (experience set {})\n{} harness(es) evaluated, {} verified, search ended: {:?}\n",
            self.baseline,
            self.baseline_set,
            self.metrics.evaluations,
            self.metrics.verified_evaluations,
            self.stop,
        );
        for line in &self.candidates {
            let _ = writeln!(
                out,
                "  #{} {:?} {} set {}: {}",
                line.index,
                line.decision.verdict,
                line.document,
                line.experience_set,
                line.decision.reason,
            );
        }
        if let Some(p) = &self.proposals {
            let _ = writeln!(
                out,
                "proposing model: {} proposal(s) compiled, {} did not, {} replaced by a mutation",
                p.model_proposals, p.model_failures, p.mutation_fallbacks
            );
        }
        match &self.promoted {
            Some(p) => {
                let _ = writeln!(
                    out,
                    "promoted candidate #{}: champion {} (pointer harness/champion v{})",
                    p.index, p.document, p.version
                );
            }
            None => out.push_str("nothing promoted\n"),
        }
        out
    }
}

impl Report for splinter_sdk::harbor::Discovery {
    fn human(&self) -> String {
        let mut out = format!(
            "{} (commit {}{})\n",
            self.root.display(),
            self.commit.as_deref().unwrap_or("none"),
            if self.dirty == Some(true) {
                ", dirty"
            } else {
                ""
            }
        );
        for task in &self.tasks {
            let _ = match &task.refusal {
                None => writeln!(
                    out,
                    "  importable {} ({})",
                    task.dir.display(),
                    task.name.as_deref().unwrap_or("")
                ),
                Some(why) => writeln!(out, "  refused    {}: {why}", task.dir.display()),
            };
        }
        out
    }
}

impl Report for splinter_sdk::harbor::Imported {
    fn human(&self) -> String {
        let mut out = match &self.task_set {
            Some(set) => format!("imported {} task(s) into set {set}\n", self.imported.len()),
            None => "nothing imported\n".to_string(),
        };
        for origin in &self.imported {
            let _ = writeln!(
                out,
                "  {} -> {} (source digest {})",
                origin.name, origin.task, origin.source.digest
            );
        }
        for refused in &self.refused {
            let _ = writeln!(
                out,
                "  refused {}: {}",
                refused.directory.display(),
                refused.reason
            );
        }
        out
    }
}

impl Report for splinter_sdk::harbor::Exports {
    fn human(&self) -> String {
        let mut out = format!("wrote {} Harbor task(s)\n", self.written.len());
        for task in &self.written {
            let _ = writeln!(out, "  {} -> {}", task.task, task.directory.display());
        }
        out
    }
}

impl Report for splinter_sdk::harbor::Ingested {
    fn human(&self) -> String {
        let mut out = match &self.experience_set {
            Some(set) => format!("ingested into experience set {set}: {:?}\n", self.verdicts),
            None => "nothing ingested\n".to_string(),
        };
        for refused in &self.refused {
            let _ = writeln!(
                out,
                "  refused {}: {}",
                refused.directory.display(),
                refused.reason
            );
        }
        out
    }
}
