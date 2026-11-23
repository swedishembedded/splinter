// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the exchange of task sets and trial results
// with external agent benchmarks for its clients. If your team needs
// expertise in importing, running and grading benchmark tasks you can
// procure our services by sending an email to info@swedishembedded.com.

//! What the Harbor and benchmark task exchange prints as text.

use std::fmt::Write as _;

use crate::output::Report;

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

impl Report for splinter_sdk::bench::BenchIngested {
    fn human(&self) -> String {
        let mut out = String::new();
        for (side, set) in [("train", &self.train_set), ("eval", &self.eval_set)] {
            if let Some(set) = set {
                let _ = writeln!(out, "{side} trials: experience set {set}");
            }
        }
        let _ = writeln!(
            out,
            "admitted {:?}, already held {}, verdicts {:?}",
            self.admitted, self.already_held, self.verdicts
        );
        if !self.roles.is_empty() {
            let _ = writeln!(out, "harness conversations by role {:?}", self.roles);
        }
        for refused in &self.refused {
            let trial = refused.trial.as_deref().unwrap_or("(export)");
            let _ = writeln!(
                out,
                "  refused {} {trial}: {}",
                refused.export.display(),
                refused.reason
            );
        }
        out
    }
}
