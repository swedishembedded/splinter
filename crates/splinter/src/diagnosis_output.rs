// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements causal diagnosis of why an agent failed, from
// the evidence its run left, for its clients. If your team needs expertise in
// telling model failures from infrastructure failures before you train on
// them, you can procure our services by sending an email to
// info@swedishembedded.com.

//! How `experiences diagnose` and `experiences repair` read to a person.

use std::fmt::Write as _;

use splinter_sdk::bench::diagnose::DiagnosisReport;
use splinter_sdk::bench::repair::RepairReport;
use splinter_sdk::set_import::SetsImported;

use crate::output::Report;

impl Report for DiagnosisReport {
    fn human(&self) -> String {
        let mut out = String::new();
        let failed = self.trials - self.passed;
        let _ = writeln!(
            out,
            "{} trials, {} passed, {failed} failed (diagnosis rules {})",
            self.trials, self.passed, self.rules
        );
        for (attribution, count) in &self.by_attribution {
            let _ = writeln!(out, "  {attribution}: {count}");
        }
        for (domain, classes) in &self.by_domain {
            let line: Vec<String> = classes.iter().map(|(c, n)| format!("{c} {n}")).collect();
            let _ = writeln!(out, "{domain}: {}", line.join(", "));
        }
        for attempt in &self.attempts {
            let Some(result) = &attempt.diagnosis else {
                continue;
            };
            let step = result
                .decisive_step
                .map_or_else(String::new, |s| format!(" at step {s}"));
            let _ = writeln!(
                out,
                "  {} {}: {}{step} ({})",
                attempt.domain,
                attempt.trial,
                result.class.as_str(),
                result.rule
            );
        }
        for refused in &self.refused {
            let trial = refused.trial.as_deref().unwrap_or("-");
            let _ = writeln!(
                out,
                "refused {} {trial}: {}",
                refused.export.display(),
                refused.reason
            );
        }
        out
    }
}

impl Report for RepairReport {
    fn human(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "repairs continued by {}", self.teacher);
        for (class, y) in &self.by_class {
            let _ = writeln!(
                out,
                "  {class}: failed {}, routed {}, accepted {}, rejected {}, refused {}, held {}",
                y.failed, y.routed, y.accepted, y.rejected, y.refused, y.held
            );
        }
        for record in &self.repairs {
            let why = record.reason.as_deref().unwrap_or("");
            let _ = writeln!(
                out,
                "  {} {} after step {}: {:?} {why}",
                record.domain, record.trial, record.keep_through, record.outcome
            );
        }
        if let Some(set) = &self.repair_set {
            let _ = writeln!(out, "accepted repairs: experience set {set}");
        }
        out
    }
}

impl Report for SetsImported {
    fn human(&self) -> String {
        let mut out = String::new();
        for set in &self.sets {
            let _ = writeln!(out, "imported experience set {set}");
        }
        let _ = writeln!(
            out,
            "experiences: {} new, {} held ({} named by relations); annotations: {} new; trial records: {} new",
            self.experiences_new, self.experiences_held, self.related, self.annotations_new, self.trial_records_new
        );
        out
    }
}
