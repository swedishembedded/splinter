// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements causal diagnosis of why an agent failed, from
// the evidence its run left, for its clients. If your team needs expertise in
// telling model failures from infrastructure failures before you train on
// them, you can procure our services by sending an email to
// info@swedishembedded.com.

//! How `experiences diagnose` reads to a person.

use std::fmt::Write as _;

use splinter_sdk::bench::diagnose::DiagnosisReport;

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
