// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements causal diagnosis of why an agent failed, from
// the evidence its run left, for its clients. If your team needs expertise in
// telling model failures from infrastructure failures before you train on
// them, you can procure our services by sending an email to
// info@swedishembedded.com.

//! How `experiences diagnose`, `experiences capabilities` and `experiences
//! repair` read to a person.

use std::fmt::Write as _;

use splinter_sdk::bench::capabilities::StoredCapabilities;
use splinter_sdk::bench::diagnose::DiagnosisReport;
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

impl Report for StoredCapabilities {
    fn human(&self) -> String {
        let mut out = String::new();
        let report = &self.report;
        let l = &report.labelled;
        let _ = writeln!(
            out,
            "kept as {} (labels: {} from exports, {} from the material, {} by source digest, {} none)",
            self.document, l.export, l.material, l.labels, l.none
        );
        for policy in &report.policies {
            let _ = writeln!(out, "{}: {} trials", policy.policy, policy.trials);
            let _ = writeln!(
                out,
                "  {:<28} {:>5} {:>6} {:>6}  {:<21} {:>5} {:>13}  {:>6}  failures",
                "capability",
                "tasks",
                "graded",
                "passed",
                "rate [95% interval]",
                "infra",
                "F/C/R/U",
                "weight"
            );
            for c in &policy.capabilities {
                let rate = c.pass_rate.map_or_else(
                    || "-".to_string(),
                    |s| format!("{:.3} [{:.3}, {:.3}]", s.value, s.low, s.high),
                );
                let bands = format!(
                    "{}/{}/{}/{}",
                    c.bands.frontier, c.bands.cliff, c.bands.retired, c.bands.unmeasured
                );
                let failures: Vec<String> = c
                    .failures
                    .iter()
                    .map(|(class, n)| format!("{class} {n}"))
                    .collect();
                let _ = writeln!(
                    out,
                    "  {:<28} {:>5} {:>6} {:>6}  {:<21} {:>5} {:>13}  {:>6.3}  {}",
                    c.capability,
                    c.tasks,
                    c.graded,
                    c.passes,
                    rate,
                    c.ungraded,
                    bands,
                    c.weight,
                    failures.join(", ")
                );
            }
            if let Some(signal) = &policy.signal {
                crate::dataset_output::signal_lines(&mut out, "strict reward", &signal.verdict);
                crate::dataset_output::signal_lines(
                    &mut out,
                    "shaped-partial",
                    &signal.shaped_partial,
                );
                let _ = writeln!(
                    out,
                    "  shaped-partial: {} task(s) with a naive floor, {} failure(s) without \
                     functional_correctness left out",
                    signal.floored_tasks, signal.unmeasured_failures
                );
            }
        }
        for refused in &report.refused {
            let trial = refused.trial.as_deref().unwrap_or("-");
            let _ = writeln!(
                out,
                "refused {} {trial}: {}",
                refused.dir.display(),
                refused.reason
            );
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
