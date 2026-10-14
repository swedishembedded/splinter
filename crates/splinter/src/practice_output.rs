// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that learn a codebase from its own
// tests, for its clients. If your team needs expertise in training
// code-editing agents on executable tasks drawn from a repository, you can
// procure our services by sending an email to info@swedishembedded.com.

//! What `practice` prints as text: each stage's line as it finishes, and the
//! whole run's report.

use std::fmt::Write as _;

use splinter_sdk::practice::PracticeReport;

use crate::output::Report;

/// One line saying what a finished `practice` stage did, from its summary;
/// `None` for a stage whose summary is a `learn` stage's.
pub fn stage_line(stage: &str, summary: &serde_json::Value) -> Option<String> {
    let n = |value: &serde_json::Value| value.as_u64().unwrap_or(0);
    match stage {
        "snapshot" if summary["commit"].is_string() => Some(format!(
            "commit {}, tree {}{}",
            summary["commit"].as_str().unwrap_or("?"),
            summary["tree_id"].as_str().unwrap_or("?"),
            if summary["dirty"] == true {
                " (uncommitted changes are not in it)"
            } else {
                ""
            }
        )),
        "profile" => Some(format!(
            "`{}`: {} test(s) pass twice, {} file(s) under test",
            summary["test_command"]
                .as_array()
                .map(|words| words
                    .iter()
                    .filter_map(|w| w.as_str())
                    .collect::<Vec<_>>()
                    .join(" "))
                .unwrap_or_default(),
            n(&summary["tests"]),
            n(&summary["files_under_test"])
        )),
        "tasks" if summary["admitted"].is_u64() => Some(format!(
            "{} task(s) admitted, {} already held, {} fault(s) refused",
            n(&summary["admitted"]),
            n(&summary["held_before"]),
            summary["refused"]
                .as_object()
                .map_or(0, |reasons| reasons.values().map(n).sum::<u64>())
        )),
        "reserve" if summary["exam_tasks"].is_u64() => Some(format!(
            "{} exam family(ies) set apart ({} task(s)), {} task(s) in {} family(ies) to train on",
            summary["families"].as_array().map_or(0, Vec::len),
            n(&summary["exam_tasks"]),
            n(&summary["training_tasks"]),
            n(&summary["training_families"])
        )),
        "verify" if summary["exposed"].is_u64() => Some(format!(
            "{} passed, {} failed, {} undecided; {} exposed to what only the grader holds",
            n(&summary["passed"]),
            n(&summary["failed"]),
            n(&summary["undecided"]),
            n(&summary["exposed"])
        )),
        _ => None,
    }
}

/// The line for a stage whose summary is this pipeline's own, from the
/// report the summary is made from.
fn own(stage: &str, report: &impl serde::Serialize) -> String {
    serde_json::to_value(report)
        .ok()
        .and_then(|summary| stage_line(stage, &summary))
        .unwrap_or_default()
}

impl Report for PracticeReport {
    fn human(&self) -> String {
        let mut out = String::new();
        let stage = |out: &mut String, name: &str, body: String| {
            for (i, line) in body.lines().enumerate() {
                let label = if i == 0 { name } else { "" };
                let _ = writeln!(out, "{label:<9}{line}");
            }
        };
        let policy = match &self.policy.release {
            Some(release) => format!("policy:{} = release {release}", self.policy.alias),
            None => format!("policy:{} = the base: no release yet", self.policy.alias),
        };
        stage(&mut out, "policy", policy);
        if let Some(r) = &self.snapshot {
            stage(&mut out, "snapshot", own("snapshot", r));
        }
        if let Some(r) = &self.profile {
            stage(&mut out, "profile", own("profile", r));
        }
        if let Some(r) = &self.tasks {
            stage(&mut out, "tasks", own("tasks", r));
        }
        if let Some(r) = &self.reserve {
            stage(&mut out, "reserve", own("reserve", r));
        }
        if let Some(r) = &self.solve {
            stage(&mut out, "solve", r.human());
        }
        if let Some(r) = &self.verify {
            stage(&mut out, "verify", r.human());
        }
        if let Some(r) = &self.frontier {
            stage(&mut out, "frontier", r.human());
        }
        if let Some(r) = &self.dataset {
            stage(&mut out, "dataset", r.human());
        }
        if let Some(r) = &self.candidate {
            stage(&mut out, "train", r.human());
        }
        if let Some(r) = &self.release {
            stage(&mut out, "release", r.human());
        }
        if let Some(why) = &self.stopped {
            let _ = writeln!(out, "stopped  {why}");
        }
        out
    }
}
