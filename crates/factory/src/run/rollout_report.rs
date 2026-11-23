// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The report of a rollout pass or evaluation arm: its trials by status,
//! the sandboxes they ran in, and the tasks left unsolved.

use std::collections::{BTreeMap, BTreeSet};

use splinter_core::factory::product::TrialResult;

use super::rollout::AttemptEntry;

/// The pass's report for a person.
pub(crate) fn report(name: &str, trials: &[TrialResult], attempts: &[AttemptEntry]) -> String {
    let mut by_status: BTreeMap<String, usize> = BTreeMap::new();
    for trial in trials {
        let status = serde_json::to_value(trial.grade.status)
            .ok()
            .and_then(|v| v.as_str().map(str::to_owned));
        *by_status.entry(status.unwrap_or_default()).or_default() += 1;
    }
    let solved: BTreeSet<&str> = trials
        .iter()
        .filter(|t| t.verified_success())
        .map(|t| t.task.as_str())
        .collect();
    let tasks: BTreeSet<&str> = trials.iter().map(|t| t.task.as_str()).collect();
    let passed = trials.iter().filter(|t| t.verified_success()).count();
    let counted = trials.iter().filter(|t| t.grade.counted).count();
    let mut out = format!(
        "# Rollout {name}\n\n| | |\n|---|---|\n| trials | {} |\n| physical attempts | {} |\n| counted | {counted} |\n\
         | verified successes | {passed} |\n| tasks solved | {} of {} |\n\n## Trials by status\n\n| status | trials |\n|---|---|\n",
        trials.len(),
        attempts.len(),
        solved.len(),
        tasks.len()
    );
    for (status, n) in by_status {
        out.push_str(&format!("| {status} | {n} |\n"));
    }
    // Which sandbox the trials ran in, so a weaker one than expected is seen.
    let mut sandboxes: BTreeMap<String, usize> = BTreeMap::new();
    for trial in trials {
        let key = trial.sandbox.as_ref().map_or_else(
            || "not reported".to_owned(),
            |s| format!("{} ({})", s.provider, s.guarantees.join(", ")),
        );
        *sandboxes.entry(key).or_default() += 1;
    }
    out.push_str("\n## Sandboxes\n\n| sandbox (what it enforced) | trials |\n|---|---|\n");
    for (sandbox, n) in sandboxes {
        out.push_str(&format!("| {sandbox} | {n} |\n"));
    }
    let unsolved: Vec<String> = tasks
        .difference(&solved)
        .map(|t| format!("- {t}"))
        .collect();
    if !unsolved.is_empty() {
        out.push_str(&format!("\n## Unsolved tasks\n\n{}\n", unsolved.join("\n")));
    }
    out
}
