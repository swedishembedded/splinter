// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What the commands print: an attempt's outcome, and the run index.

use splinter_agent::outcome::{Outcome, Status};
use splinter_store::runs::{list_runs, read_manifest};
use splinter_store::trace::read_events;
use splinter_store::StateRoot;

/// Prints an attempt's outcome and exits non-zero unless it completed - in
/// both output modes: a script that delegated work must not read a timeout
/// as success, and a JSON consumer checks the exit code, not the prose.
pub fn attempt(outcome: &Outcome, json: bool) -> anyhow::Result<()> {
    if json {
        println!("{}", serde_json::to_string_pretty(outcome)?);
    } else {
        attempt_human(outcome);
    }
    // Non-zero exit on a non-completed attempt - in BOTH output modes. A
    // script that delegated work must not read a timeout as success, and a
    // JSON consumer checks the exit code, not the prose.
    if outcome.status != Status::Completed {
        std::process::exit(1);
    }
    Ok(())
}

fn attempt_human(outcome: &Outcome) {
    println!("run:     {}", outcome.run_id);
    println!("status:  {}", outcome.status.as_str());
    if !outcome.changed_files.is_empty() {
        println!(
            "changed: {} file(s) ({})",
            outcome.changed_files.len(),
            outcome.changed_files_basis
        );
        for file in outcome.changed_files.iter().take(10) {
            println!("  {} ({})", file.path, file.kind);
        }
    }
    for check in &outcome.checks {
        println!(
            "check:   {} -> {} ({})",
            check.command,
            check.exit,
            if check.passed { "pass" } else { "FAIL" }
        );
    }
    println!(
        "usage:   {} tool call(s), {} failed, {} in / {} out tok{}",
        outcome.usage.tool_calls,
        outcome.usage.failed_tool_calls,
        outcome.usage.input_tokens,
        outcome.usage.output_tokens,
        outcome
            .usage
            .cost_usd
            .map(|c| format!(", ${c:.4}"))
            .unwrap_or_else(|| ", cost unmeasured".into()),
    );
    if !outcome.tool_failures.is_empty() {
        println!("failures:");
        for failure in outcome.tool_failures.iter().take(10) {
            println!("  {failure}");
        }
    }
    if !outcome.unresolved.is_empty() {
        println!("unresolved:");
        for issue in &outcome.unresolved {
            println!("  {issue}");
        }
    }
}

/// Prints one run's manifest, outcome and trace index, or - with no run
/// named - every run under the state root.
pub fn show(root: &StateRoot, run: Option<&str>) -> anyhow::Result<()> {
    if let Some(run_id) = run {
        let manifest = read_manifest(root, run_id)?;
        println!(
            "run:      {}\nstatus:   {}\nattempts: {}\nmodel:    {}\nstarted:  {}",
            manifest.run_id,
            manifest.status,
            manifest.attempts,
            manifest.model,
            manifest.started_ts,
        );
        println!("workspace: {}", manifest.workspace);
        println!("task: {}", manifest.task);
        let dir = root.run_dir(run_id);
        match Outcome::load(&dir) {
            Ok(outcome) => {
                println!(
                    "outcome:  {} ({} check(s), {} changed file(s))",
                    outcome.status.as_str(),
                    outcome.checks.len(),
                    outcome.changed_files.len()
                );
            }
            Err(e) => println!("outcome:  not written yet ({e})"),
        }
        let events = read_events(&dir)?;
        println!("trace:    {} event(s); last:", events.len());
        if let Some(last) = events.last() {
            println!(
                "  [{} {} {}]",
                last.get("ts").and_then(|v| v.as_str()).unwrap_or("?"),
                last.get("seq").and_then(|v| v.as_u64()).unwrap_or(0),
                last.get("type").and_then(|v| v.as_str()).unwrap_or("?")
            );
        }
        return Ok(());
    }
    let runs = list_runs(root)?;
    if runs.is_empty() {
        println!("no runs recorded under {}", root.path().display());
        return Ok(());
    }
    for manifest in runs {
        println!(
            "{}  {:<10}  {}  {}",
            manifest.run_id,
            manifest.status,
            manifest.model,
            manifest.task.chars().take(60).collect::<String>()
        );
    }
    Ok(())
}

/// Prints a document-learning pipeline's two-line verdict: what was
/// learned, and how well - recall on the trained questions, generalization
/// on the held-out ones. A score over zero questions is printed as not
/// measured, never as 0.
pub fn facts(report: &splinter_campaign::facts::FactsReport, source: &str) {
    println!(
        "facts: {} fact(s) from {source} ({} train / {} eval), training {}",
        report.facts,
        report.train_records,
        report.eval_records,
        if report.promoted {
            format!("promoted ({})", report.train_id)
        } else {
            "REJECTED by the held-out gate".into()
        },
    );
    println!(
        "facts: recall {}, holdout {}",
        ratio(report.recall_correct, report.recall_total),
        ratio(report.holdout_correct, report.holdout_total),
    );
}

fn ratio(correct: usize, total: usize) -> String {
    if total == 0 {
        "not measured (0 questions)".into()
    } else {
        format!("{correct}/{total} = {:.3}", correct as f64 / total as f64)
    }
}
