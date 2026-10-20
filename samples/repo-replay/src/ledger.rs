// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The ledger of a run: one line per commit attempt or skip, and the bias
//! report computed from it.
//!
//! The ledger is the run's own bookkeeping (what was looked at, what passed,
//! why the rest did not); the experiences it points to are in splinter's
//! store. It makes a rerun resumable and is what the bias report counts, so
//! that a dataset's make-up (which commits are in it, which kinds of commit
//! are missing) is stated and not guessed.

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::attempt::CheckOutcome;
use crate::plan_prompt::PlanStyle;

/// How a ledger line ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    /// The gate passed: a positive.
    Passed,
    /// The attempt ran and the gate refused it: a negative.
    Failed,
    /// No acceptable plan could be written.
    PlanRefused,
    /// The commit was not replayable.
    Skipped,
    /// Infrastructure failed (model unreachable); not counted in any rate.
    Error,
}

/// One line of the ledger.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Entry {
    /// The commit.
    pub commit: String,
    /// The plan style attempted; absent for a skip.
    pub style: Option<PlanStyle>,
    /// How it ended.
    pub outcome: Outcome,
    /// Reason classes: the gate's, the plan's, or the skip's.
    pub classes: Vec<String>,
    /// Lines the commit changes (generated paths excluded).
    pub changed_lines: usize,
    /// Files the commit changes (generated paths excluded).
    pub files: usize,
    /// Tool calls the attempt made.
    pub tool_calls: usize,
    /// Wall seconds of the attempt.
    pub seconds: f64,
    /// The stored experience, when an attempt ran.
    pub experience: Option<String>,
    /// Project checks run after the edits, never gated.
    pub checks: Vec<CheckOutcome>,
    /// Words in the plan.
    pub plan_words: Option<usize>,
    /// Free text: the first reason's detail, or the infrastructure error.
    pub detail: Option<String>,
}

/// The ledger file of a state directory.
#[must_use]
pub fn path(state: &Path) -> std::path::PathBuf {
    state.join("ledger.jsonl")
}

/// Writes one line to the state directory's run log: the tools a run was
/// made with (`git --version`), so a verdict can be traced to the git that
/// produced the commits and diffs it rests on.
pub fn note_run(
    state: &Path,
    git_version: &str,
    source: &Path,
    from: Option<&str>,
    to: Option<&str>,
) -> Result<()> {
    let file = state.join("runs.jsonl");
    let mut out = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file)?;
    let line = serde_json::json!({"git_version": git_version, "source": source.display().to_string(), "from": from, "to": to});
    writeln!(out, "{line}")?;
    Ok(())
}

/// The `(from, to)` range of the last run noted in `state`.
pub fn last_range(state: &Path) -> Result<(Option<String>, Option<String>)> {
    let file = state.join("runs.jsonl");
    let text = fs::read_to_string(&file).with_context(|| format!("reading {}", file.display()))?;
    let last: serde_json::Value =
        serde_json::from_str(text.lines().last().context("runs.jsonl is empty")?)?;
    let get = |k: &str| last[k].as_str().map(String::from);
    Ok((get("from"), get("to")))
}

/// Every entry written so far, in order.
pub fn read(state: &Path) -> Result<Vec<Entry>> {
    let file = path(state);
    if !file.exists() {
        return Ok(Vec::new());
    }
    fs::read_to_string(&file)
        .with_context(|| format!("reading {}", file.display()))?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).with_context(|| format!("unreadable ledger line {l:?}")))
        .collect()
}

/// Appends `entry`.
pub fn append(state: &Path, entry: &Entry) -> Result<()> {
    let file = path(state);
    let mut out = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file)?;
    writeln!(out, "{}", serde_json::to_string(entry)?)?;
    Ok(())
}

/// The size bucket of a commit, by changed lines.
#[must_use]
pub fn bucket(changed_lines: usize) -> &'static str {
    match changed_lines {
        0..=10 => "1-10 lines",
        11..=50 => "11-50 lines",
        51..=200 => "51-200 lines",
        _ => "over 200 lines",
    }
}

/// Counts of one group of attempts.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Counts {
    /// Attempts that ran to a verdict or were refused a plan.
    pub attempts: usize,
    /// Of those, passed.
    pub passed: usize,
}

/// What a run's dataset is made of and what it lacks.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Bias {
    /// Attempts and passes over everything.
    pub overall: Counts,
    /// Tool calls over the attempts that ran to a verdict, and how many ran.
    pub tool_calls: usize,
    /// Attempts that ran to a verdict (the denominator of `tool_calls`).
    pub verdicts: usize,
    /// By size bucket.
    pub by_size: BTreeMap<String, Counts>,
    /// By plan style.
    pub by_style: BTreeMap<String, Counts>,
    /// Failed attempts and refused plans by reason class (an attempt can have several).
    pub reasons: BTreeMap<String, usize>,
    /// Skipped commits by class.
    pub skipped: BTreeMap<String, usize>,
    /// Infrastructure errors (excluded from every rate).
    pub errors: usize,
}

/// The bias report of `entries`.
#[must_use]
pub fn bias(entries: &[Entry]) -> Bias {
    let mut report = Bias::default();
    for e in entries {
        match e.outcome {
            Outcome::Skipped => {
                for class in &e.classes {
                    *report.skipped.entry(class.clone()).or_default() += 1;
                }
            }
            Outcome::Error => report.errors += 1,
            Outcome::Passed | Outcome::Failed | Outcome::PlanRefused => {
                let style = e.style.map_or("-", PlanStyle::name);
                for group in [
                    report
                        .by_size
                        .entry(bucket(e.changed_lines).to_string())
                        .or_default(),
                    report.by_style.entry(style.to_string()).or_default(),
                ] {
                    group.attempts += 1;
                    group.passed += usize::from(e.outcome == Outcome::Passed);
                }
                report.overall.attempts += 1;
                report.overall.passed += usize::from(e.outcome == Outcome::Passed);
                if e.outcome != Outcome::PlanRefused {
                    report.tool_calls += e.tool_calls;
                    report.verdicts += 1;
                }
                if e.outcome != Outcome::Passed {
                    for class in &e.classes {
                        *report.reasons.entry(class.clone()).or_default() += 1;
                    }
                }
            }
        }
    }
    report
}

impl std::fmt::Display for Bias {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let rate = |c: &Counts| format!("{}/{}", c.passed, c.attempts);
        writeln!(f, "pass rate overall: {}", rate(&self.overall))?;
        if self.verdicts > 0 {
            writeln!(
                f,
                "tool calls per attempt that ran: {:.2} over {}",
                self.tool_calls as f64 / self.verdicts as f64,
                self.verdicts
            )?;
        }
        writeln!(f, "pass rate by commit size:")?;
        for (k, c) in &self.by_size {
            writeln!(f, "  {k:<16} {}", rate(c))?;
        }
        writeln!(f, "pass rate by plan style:")?;
        for (k, c) in &self.by_style {
            writeln!(f, "  {k:<16} {}", rate(c))?;
        }
        writeln!(f, "reasons of failed attempts:")?;
        for (k, n) in &self.reasons {
            writeln!(f, "  {k:<28} {n}")?;
        }
        writeln!(f, "commits skipped:")?;
        for (k, n) in &self.skipped {
            writeln!(f, "  {k:<28} {n}")?;
        }
        writeln!(f, "infrastructure errors (in no rate): {}", self.errors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(outcome: Outcome, lines: usize, style: Option<PlanStyle>, classes: &[&str]) -> Entry {
        Entry {
            commit: "c".into(),
            style,
            outcome,
            classes: classes.iter().map(|c| (*c).into()).collect(),
            changed_lines: lines,
            files: 1,
            tool_calls: 1,
            seconds: 1.0,
            experience: None,
            checks: vec![],
            plan_words: None,
            detail: None,
        }
    }

    #[test]
    fn the_report_counts_by_size_style_and_reason_and_leaves_errors_out_of_every_rate() {
        let terse = Some(PlanStyle::Terse);
        let entries = [
            entry(Outcome::Passed, 5, terse, &[]),
            entry(Outcome::Failed, 5, terse, &["tree_mismatch", "unread_edit"]),
            entry(Outcome::Failed, 120, Some(PlanStyle::Detailed), &["junk"]),
            entry(Outcome::Skipped, 0, None, &["merge"]),
            entry(Outcome::Error, 5, terse, &[]),
        ];
        let report = bias(&entries);
        assert_eq!(
            report.by_size["1-10 lines"],
            Counts {
                attempts: 2,
                passed: 1
            }
        );
        assert_eq!(
            report.by_size["51-200 lines"],
            Counts {
                attempts: 1,
                passed: 0
            }
        );
        assert_eq!(
            report.by_style["terse"],
            Counts {
                attempts: 2,
                passed: 1
            }
        );
        assert_eq!(report.reasons["unread_edit"], 1);
        assert_eq!(report.skipped["merge"], 1);
        assert_eq!(report.errors, 1);
    }
}
