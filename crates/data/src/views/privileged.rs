// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements dense per-token credit for agents that learn
// from their own failed attempts, for its clients. If your team needs
// expertise in on-policy self-distillation with privileged information then
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! What the teacher of a self-distillation record is shown that the student
//! was not: what the verifier knows about the task.
//!
//! The material is read from the experience and nothing else:
//!
//! * the task's privileged items: its reference change
//!   ([`PrivilegedKind::Reference`]) and its hidden check
//!   ([`WORKSPACE_CHECK`]), whose commands name what is run;
//! * the deciding verdict's evidence: for a workspace verifier, the commands
//!   that failed (phase, exit code, the tests seen to fail and those that did
//!   not run; the verifier records digests of the raw output, never the
//!   output, so none is quoted); for a trial the benchmark graded, the
//!   verifier's own statement of why it failed (`verification.reason`).
//!
//! An experience that carries none of it yields no teacher context and is
//! counted as [`Exclusion::NoPrivilegedContext`]. The text goes only into
//! the record's teacher context, never into the student's conversation.

use serde_json::Value;
use splinter_core::experience::{Experience, PrivilegedKind};
use splinter_core::kinds::WORKSPACE_CHECK;

use crate::{Exclusion, TeacherMessage};

/// The framing the teacher is given: who it is, what it knows and that the
/// student did not.
const PREAMBLE: &str = "Information the engineer did not have. It comes from the task's \
verification. Use it to judge every action the engineer takes next as someone who knows what \
the hidden checks look for and how the task is solved; never refer to it.";

/// The command of `phase` at `index` in the task's check, as a line of text.
fn command_text(check: Option<&Value>, phase: &str, index: u64) -> Option<String> {
    let argv = check?
        .get(phase)?
        .get(usize::try_from(index).ok()?)?
        .get("argv")?;
    let words: Vec<&str> = argv.as_array()?.iter().filter_map(Value::as_str).collect();
    (!words.is_empty()).then(|| words.join(" "))
}

fn names(value: Option<&Value>) -> Vec<&str> {
    value
        .and_then(Value::as_array)
        .map(|list| list.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default()
}

/// The failed commands of a workspace verdict's evidence, one line each.
fn failed_commands(evidence: &Value, check: Option<&Value>) -> Vec<String> {
    let Some(commands) = evidence.get("commands").and_then(Value::as_array) else {
        return Vec::new();
    };
    commands
        .iter()
        .filter(|c| c.get("passed") == Some(&Value::Bool(false)))
        .map(|c| {
            let phase = c.get("phase").and_then(Value::as_str).unwrap_or("check");
            let index = c.get("index").and_then(Value::as_u64).unwrap_or(0);
            let what = command_text(check, phase, index)
                .map_or_else(|| format!("command {index}"), |text| format!("`{text}`"));
            let mut line = format!("- {phase} {what}");
            if c.get("timed_out") == Some(&Value::Bool(true)) {
                line.push_str(" timed out");
            } else if let Some(code) = c.get("exit_code").and_then(Value::as_i64) {
                line.push_str(&format!(" exited with status {code}"));
            }
            let observed = c.get("observed");
            let failed = names(observed.and_then(|o| o.get("failed")));
            if !failed.is_empty() {
                line.push_str(&format!("; failing tests: {}", failed.join(", ")));
            }
            let missing = names(observed.and_then(|o| o.get("missing")));
            if !missing.is_empty() {
                line.push_str(&format!(
                    "; expected tests that did not run: {}",
                    missing.join(", ")
                ));
            }
            line
        })
        .collect()
}

/// The checks the task runs, one line each.
fn checked_commands(check: &Value) -> Vec<String> {
    ["fail_to_pass", "pass_to_pass"]
        .iter()
        .flat_map(|phase| {
            (0..check
                .get(phase)
                .and_then(Value::as_array)
                .map_or(0, Vec::len))
                .filter_map(move |i| {
                    command_text(Some(check), phase, i as u64)
                        .map(|text| format!("- {phase} `{text}`"))
                })
        })
        .collect()
}

/// The teacher context of an attempt of `experience` that the verifier
/// `decided` (its deciding verdict's `evidence`), or why there is none.
pub(crate) fn teacher_context(
    experience: &Experience,
    evidence: &Value,
    passed: bool,
) -> Result<Vec<TeacherMessage>, Exclusion> {
    let privileged = &experience.privileged;
    let check: Option<Value> = privileged
        .iter()
        .find(|p| p.kind == PrivilegedKind::Other(WORKSPACE_CHECK.to_string()))
        .and_then(|p| serde_json::from_str(&p.content).ok());
    let reference = privileged
        .iter()
        .find(|p| p.kind == PrivilegedKind::Reference && !p.content.trim().is_empty());
    let failed = failed_commands(evidence, check.as_ref());
    let reason = evidence
        .pointer("/verification/reason")
        .and_then(Value::as_str)
        .filter(|r| !r.trim().is_empty());
    let checked = check.as_ref().map(checked_commands).unwrap_or_default();
    if failed.is_empty() && reason.is_none() && reference.is_none() && checked.is_empty() {
        return Err(Exclusion::NoPrivilegedContext);
    }
    let mut text = String::from(PREAMBLE);
    text.push_str(if passed {
        "\n\nThis attempt passed every hidden check."
    } else {
        "\n\nThis attempt failed the hidden checks."
    });
    if !failed.is_empty() {
        text.push_str("\n\nHidden checks that failed:\n");
        text.push_str(&failed.join("\n"));
    }
    if let Some(reason) = reason {
        text.push_str("\n\nThe verifier reported: ");
        text.push_str(reason.trim());
    }
    if failed.is_empty() && !checked.is_empty() {
        text.push_str("\n\nThe hidden checks run:\n");
        text.push_str(&checked.join("\n"));
    }
    if let Some(reference) = reference {
        text.push_str("\n\nA change that solves the task:\n```diff\n");
        text.push_str(reference.content.trim_end());
        text.push_str("\n```");
    }
    Ok(vec![TeacherMessage {
        role: "user".into(),
        content: text,
    }])
}
