// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The one shape a generator model's reply may take, parsed strictly.
//!
//! ```json
//! {"tasks": [{
//!   "instruction": "...",
//!   "reference": "...",
//!   "evidence": [{"section": 0, "quote": "verbatim text of section 0"}],
//!   "material": "...",
//!   "hints": ["..."],
//!   "checks": [{"code": "...", "stdin": null, "exit_code": 0, "stdout": null}],
//!   "tests": [{"code": "...", "exit_code": 0}]
//! }]}
//! ```
//!
//! `quote`, `material`, `hints`, `checks`, `tests`, `stdin`, `exit_code`
//! and `stdout` may be left out. Anything else - prose around the object,
//! a missing or mistyped field, a field the shape does not name - makes
//! the whole reply malformed; one surrounding code fence is tolerated.

use serde::Deserialize;
use splinter_lab::verifiers::executable::{ExecutableCheck, Expectation, ExpectedStdout};
use splinter_lab::verifiers::normalise::Normalisation;
use splinter_store::experience::Environment;

/// A generator model's reply.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Reply {
    pub(crate) tasks: Vec<Candidate>,
}

/// One task as the model wrote it, before anything about it is trusted.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Candidate {
    pub(crate) instruction: String,
    pub(crate) reference: String,
    pub(crate) evidence: Vec<Citation>,
    #[serde(default)]
    pub(crate) material: Option<String>,
    #[serde(default)]
    pub(crate) hints: Vec<String>,
    #[serde(default)]
    pub(crate) checks: Vec<Check>,
    #[serde(default)]
    pub(crate) tests: Vec<Check>,
}

/// Where in the sections a candidate says its answer comes from.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Citation {
    /// The section's position in the list the model was shown.
    pub(crate) section: usize,
    /// Verbatim text of that section; the whole section when absent.
    #[serde(default)]
    pub(crate) quote: Option<String>,
}

/// A check or test as the model wrote it.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Check {
    code: String,
    #[serde(default)]
    stdin: Option<String>,
    #[serde(default)]
    exit_code: Option<i32>,
    #[serde(default)]
    stdout: Option<String>,
}

impl Check {
    /// The lab's check, run in `environment` (the task's own when `None`),
    /// its output compared with whitespace normalised; `None` when it
    /// expects nothing, since such a check passes everything.
    pub(crate) fn to_check(&self, environment: Option<Environment>) -> Option<ExecutableCheck> {
        if self.exit_code.is_none() && self.stdout.is_none() {
            return None;
        }
        Some(ExecutableCheck {
            code: self.code.clone(),
            stdin: self.stdin.clone(),
            expect: Expectation {
                exit_code: self.exit_code,
                stdout: self.stdout.as_ref().map(|text| ExpectedStdout {
                    text: text.clone(),
                    normalisation: Normalisation::WHITESPACE,
                }),
            },
            environment,
        })
    }
}

/// `reply` parsed strictly; the error says why it is not the shape.
pub(crate) fn parse(reply: &str) -> Result<Reply, String> {
    serde_json::from_str(strip_fences(reply)).map_err(|e| e.to_string())
}

/// `reply` without optional markdown code fences around it, so a model that
/// answered perfectly inside ```json fences still parses. Fences are the
/// one tolerated decoration; prose around the object is not.
fn strip_fences(reply: &str) -> &str {
    let trimmed = reply.trim();
    let without = trimmed
        .strip_prefix("```")
        .and_then(|r| {
            r.trim_start_matches(|c: char| c.is_ascii_alphanumeric())
                .strip_prefix('\n')
        })
        .unwrap_or(trimmed);
    without
        .strip_suffix("```")
        .map(|r| r.trim())
        .unwrap_or(without)
}

/// The shape, as the model is told it.
pub(crate) const SHAPE: &str = r#"{"tasks": [{"instruction": string, "reference": string, "evidence": [{"section": number, "quote": string or null}], "material": string or null, "hints": [string], "checks": [{"code": string, "stdin": string or null, "exit_code": number or null, "stdout": string or null}], "tests": [same as checks]}]}"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// Fences around the object are tolerated; prose around it is not.
    #[test]
    fn a_fenced_reply_parses_and_prose_does_not() {
        let fenced = "```json\n{\"tasks\": []}\n```";
        assert!(parse(fenced).is_ok_and(|reply| reply.tasks.is_empty()));
        assert!(parse("Here you go: {\"tasks\": []}").is_err());
    }
}
