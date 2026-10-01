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
//! and `stdout` may be left out. Anything else - a missing or mistyped
//! field, a field the shape does not name - makes the whole reply
//! malformed. The generator asks for it as a typed call ([`Reply`]'s JSON
//! schema is the shape the model is shown), so sven parses the reply and
//! sends a malformed one back for correction.

use serde::Deserialize;
use splinter_lab::verifiers::executable::{ExecutableCheck, Expectation, ExpectedStdout};
use splinter_lab::verifiers::normalise::Normalisation;
use splinter_store::experience::Environment;
use sven_sdk::schemars::JsonSchema;

/// One reply of the shape [`Reply`] parses, shown to the model beside the
/// schema: a small model copies an example far more reliably than it reads
/// a schema's references.
pub(crate) const REPLY_EXAMPLE: &str = r#"{"tasks": [{"instruction": "At what baud rate does the console UART run?", "reference": "115200 baud", "evidence": [{"section": 0}], "material": null}]}"#;

/// A generator model's reply.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(crate = "sven_sdk::schemars")]
pub(crate) struct Reply {
    pub(crate) tasks: Vec<Candidate>,
}

/// One task as the model wrote it, before anything about it is trusted.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(crate = "sven_sdk::schemars")]
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
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(crate = "sven_sdk::schemars")]
pub(crate) struct Citation {
    /// The section's position in the list the model was shown.
    pub(crate) section: usize,
    /// Verbatim text of that section; the whole section when absent.
    #[serde(default)]
    pub(crate) quote: Option<String>,
}

/// A check or test as the model wrote it.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(crate = "sven_sdk::schemars")]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_example_shown_to_the_model_is_a_reply() {
        let example: Result<Reply, _> = serde_json::from_str(REPLY_EXAMPLE);
        assert!(example.is_ok_and(|r| r.tasks.len() == 1));
    }
}
