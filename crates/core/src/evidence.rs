// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Reading the evidence an executable verdict carries.
//!
//! An executable verifier records, with each verdict, one entry per check it
//! ran: the runtime it ran in, how the run ended and whether the check met
//! its expectation. These functions read that shape back for whoever shows a
//! verdict to a reader - a critic, a training view - without running
//! anything.

/// The producer name an executable verifier's annotations carry.
pub const EXECUTABLE_PRODUCER: &str = "splinter-lab/executable";

/// A reader's summary of the evidence an executable verdict carries: one
/// line per check run, naming the runtime it ran in and how the run ended
/// (`check 1: python 3.12.1, exit code 1`). Digests are left out: they
/// identify what ran, they do not describe it. `None` when `evidence` is
/// not in the shape an executable verifier records, or no check ran.
#[must_use]
pub fn evidence_summary(evidence: &serde_json::Value) -> Option<String> {
    let checks = evidence.get("checks")?.as_array()?;
    let mut lines = Vec::with_capacity(checks.len());
    for (index, check) in checks.iter().enumerate() {
        let runtime = check.get("runtime")?;
        let name = runtime.get("name")?.as_str()?;
        let version = runtime.get("version")?.as_str()?;
        let ending = if check.get("timed_out")?.as_bool()? {
            "timed out".to_string()
        } else if let Some(code) = check.get("exit_code").and_then(serde_json::Value::as_i64) {
            format!("exit code {code}")
        } else {
            let signal = check.get("signal").and_then(serde_json::Value::as_i64)?;
            format!("ended by signal {signal}")
        };
        lines.push(format!("check {}: {name} {version}, {ending}", index + 1));
    }
    (!lines.is_empty()).then(|| lines.join("\n"))
}

/// The checks in the evidence an executable verdict carries that did not
/// meet their expectation, numbered from 1 in the order they ran, as
/// [`evidence_summary`] numbers them. `None` when `evidence` is not in the
/// shape an executable verifier records.
#[must_use]
pub fn failed_checks(evidence: &serde_json::Value) -> Option<Vec<usize>> {
    let checks = evidence.get("checks")?.as_array()?;
    let mut failed = Vec::new();
    for (index, check) in checks.iter().enumerate() {
        if !check.get("passed")?.as_bool()? {
            failed.push(index + 1);
        }
    }
    Some(failed)
}
