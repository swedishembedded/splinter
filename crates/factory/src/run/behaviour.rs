// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training data built from what agents
// actually did, every record traceable to the attempt it came from, for its
// clients. If your team needs expertise in learning from agent experience,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! How an attempt behaved, read from its files: whether the agent's loop watchdog ended it and how often it
//! repeated a call that had just failed. Both are metrics in `[0, 1]` of the attempt's verdict, so a reward can
//! penalise them (`dataset.project`'s `reward.penalties`).

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::Value;
use splinter_core::factory::product::TrialResult;

/// The metric that is 1 when a loop watchdog ended the attempt.
pub const LOOP: &str = "loop";
/// The metric that is the share of the calls after a failed call that repeated it.
pub const REPEAT_AFTER_ERROR: &str = "repeat_after_error";

/// The text of a tool result that reports an editor error.
const ERROR_MARK: &str = "Error detected";

/// The behaviour metrics of `trial`, whose files are under `view`; one it left no files for has none.
#[must_use]
pub fn metrics(view: &Path, trial: &TrialResult) -> BTreeMap<String, f64> {
    let mut out = BTreeMap::new();
    let Some(trace) = trial.trace.as_ref() else {
        return out;
    };
    let trace = view.join(&trace.path);
    if let Some(dir) = trace.parent() {
        if let Ok(stderr) = std::fs::read_to_string(dir.join("stderr.txt")) {
            out.insert(LOOP.into(), f64::from(u8::from(stderr.contains("loop:"))));
        }
    }
    if let Some(share) = std::fs::read_to_string(&trace)
        .ok()
        .and_then(|text| serde_json::from_str::<Value>(&text).ok())
        .map(|trajectory| repeat_after_error(&trajectory))
    {
        out.insert(REPEAT_AFTER_ERROR.into(), share);
    }
    out
}

/// Of the calls that followed a call whose result showed an error, the share identical to it (0 when none did).
fn repeat_after_error(trajectory: &Value) -> f64 {
    let steps = trajectory["steps"]
        .as_array()
        .map_or(&[][..], Vec::as_slice);
    let calls: Vec<String> = steps
        .iter()
        .filter(|s| s["source"] == "agent")
        .flat_map(|s| s["tool_calls"].as_array().into_iter().flatten())
        .map(|c| c["arguments"].to_string())
        .collect();
    let results: Vec<String> = steps
        .iter()
        .flat_map(|s| s["observation"]["results"].as_array().into_iter().flatten())
        .map(|r| r["content"].to_string())
        .collect();
    let (mut after_error, mut repeats) = (0u32, 0u32);
    for i in 1..calls.len().min(results.len() + 1) {
        if results[i - 1].contains(ERROR_MARK) {
            after_error += 1;
            repeats += u32::from(calls[i] == calls[i - 1]);
        }
    }
    if after_error == 0 {
        0.0
    } else {
        f64::from(repeats) / f64::from(after_error)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn step(call: &str, result: &str) -> Vec<Value> {
        vec![
            json!({"source": "agent", "tool_calls": [{"arguments": {"cmd": call}}]}),
            json!({"source": "system", "observation": {"results": [{"content": result}]}}),
        ]
    }

    #[test]
    fn a_call_repeated_after_an_error_is_counted_and_a_corrected_one_is_not() {
        let mut steps = step(":s/a/b/", "Error detected while processing");
        steps.extend(step(":s/a/b/", "Error detected while processing"));
        steps.extend(step(":s/a/c/", "ok"));
        let share = repeat_after_error(&json!({ "steps": steps }));
        assert!((share - 0.5).abs() < 1e-9, "{share}");
        assert_eq!(repeat_after_error(&json!({ "steps": [] })), 0.0);
    }
}
