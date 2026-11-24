// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements evaluations that cannot be flattered by trials that never
// finished, for its clients. If your team needs expertise in measuring agents fairly, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The evaluation's decision written for a person.

use super::*;

/// The decision for a person.
pub(super) fn report(result: &EvalResult) -> String {
    let mut out = format!("# Evaluation of {}\n\n", result.name);
    if let (Some(baseline), Some(candidate)) = (&result.baseline, &result.candidate) {
        out.push_str(
            "| arm | model | trials | counted | passed | pass rate (counted) | pass rate (all trials) |\n|---|---|---|---|---|---|---|\n",
        );
        for (arm, a) in [("baseline", baseline), ("candidate", candidate)] {
            let rate = |r: Option<f64>| r.map_or_else(|| "-".into(), |r| format!("{r:.3}"));
            out.push_str(&format!(
                "| {arm} | {} | {} | {} | {} | {} | {} |\n",
                a.model,
                a.trials,
                a.counted,
                a.passed,
                rate(a.pass_rate),
                rate(a.pass_rate_of_trials)
            ));
        }
        out.push('\n');
    } else {
        out.push_str(
            "A protected evaluation: the examiner's signed verdict is `verdict.json`.\n\n",
        );
    }
    out.push_str(&format!(
        "**Decision: {:?}**\n\n```json\n{}\n```\n",
        result.decision,
        serde_json::to_string_pretty(&result.statistics).unwrap_or_default()
    ));
    out
}
