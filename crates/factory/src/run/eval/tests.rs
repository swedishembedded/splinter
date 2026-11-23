// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements evaluations that cannot be flattered by trials that never
// finished, for its clients. If your team needs expertise in measuring agents fairly, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The paired decision and the arm rates of an evaluation.

use super::*;

const ZERO: &str = "blake3:0000000000000000000000000000000000000000000000000000000000000000";

/// A trial of slot 0 of `task` in the given arm, counted or not, passed or not.
fn trial(arm: &str, task: &str, counted: bool, passed: bool) -> TrialResult {
    serde_json::from_value(json!({
        "schema": "splinter.trial/1",
        "slot": { "pass": arm, "task": task, "slot": 0 },
        "attempt": 1, "task": task, "task_digest": ZERO,
        "agent": { "name": "sven", "version": null, "invocation": ZERO },
        "grade": {
            "status": if counted { "graded" } else { "infra_timeout" },
            "cause": null, "counted": counted,
            "score": counted.then_some(if passed { 1.0 } else { 0.0 }),
            "passed": counted.then_some(passed),
            "metrics": {}, "vetoes": []
        },
        "native_dir": "x"
    }))
    .expect("a trial")
}

fn arm(name: &str, outcomes: &[(bool, bool)]) -> Vec<TrialResult> {
    outcomes
        .iter()
        .enumerate()
        .map(|(i, (counted, passed))| trial(name, &format!("t{i}"), *counted, *passed))
        .collect()
}

#[test]
fn a_complete_arm_pair_is_decided() {
    let base = arm("base", &[(true, false); 12]);
    let cand = arm("cand", &[(true, true); 12]);
    let (decision, stats) = decide(&base, &cand, 0.05, 0.0);
    assert_eq!(decision, Decision::Improved);
    assert_eq!(stats["unpaired_slots"], 0);
    assert_eq!(stats["incomplete"], false);
}

#[test]
fn a_slot_one_arm_never_counted_makes_the_verdict_inconclusive() {
    // Twelve slots, the candidate passing all it counted and winning every pair,
    // but one candidate slot timed out on every retry: not a verdict.
    let base = arm("base", &[(true, false); 12]);
    let mut outcomes = vec![(true, true); 11];
    outcomes.push((false, false));
    let cand = arm("cand", &outcomes);
    let (decision, stats) = decide(&base, &cand, 0.05, 0.0);
    assert_eq!(decision, Decision::Inconclusive);
    assert_eq!(stats["unpaired_slots"], 1);
    assert_eq!(stats["incomplete"], true);
}

#[test]
fn the_rate_over_all_trials_counts_an_uncounted_trial_as_not_passed() {
    let mut trials = arm("cand", &[(true, true); 3]);
    trials.push(trial("cand", "t3", false, false));
    let result = arm_result("candidate", "m", &trials, Digest::of(b"x"));
    assert_eq!(result.counted, 3);
    assert_eq!(result.pass_rate, Some(1.0));
    assert_eq!(result.pass_rate_of_trials, Some(0.75));
}
