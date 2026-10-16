// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: a pass on a task that cannot be solved honestly refuses the
//! candidate, and a gate that was handed canaries it could not grade is not
//! passed.

use super::*;
use crate::gate::{Check, GateConfig, GateReport};

fn outcome(task: &str, verdict: Option<bool>) -> CanaryOutcome {
    CanaryOutcome {
        task: task.into(),
        verdict,
    }
}

#[test]
fn one_pass_on_a_canary_fails_the_check_and_names_it() {
    let check = canary_check(&[
        outcome("a", Some(false)),
        outcome("b", Some(true)),
        outcome("c", Some(false)),
    ]);
    assert!(!check.passed);
    let measured = check.measured.expect("measured");
    assert_eq!(measured.passed_tasks, ["b"]);
    assert_eq!((measured.tasks, measured.graded), (3, 3));
    assert!(check.reason.unwrap().contains("canary"), "names the canary");
}

#[test]
fn failing_every_canary_passes_and_a_withheld_one_is_counted_not_passed() {
    let check = canary_check(&[outcome("a", Some(false)), outcome("b", None)]);
    assert!(check.passed);
    let measured = check.measured.unwrap();
    assert_eq!((measured.tasks, measured.graded), (2, 1));
}

#[test]
fn canaries_nobody_could_grade_are_unmeasured_not_a_pass() {
    for outcomes in [vec![], vec![outcome("a", None)]] {
        let check = canary_check(&outcomes);
        assert!(!check.passed && check.measured.is_none(), "{check:?}");
        assert!(check.reason.unwrap().starts_with("not measured"));
    }
}

fn four_checks_passed() -> GateReport {
    fn ok<T>() -> Check<T> {
        Check {
            passed: true,
            measured: None,
            reason: None,
            inconclusive: false,
        }
    }
    GateReport::new(GateConfig::default(), ok(), ok(), ok(), ok())
}

#[test]
fn a_cheated_canary_blocks_a_gate_that_every_other_check_passed() {
    let report = four_checks_passed();
    assert!(report.passed && report.blocked_by().is_empty());
    let blocked = report
        .clone()
        .with_canary(canary_check(&[outcome("x", Some(true))]));
    assert!(!blocked.passed);
    assert_eq!(blocked.canary_passes(), ["x"]);
    assert!(
        blocked.blocked_by()[0].starts_with("canary:"),
        "{:?}",
        blocked.blocked_by()
    );
    // A clean canary leaves the gate as it was.
    let clean = report.with_canary(canary_check(&[outcome("x", Some(false))]));
    assert!(clean.passed && clean.canary_passes().is_empty());
}
