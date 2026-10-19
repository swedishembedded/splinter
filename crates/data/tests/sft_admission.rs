// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training small models only on
// work that was independently verified, for its clients. If your team needs
// expertise in keeping unverified agent output out of a training set then you
// can procure our services by sending an email to info@swedishembedded.com.

//! Spec: an experience is training evidence only when an independent verdict
//! passed it at the required strength. An experience nobody verified, one
//! whose verdict failed, and one only a weak judge passed produce no record,
//! whatever the solver said in its own final answer.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use common::*;
use serde_json::json;
use splinter_core::annotation::{Label, Outcome, Strength};
use splinter_core::experience::{PrivilegedKind, Task};
use splinter_data::{Control, Corpus, Exclusion, SftStep, View};

const INSTRUCTION: &str = "Compute one plus one and report it.";

fn arithmetic() -> Task {
    task(
        "arithmetic",
        INSTRUCTION,
        vec![privileged(PrivilegedKind::Reference, "2")],
    )
}

/// The solver claims success in its final answer; that is its own claim.
fn claimed_success(task: &Task) -> splinter_core::experience::Experience {
    experience(
        task,
        trajectory(vec![
            user(1, INSTRUCTION),
            call(2, "c1", "run_code", json!({"code": "1+1"}), "2"),
            say(3, "Done: all tests pass and the answer is 2"),
        ]),
        Some("Done: all tests pass and the answer is 2"),
        "2026-10-18T01:00:00.000Z",
    )
}

fn records_for(verdicts: &[(Outcome, Strength)]) -> (usize, Option<Exclusion>) {
    let task = arithmetic();
    let experience = claimed_success(&task);
    let notes = verdicts
        .iter()
        .map(|(outcome, strength)| verdict(&experience, *outcome, *strength))
        .collect();
    let mut corpus = Corpus::new();
    corpus.insert(experience, notes).unwrap();
    let projection = SftStep::new(Strength::Executable).project(&corpus).unwrap();
    let excluded = [Exclusion::Undecided, Exclusion::Failed, Exclusion::TooWeak]
        .into_iter()
        .find(|e| projection.count(*e) > 0);
    (projection.records.len(), excluded)
}

#[test]
fn an_experience_nobody_verified_is_not_training_evidence() {
    assert_eq!(records_for(&[]), (0, Some(Exclusion::Undecided)));
}

#[test]
fn a_failed_verdict_is_not_training_evidence_whatever_the_solver_claimed() {
    assert_eq!(
        records_for(&[(Outcome::Fail, Strength::Executable)]),
        (0, Some(Exclusion::Failed))
    );
}

#[test]
fn a_pass_weaker_than_execution_is_not_training_evidence() {
    assert_eq!(
        records_for(&[(Outcome::Pass, Strength::Judged)]),
        (0, Some(Exclusion::TooWeak))
    );
}

#[test]
fn an_executable_pass_is_admitted() {
    let (records, excluded) = records_for(&[(Outcome::Pass, Strength::Executable)]);
    assert!(records > 0 && excluded.is_none());
}

/// An experience nobody verified, or whose verdict failed, in which the
/// labeller marked the tool call good.
fn labelled_good(verdicts: &[(Outcome, Strength)]) -> (usize, usize) {
    let task = arithmetic();
    let experience = claimed_success(&task);
    let mut notes: Vec<_> = verdicts
        .iter()
        .map(|(outcome, strength)| verdict(&experience, *outcome, *strength))
        .collect();
    notes.push(label(&experience, 2, Label::Good));
    let mut corpus = Corpus::new();
    corpus.insert(experience, notes).unwrap();
    let step_credit = SftStep::new(Strength::Executable).project(&corpus).unwrap();
    let verified = SftStep::new(Strength::Executable)
        .control(Control::OutcomeVerified)
        .project(&corpus)
        .unwrap();
    assert!(!step_credit.outcome_verified_only && verified.outcome_verified_only);
    (step_credit.records.len(), verified.records.len())
}

#[test]
fn a_good_step_of_an_unverified_or_failed_experience_is_admitted_only_without_outcome_verification()
{
    for verdicts in [
        vec![],
        vec![(Outcome::Fail, Strength::Executable)],
        vec![(Outcome::Pass, Strength::Judged)],
    ] {
        let (step_credit, verified) = labelled_good(&verdicts);
        assert!(
            step_credit > 0,
            "{verdicts:?}: the default keeps step-level credit"
        );
        assert_eq!(verified, 0, "{verdicts:?}: outcome-verified admits nothing");
    }
    // A verified experience is admitted either way.
    let (step_credit, verified) = labelled_good(&[(Outcome::Pass, Strength::Executable)]);
    assert!(step_credit > 0 && verified > 0);
}
