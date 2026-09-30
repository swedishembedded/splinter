// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: the executable verifier runs a task's checks against the solver's
//! answer in the environment the task names, passes only when every check
//! meets its expectation, fails a run a limit stopped, abstains where it
//! has nothing to run, and records digests of what ran and what came back
//! rather than the outputs themselves.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use common::{experience, python, record, task, verdict, Scratch};
use splinter_lab::verifiers::executable::{
    evidence_summary, ExecutableCheck, ExecutableVerifier, Expectation, ExpectedStdout,
};
use splinter_lab::verifiers::normalise::Normalisation;
use splinter_sandbox::Limits;
use splinter_store::annotation::{Outcome, Strength};
use splinter_store::experience::Environment;

const CORRECT: &str = "Here it is:\n```python\ndef add(a, b):\n    return a + b\n```\n";
const WRONG: &str = "def add(a, b):\n    return a - b\n";
const SPINS: &str = "def add(a, b):\n    while True:\n        pass\n";

fn checks() -> Vec<ExecutableCheck> {
    vec![
        ExecutableCheck {
            code: "assert add(2, 3) == 5\nassert add(-1, 1) == 0\n".into(),
            stdin: None,
            expect: Expectation::exit_code(0),
            environment: None,
        },
        ExecutableCheck {
            code: "print(add(2, 2))\n".into(),
            stdin: None,
            expect: Expectation {
                exit_code: Some(0),
                stdout: Some(ExpectedStdout {
                    text: "4".into(),
                    normalisation: Normalisation::WHITESPACE,
                }),
            },
            environment: None,
        },
    ]
}

fn code_task(env: &splinter_sandbox::RuntimeEnvironment) -> splinter_store::experience::Task {
    let privileged = checks().iter().map(|c| c.as_check().unwrap()).collect();
    task("code", record(env), privileged)
}

#[test]
fn a_correct_solution_passes_and_the_evidence_records_digests_not_outputs() {
    let scratch = Scratch::new("exec-correct");
    let env = python(&scratch, Limits::default());
    let task = code_task(&env);
    let verifier = ExecutableVerifier::new(vec![env]);
    let (outcome, strength, evidence) =
        verdict(&verifier, &task, &experience(&task, Some(CORRECT), "s"));
    assert_eq!(outcome, Outcome::Pass, "{evidence}");
    assert_eq!(strength, Strength::Executable);
    let runs = evidence["checks"].as_array().unwrap();
    assert_eq!(runs.len(), 2);
    for run in runs {
        assert!(run["program"].as_str().unwrap().starts_with("sha256:"));
        assert!(run["stdout"].as_str().unwrap().starts_with("sha256:"));
        assert_eq!(
            run["environment"],
            task.environment.snapshot.as_ref().unwrap().as_str()
        );
        assert_eq!(run["timed_out"], false);
    }
    assert!(
        !evidence.to_string().contains("return a + b"),
        "the evidence carries no program text: {evidence}"
    );
}

#[test]
fn a_wrong_solution_or_no_answer_fails() {
    let scratch = Scratch::new("exec-wrong");
    let env = python(&scratch, Limits::default());
    let task = code_task(&env);
    let verifier = ExecutableVerifier::new(vec![env]);
    let (outcome, _, evidence) = verdict(&verifier, &task, &experience(&task, Some(WRONG), "s"));
    assert_eq!(outcome, Outcome::Fail, "{evidence}");
    assert_eq!(evidence["checks"][0]["passed"], false);
    let summary = evidence_summary(&evidence).unwrap();
    assert!(
        summary.starts_with("check 1: python") && summary.contains(", exit code 1\n"),
        "a reader's summary of how the checks ended: {summary}"
    );
    let (outcome, _, _) = verdict(&verifier, &task, &experience(&task, None, "s"));
    assert_eq!(outcome, Outcome::Fail);
}

#[test]
fn a_solution_stopped_by_the_time_limit_fails_and_says_so() {
    let scratch = Scratch::new("exec-timeout");
    let limits = Limits {
        wall_time_ms: 1_000,
        cpu_seconds: 1,
        ..Limits::default()
    };
    let env = python(&scratch, limits);
    let task = code_task(&env);
    let verifier = ExecutableVerifier::new(vec![env]);
    let (outcome, _, evidence) = verdict(&verifier, &task, &experience(&task, Some(SPINS), "s"));
    assert_eq!(outcome, Outcome::Fail);
    assert_eq!(evidence["checks"][0]["timed_out"], true, "{evidence}");
    assert!(evidence_summary(&evidence).unwrap().contains("timed out"));
}

#[test]
fn a_task_it_cannot_run_is_abstained_on() {
    let scratch = Scratch::new("exec-abstain");
    let env = python(&scratch, Limits::default());
    let verifier = ExecutableVerifier::new(vec![env.clone()]);

    let no_checks = task("recall", Environment::closed_book(), vec![]);
    let (outcome, strength, _) = verdict(
        &verifier,
        &no_checks,
        &experience(&no_checks, Some(CORRECT), "s"),
    );
    assert_eq!(
        (outcome, strength),
        (Outcome::Abstain, Strength::Executable)
    );

    // Checks in an environment the verifier was not offered.
    let other = python(
        &scratch,
        Limits {
            cpu_seconds: 2,
            ..Limits::default()
        },
    );
    let elsewhere = code_task(&other);
    let (outcome, _, evidence) = verdict(
        &verifier,
        &elsewhere,
        &experience(&elsewhere, Some(CORRECT), "s"),
    );
    assert_eq!(outcome, Outcome::Abstain, "{evidence}");
}
