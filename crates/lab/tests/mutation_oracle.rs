// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: a generated test becomes an executable oracle only when it passes
//! on the reference solution and fails on its mutants. A strong test is
//! admitted, a vacuous one or one the reference fails is not, the kill
//! counts say why, and the mutation-validated verifier grades with admitted
//! tests alone.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use common::{experience, python, record, reference, task, verdict, Scratch};
use splinter_lab::verifiers::executable::{ExecutableCheck, Expectation};
use splinter_lab::verifiers::mutation::{
    mutants, validate_oracle, MutationOperator, MutationPolicy, MutationValidatedVerifier,
};
use splinter_sandbox::Limits;
use splinter_store::annotation::{Outcome, Strength};

const REFERENCE: &str = "def clamp(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x
";

const STRONG: &str = "assert clamp(-1, 0, 10) == 0
assert clamp(11, 0, 10) == 10
assert clamp(5, 0, 10) == 5
assert clamp(0, 0, 10) == 0
assert clamp(10, 0, 10) == 10
";

fn test(code: &str) -> ExecutableCheck {
    ExecutableCheck {
        code: code.into(),
        stdin: None,
        expect: Expectation::exit_code(0),
        environment: None,
    }
}

#[test]
fn mutants_are_deterministic_distinct_and_differ_from_the_source() {
    let first = mutants(REFERENCE, 64);
    assert_eq!(first, mutants(REFERENCE, 64));
    assert!(first.iter().all(|m| m.text != REFERENCE));
    let mut texts: Vec<&str> = first.iter().map(|m| m.text.as_str()).collect();
    texts.sort_unstable();
    texts.dedup();
    assert_eq!(texts.len(), first.len(), "no two mutants are the same text");
    for op in [
        MutationOperator::FlipComparison,
        MutationOperator::DeleteLine,
    ] {
        assert!(first.iter().any(|m| m.operator == op), "{op:?} applies");
    }
    assert!(mutants(REFERENCE, 3).len() <= 3, "the cap holds");
}

#[test]
fn a_strong_test_is_admitted_having_killed_every_mutant() {
    let scratch = Scratch::new("mutation-strong");
    let env = python(&scratch, Limits::default());
    let v = validate_oracle(REFERENCE, &test(STRONG), &env, &MutationPolicy::default()).unwrap();
    assert!(v.passes_reference);
    assert!(v.admitted, "{v:?}");
    assert!(v.killed > 0);
    assert_eq!(v.survived, 0);
    assert!(
        v.errored > 0,
        "deleting a block's only line cannot run: {v:?}"
    );
}

#[test]
fn a_vacuous_test_or_one_the_reference_fails_is_rejected() {
    let scratch = Scratch::new("mutation-weak");
    let env = python(&scratch, Limits::default());
    let policy = MutationPolicy::default();

    let vacuous = validate_oracle(REFERENCE, &test("assert True\n"), &env, &policy).unwrap();
    assert!(vacuous.passes_reference);
    assert!(!vacuous.admitted);
    assert_eq!(vacuous.killed, 0);
    assert!(vacuous.survived > 0);

    let wrong = validate_oracle(
        REFERENCE,
        &test("assert clamp(5, 0, 10) == 6\n"),
        &env,
        &policy,
    )
    .unwrap();
    assert!(!wrong.passes_reference);
    assert!(!wrong.admitted);
}

#[test]
fn the_verifier_grades_with_admitted_tests_only() {
    let scratch = Scratch::new("mutation-verifier");
    let env = python(&scratch, Limits::default());
    let generated = |code: &str| test(code).as_generated_test().unwrap();
    let graded = task(
        "code",
        record(&env),
        vec![
            reference(REFERENCE),
            generated(STRONG),
            generated("assert True\n"),
        ],
    );
    let verifier = MutationValidatedVerifier::new(vec![env.clone()], MutationPolicy::default());

    let (outcome, strength, evidence) = verdict(
        &verifier,
        &graded,
        &experience(&graded, Some(REFERENCE), "s"),
    );
    assert_eq!(
        (outcome, strength),
        (Outcome::Pass, Strength::Executable),
        "{evidence}"
    );
    let tests = evidence["tests"].as_array().unwrap();
    assert_eq!(tests.len(), 2);
    assert_eq!(tests[0]["validation"]["admitted"], true);
    assert_eq!(tests[1]["validation"]["admitted"], false);

    let wrong = "def clamp(x, lo, hi):\n    return x\n";
    let (outcome, _, _) = verdict(&verifier, &graded, &experience(&graded, Some(wrong), "s"));
    assert_eq!(outcome, Outcome::Fail);

    let only_vacuous = task(
        "code",
        record(&env),
        vec![reference(REFERENCE), generated("assert True\n")],
    );
    let (outcome, _, _) = verdict(
        &verifier,
        &only_vacuous,
        &experience(&only_vacuous, Some(wrong), "s"),
    );
    assert_eq!(outcome, Outcome::Abstain, "no admitted oracle, no verdict");
}
