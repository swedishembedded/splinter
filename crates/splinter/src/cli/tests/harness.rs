// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The command line of `solve --harness` and `harness optimise`.

use super::*;

#[test]
fn a_harness_solves_tasks_and_is_searched_with_an_explicit_seed() {
    assert!(matches!(
        command(&["solve", "ab12", "--harness", "h.yaml"]),
        Command::Solve(s) if s.harness.is_some()
    ));
    assert!(
        parse(&["solve", "ab12", "--harness", "h.yaml", "--frontier"]).is_err(),
        "a harness is not a frontier measurement"
    );
    let Command::Harness(HarnessCommand::Optimise(optimise)) = command(&[
        "harness",
        "optimise",
        "ab12",
        "--baseline",
        "b.yaml",
        "--space",
        "s.yaml",
        "--evaluations",
        "8",
        "--search-seed",
        "3",
    ]) else {
        panic!("harness optimise");
    };
    assert_eq!(
        (
            optimise.evaluations,
            optimise.search_seed,
            optimise.attempts
        ),
        (8, 3, 1)
    );
    assert_eq!(optimise.strategy, StrategyArg::Annealing);
    assert!(
        parse(&[
            "harness",
            "optimise",
            "ab12",
            "--baseline",
            "b.yaml",
            "--space",
            "s.yaml",
            "--evaluations",
            "8"
        ])
        .is_err(),
        "the search seed has no default"
    );
    assert!(
        parse(&[
            "harness",
            "optimise",
            "ab12",
            "--baseline",
            "b.yaml",
            "--evaluations",
            "8",
            "--search-seed",
            "3",
            "--proposer",
            "model",
        ])
        .is_err(),
        "a model proposer names its model"
    );
    let Command::Harness(HarnessCommand::Optimise(by_model)) = command(&[
        "harness",
        "optimise",
        "ab12",
        "--baseline",
        "b.yaml",
        "--evaluations",
        "8",
        "--search-seed",
        "3",
        "--proposer",
        "model",
        "--proposer-model",
        "local:/p",
        "--parallel",
        "4",
    ]) else {
        panic!("harness optimise");
    };
    assert!(
        by_model.space.is_none(),
        "a space is optional with a model proposer"
    );
    assert_eq!((by_model.parallel, by_model.proposer_repairs), (4, 3));
}
