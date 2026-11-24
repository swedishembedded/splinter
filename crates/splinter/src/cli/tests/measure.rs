// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The grammar's specs for the measurement verbs: the release gate's settings
//! and canaries, the results matrix, the experiment plan, and the comparison
//! of arms.

use super::{command, parse, Command};
use crate::cli::{EvalArgs, EvalCommand, ExperiencesCommand, ExperimentCommand, TasksCommand};
use splinter_sdk::vocabulary::model_ref::POLICY_DEFAULT;

#[test]
fn eval_matrix_reads_a_lineage() {
    let Command::Eval(matrix) = command(&["eval", "matrix", "--alias", "staging"]) else {
        panic!("eval matrix");
    };
    assert!(matches!(
        matrix.command,
        Some(EvalCommand::Matrix { ref alias }) if alias == "staging"
    ));
    assert!(matches!(
        command(&["eval", "matrix"]),
        Command::Eval(EvalArgs { command: Some(EvalCommand::Matrix { alias }), .. }) if alias == POLICY_DEFAULT
    ));
}

#[test]
fn an_experiment_plan_is_checked_against_a_catalog() {
    assert!(matches!(
        command(&["experiment", "check", "plan.toml"]),
        Command::Experiment(ExperimentCommand::Check { catalog: None, .. })
    ));
    assert!(matches!(
        command(&["experiment", "check", "plan.toml", "--catalog", "c.toml"]),
        Command::Experiment(ExperimentCommand::Check {
            catalog: Some(_),
            ..
        })
    ));
    assert!(parse(&["experiment", "check"]).is_err());
}

#[test]
fn tasks_history_takes_a_repository_a_range_and_an_optional_test_command() {
    let Command::Tasks(TasksCommand::History {
        repository,
        range,
        test_command,
        max_commits,
        store,
    }) = command(&[
        "tasks",
        "history",
        "./shop",
        "--range",
        "v1..HEAD",
        "--max-commits",
        "4",
        "--store",
        "--test-command",
        "python3",
        "-m",
        "unittest",
    ])
    else {
        panic!("tasks history");
    };
    assert_eq!(repository, std::path::PathBuf::from("./shop"));
    assert_eq!(range, "v1..HEAD");
    assert_eq!(test_command, ["python3", "-m", "unittest"]);
    assert_eq!(max_commits, Some(4));
    assert!(store);

    // A bare revision reads its history; nothing is stored unless asked.
    let Command::Tasks(TasksCommand::History {
        range,
        store,
        max_commits,
        ..
    }) = command(&["tasks", "history", "./shop"])
    else {
        panic!("tasks history");
    };
    assert_eq!(range, "HEAD");
    assert_eq!(max_commits, None);
    assert!(!store);
}

#[test]
fn compare_takes_the_difficulty_cutoffs_together_or_not_at_all() {
    let Command::Experiences(ExperiencesCommand::Compare { strata, .. }) = command(&[
        "experiences",
        "compare",
        "ab12",
        "cd34",
        "--stratum-checkers",
        "2",
        "--stratum-footprint",
        "3",
        "--stratum-instruction-chars",
        "500",
    ]) else {
        panic!("not a compare");
    };
    assert_eq!(
        strata
            .cutoffs()
            .map(|c| (c.checkers, c.footprint, c.instruction_chars)),
        Some((2, 3, 500))
    );
    assert!(
        parse(&[
            "experiences",
            "compare",
            "ab12",
            "cd34",
            "--stratum-checkers",
            "2"
        ])
        .is_err(),
        "the cutoffs come together"
    );
}

#[test]
fn compare_arms_takes_arms_trials_files_a_reference_and_the_interval_settings() {
    let Command::Experiences(ExperiencesCommand::CompareArms(args)) = command(&[
        "experiences",
        "compare-arms",
        "--arm",
        "base=jobs/e2a-base",
        "--arm",
        "gen1=jobs/e2a-gen1",
        "--trials",
        "more.jsonl",
        "--reference",
        "base",
        "--k",
        "16",
        // seed: the grammar is asked to carry a given seed through unchanged.
        "--interval-seed",
        "20261121",
    ]) else {
        panic!("experiences compare-arms");
    };
    assert_eq!(args.arms, ["base=jobs/e2a-base", "gen1=jobs/e2a-gen1"]);
    assert_eq!(args.reference, "base");
    assert_eq!(args.k, [16]);
    assert_eq!(args.resamples, splinter_sdk::arm_compare::DEFAULT_RESAMPLES);
    assert_eq!(args.level, splinter_sdk::arm_compare::DEFAULT_LEVEL);
    assert_eq!(args.interval_seed.map(|s| s.value()), Some(20_261_121));
    assert!(
        parse(&["experiences", "compare-arms", "--arm", "a=x"]).is_err(),
        "the reference arm is named"
    );
}
