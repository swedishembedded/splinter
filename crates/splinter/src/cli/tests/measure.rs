// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The grammar's specs for the measurement verbs: the release gate's settings
//! and canaries, the results matrix, and the experiment plan.

use super::{command, parse, Command};
use crate::cli::{EvalArgs, EvalCommand, ExperiencesCommand, ExperimentCommand, TasksCommand};
use splinter_sdk::vocabulary::model_ref::POLICY_DEFAULT;

#[test]
fn the_release_gate_takes_its_statistical_settings_and_canaries() {
    let Command::Release(planned) = command(&[
        "release",
        "candidate-1",
        "--alpha-spending",
        "--target-effect",
        "0.2",
        "--anchor-max-net-lost",
        "6",
    ]) else {
        panic!("release with a gate plan");
    };
    assert_eq!(planned.gate.anchor_max_net_lost, Some(6));
    assert!(planned.gate.alpha_spending);
    assert_eq!(planned.gate.target_effect, Some(0.2));
    let Command::Release(canaried) = command(&["release", "c", "--canaries", "ab12"]) else {
        panic!("release with canaries");
    };
    assert_eq!(canaried.canaries.as_deref(), Some("ab12"));
    assert!(matches!(
        command(&["tasks", "canaries", "ab12", "--count", "5"]),
        Command::Tasks(TasksCommand::Canaries { ref task_set, count: 5 }) if task_set == "ab12"
    ));
    assert!(matches!(
        command(&["tasks", "canaries", "ab12"]),
        Command::Tasks(TasksCommand::Canaries { count: 20, .. })
    ));
    assert!(parse(&["tasks", "canaries", "ab12", "--count", "0"]).is_err());
    assert!(matches!(
        command(&["tasks", "pick", "ab", "cd", "--family-share", "0.25", "--seed", "7"]),
        Command::Tasks(TasksCommand::Pick { ref task_sets, family_share: Some(f), per_family: None, .. })
            if task_sets.len() == 2 && f == 0.25
    ));
    assert!(matches!(
        command(&["tasks", "pick", "ab", "--per-family", "1", "--seed", "7"]),
        Command::Tasks(TasksCommand::Pick {
            per_family: Some(1),
            family_share: None,
            ..
        })
    ));
    // Exactly one quota, in range, and the seed is not drawn for a split.
    assert!(parse(&["tasks", "pick", "ab", "--seed", "7"]).is_err());
    assert!(parse(&[
        "tasks",
        "pick",
        "ab",
        "--family-share",
        "0.5",
        "--per-family",
        "1",
        "--seed",
        "7"
    ])
    .is_err());
    assert!(parse(&[
        "tasks",
        "pick",
        "ab",
        "--family-share",
        "1.5",
        "--seed",
        "7"
    ])
    .is_err());
    assert!(parse(&["tasks", "pick", "ab", "--per-family", "1"]).is_err());
    assert!(parse(&["release", "c", "--target-effect", "0"]).is_err());
    assert!(parse(&["release", "c", "--target-effect", "1.5"]).is_err());
}

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
