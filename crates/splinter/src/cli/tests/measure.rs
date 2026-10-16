// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The grammar's specs for the measurement verbs: the release gate's settings
//! and canaries, the results matrix, and the experiment plan.

use super::{command, parse, Command};
use crate::cli::{EvalArgs, EvalCommand, ExperimentCommand, TasksCommand};
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
