// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The learning factory's verbs: a recipe positional for validate, plan
//! and run (with --out), a run directory for the rest, and the three
//! older verbs that also take a run.

use super::*;

fn command(args: &[&str]) -> Command {
    Cli::try_parse_from(std::iter::once("splinter").chain(args.iter().copied()))
        .expect("parses")
        .command
        .expect("a command")
}

#[test]
fn a_recipe_is_validated_planned_and_run_into_a_run_directory() {
    assert!(
        matches!(command(&["validate", "recipes/vim.yaml"]), Command::Validate(a) if a.recipe.as_os_str() == "recipes/vim.yaml")
    );
    assert!(matches!(
        command(&["plan", "recipes/vim.yaml"]),
        Command::Plan(_)
    ));
    assert!(
        matches!(command(&["run", "r.yaml", "--out", "runs/x"]), Command::Run(a) if a.out.as_os_str() == "runs/x")
    );
    assert!(
        Cli::try_parse_from(["splinter", "run", "r.yaml"]).is_err(),
        "run needs --out"
    );
}

#[test]
fn a_run_directory_is_resumed_cancelled_inspected_forked_reported_and_exported() {
    assert!(matches!(command(&["resume", "runs/x"]), Command::Resume(_)));
    assert!(matches!(command(&["cancel", "runs/x"]), Command::Cancel(_)));
    assert!(
        matches!(command(&["inspect", "runs/x", "--stage", "rollouts", "--viewer", "harbor"]), Command::Inspect(a) if a.viewer.as_deref() == Some("harbor"))
    );
    assert!(
        Cli::try_parse_from(["splinter", "inspect", "runs/x", "--viewer", "harbor"]).is_err(),
        "a viewer needs a stage"
    );
    assert!(matches!(
        command(&["fork", "runs/x", "--recipe", "r.yaml", "--out", "runs/y"]),
        Command::Fork(_)
    ));
    assert!(matches!(
        command(&["fork", "runs/x", "runs/m", "--recipe", "r.yaml", "--out", "runs/y"]),
        Command::Fork(a) if a.runs.len() == 2
    ));
    assert!(
        Cli::try_parse_from(["splinter", "fork", "--recipe", "r.yaml", "--out", "runs/y"]).is_err(),
        "a fork needs a run"
    );
    assert!(matches!(command(&["report", "runs/x"]), Command::Report(_)));
    assert!(matches!(command(&["index", "runs/x"]), Command::Index(_)));
    assert!(matches!(
        command(&["export", "runs/x", "--out", "x.tar.zst"]),
        Command::Export(_)
    ));
    assert!(matches!(
        command(&["status", "runs/x"]),
        Command::Status(RunStatusArgs { run: Some(_) })
    ));
    assert!(matches!(
        command(&["status"]),
        Command::Status(RunStatusArgs { run: None })
    ));
    assert!(
        matches!(command(&["eval", "runs/x", "--plan", "p.yaml", "--out", "runs/z"]), Command::Eval(a) if a.plan.is_some())
    );
    assert!(
        Cli::try_parse_from(["splinter", "eval", "runs/x", "--plan", "p.yaml"]).is_err(),
        "--plan needs --out"
    );
    assert!(matches!(
        command(&["play", "runs/x", "--stage", "guided", "--task", "train/002", "--continue", "0"]),
        Command::Play(a) if a.continue_from == Some(0) && !a.no_run
    ));
    assert!(
        Cli::try_parse_from(["splinter", "play", "runs/x", "--stage", "guided"]).is_err(),
        "play needs a task"
    );
}

#[test]
fn several_runs_are_observed_with_an_optional_seed_and_utilisation_log() {
    assert!(matches!(
        command(&["observe", "runs/g1", "runs/g2", "--seed", "3"]),
        Command::Observe(a) if a.runs.len() == 2 && a.seed == Some(3) && a.utilisation.is_none()
    ));
    assert!(matches!(
        command(&["observe", "runs/g1", "--utilisation", "dmon.log", "--utilisation-utc-offset", "-60"]),
        Command::Observe(a) if a.utilisation_utc_offset == -60
    ));
    assert!(
        Cli::try_parse_from(["splinter", "observe"]).is_err(),
        "observe needs a run"
    );
}
