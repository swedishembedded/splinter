// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The grammar's specs for the flags that reach the models: the remote
//! request width, the served temperature, the solve deadline and the training
//! row's bound, and the tier the frozen base is held at.

use super::{command, parse, Command};

#[test]
fn the_remote_knobs_are_global_flags_and_bounded() {
    for args in [
        &[
            "--remote-concurrency",
            "8",
            "--temperature",
            "0.7",
            "status",
        ][..],
        &[
            "status",
            "--remote-concurrency",
            "8",
            "--temperature",
            "0.7",
        ][..],
    ] {
        let cli = parse(args).unwrap_or_else(|e| panic!("{args:?}: {e}"));
        assert_eq!(cli.global.remote_concurrency, Some(8));
        assert_eq!(cli.global.temperature, Some(0.7));
    }
    let none = parse(&["status"]).unwrap();
    assert_eq!(
        (none.global.remote_concurrency, none.global.temperature),
        (None, None)
    );
    assert!(parse(&["--remote-concurrency", "0", "status"]).is_err());
    assert!(parse(&["--temperature", "2.5", "status"]).is_err());
    assert!(parse(&["--temperature", "-0.1", "status"]).is_err());
}

#[test]
fn train_takes_a_positive_max_block() {
    match command(&["train", "ds1", "--max-block", "8192"]) {
        Command::Train(args) => assert_eq!(args.max_block, Some(8192)),
        other => panic!("{other:?}"),
    }
    assert!(parse(&["train", "ds1", "--max-block", "0"]).is_err());
}

#[test]
fn one_temperature_flag_serves_the_model_and_the_pass_at_k_sampling_the_one_after_the_verb_wins() {
    // clap folds the global flag and the verb's own into one value: the
    // last given wins, and it is both the served model's temperature and
    // the verb's pass@k sampling.
    let both = |args: &[&str]| {
        let cli = parse(args).unwrap();
        let Some(Command::Solve(solve)) = cli.command else {
            panic!("solve --frontier");
        };
        let sampled = solve.pass_at_k.pass_at_k().sampling.map(|s| s.temperature);
        (cli.global.temperature, sampled)
    };
    let verb = ["solve", "ab12", "--frontier", "--k", "8"];
    let after: Vec<&str> = ["--temperature", "0.2"]
        .into_iter()
        .chain(verb)
        .chain(["--temperature", "1.0"])
        .collect();
    assert_eq!(both(&after), (Some(1.0), Some(1.0)));
    let before: Vec<&str> = verb.into_iter().chain(["--temperature", "1.0"]).collect();
    assert_eq!(both(&before), (Some(1.0), Some(1.0)));
}

#[test]
fn the_solve_deadline_is_a_global_flag_of_whole_seconds() {
    for args in [
        &["--solve-deadline", "1800", "status"][..],
        &["status", "--solve-deadline", "1800"][..],
    ] {
        let cli = parse(args).unwrap_or_else(|e| panic!("{args:?}: {e}"));
        assert_eq!(cli.global.solve_deadline, Some(1800));
    }
    assert_eq!(parse(&["status"]).unwrap().global.solve_deadline, None);
    assert!(parse(&["--solve-deadline", "0", "status"]).is_err());
    assert!(parse(&["--solve-deadline", "1.5", "status"]).is_err());
}

#[test]
fn baseline_takes_the_binary_models_and_seeds_as_given_and_no_default_seed() {
    let Command::Baseline(args) = command(&[
        "baseline",
        "--brain-bin",
        "/b/brain",
        "--model",
        "/m/a",
        "--model",
        "/m/b",
        "--exam-seed",
        "20261017",
        "--require-clean",
    ]) else {
        panic!("not baseline");
    };
    assert_eq!(args.models.len(), 2);
    assert_eq!(
        (args.exam_seed, args.training_seed),
        (Some(20_261_017), None)
    );
    assert!(args.require_clean);
}
#[test]
fn train_names_the_tier_the_frozen_base_is_held_at() {
    use splinter_sdk::model::train::BasePrecision;
    let Command::Train(plain) = command(&["train", "ds-1"]) else {
        panic!("train");
    };
    assert_eq!(plain.base_precision, None);
    let Command::Train(nf4) = command(&["train", "ds-1", "--base-precision", "nf4"]) else {
        panic!("train --base-precision");
    };
    assert_eq!(nf4.base_precision, Some(BasePrecision::Nf4));
    assert!(parse(&["train", "ds-1", "--base-precision", "fp8"]).is_err());
}

#[test]
fn the_sampling_of_a_served_model_is_global_flags_and_bounded() {
    let cli = parse(&[
        "--temperature",
        "0.6",
        "--top-p",
        "0.95",
        "--top-k",
        "20",
        "--sampling-seed",
        "16022413115216019911",
        "--thinking",
        "status",
    ])
    .unwrap();
    assert_eq!(
        (cli.global.top_p, cli.global.top_k, cli.global.sampling_seed),
        (Some(0.95), Some(20), Some(16_022_413_115_216_019_911))
    );
    assert!(cli.global.thinking);
    let none = parse(&["status"]).unwrap();
    assert_eq!(
        (
            none.global.top_p,
            none.global.top_k,
            none.global.sampling_seed
        ),
        (None, None, None)
    );
    assert!(!none.global.thinking);
    for bad in [["--top-p", "0"], ["--top-p", "1.5"], ["--top-k", "0"]] {
        assert!(parse(&[bad[0], bad[1], "status"]).is_err(), "{bad:?}");
    }
}
