// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The grammar of `env`.

use super::*;

#[test]
fn an_objective_is_given_as_text_or_as_a_file_and_never_both_nor_neither() {
    assert!(parse(&["env", "classify"]).is_err());
    assert!(parse(&["env", "classify", "--text", "a", "--file", "b"]).is_err());
    assert!(matches!(
        command(&["env", "classify", "--text", "predict the output"]),
        Command::Env(EnvCommand::Classify { .. })
    ));
}

#[test]
fn sources_and_constraints_repeat_beside_an_objective() {
    let Command::Env(EnvCommand::Compile {
        objective,
        environments,
        instances,
        seed,
        ..
    }) = command(&[
        "env",
        "compile",
        "--file",
        "o.txt",
        "--source",
        "a.md",
        "--source",
        "b.md",
        "--constraint",
        "lua",
        "--constraint",
        "calibration set",
        "--seed",
        "7",
    ])
    else {
        panic!("not a compile");
    };
    assert_eq!(objective.sources, ["a.md", "b.md"]);
    assert_eq!(objective.constraints, ["lua", "calibration set"]);
    assert_eq!((environments, instances, seed), (4, 4, Some(7)));
}

#[test]
fn a_band_names_the_environment_the_model_and_both_counts() {
    assert!(parse(&["env", "band", "env-0123456789abcdef", "--model", "base"]).is_err());
    assert!(matches!(
        command(&[
            "env",
            "band",
            "env-0123456789abcdef",
            "--model",
            "base",
            "--attempts",
            "32",
            "--passes",
            "12",
        ]),
        Command::Env(EnvCommand::Band {
            attempts: 32,
            passes: 12,
            ..
        })
    ));
}
