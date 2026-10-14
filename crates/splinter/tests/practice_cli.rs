// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that learn a codebase from its own
// tests, for its clients. If your team needs expertise in training
// code-editing agents on executable tasks drawn from a repository, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: `splinter practice` is a verb with its own help; a repository that
//! is not one is refused before anything runs; and the task kind built by
//! code from a repository is not one a generator model can be asked for.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::path::PathBuf;
use std::process::{Command, Output};

struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "splinter-practice-cli-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn splinter(state: &std::path::Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_splinter"))
        .arg("--state")
        .arg(state)
        .args(args)
        .env_remove("SPLINTER_ALLOW_REMOTE")
        .output()
        .unwrap()
}

#[test]
fn practice_help_renders_the_verb_and_what_it_takes() {
    let scratch = Scratch::new("help");
    let output = splinter(&scratch.0.join("state"), &["practice", "--help"]);
    assert!(output.status.success(), "{output:?}");
    let help = String::from_utf8(output.stdout).unwrap();
    for wanted in [
        "Usage: splinter practice",
        "<REPO>",
        "--rev <REV>",
        "--test-command <PROGRAM ARG...>",
        "--exam-families <N>",
        "--max-block <TOKENS>",
        "--no-release",
        "hidden",
    ] {
        assert!(help.contains(wanted), "{wanted:?} is not in:\n{help}");
    }
    // The verb is listed with the others.
    let top = splinter(&scratch.0.join("state"), &["--help"]);
    assert!(String::from_utf8(top.stdout).unwrap().contains("practice"));
}

#[test]
fn a_path_that_is_not_a_repository_is_refused_by_name_before_anything_runs() {
    let scratch = Scratch::new("refused");
    let not_a_repository = scratch.0.join("empty");
    std::fs::create_dir_all(&not_a_repository).unwrap();
    let output = splinter(
        &scratch.0.join("state"),
        &[
            "practice",
            not_a_repository.to_str().unwrap(),
            "--rev",
            "HEAD",
        ],
    );
    assert_eq!(output.status.code(), Some(2), "{output:?}");
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("HEAD"), "{stderr}");
}

#[test]
fn the_kind_made_by_code_from_a_repository_is_not_one_a_model_is_asked_for() {
    let scratch = Scratch::new("kinds");
    let output = splinter(
        &scratch.0.join("state"),
        &[
            "tasks",
            "generate",
            "no-such-source",
            "--kinds",
            "workspace-change",
        ],
    );
    assert_eq!(output.status.code(), Some(2), "{output:?}");
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("unknown task kind \"workspace-change\"") && stderr.contains("recall"),
        "{stderr}"
    );
    assert!(
        !stderr
            .split("the kinds are")
            .nth(1)
            .unwrap_or("")
            .contains("workspace-change"),
        "the kinds a model is asked for are listed without it: {stderr}"
    );
}
