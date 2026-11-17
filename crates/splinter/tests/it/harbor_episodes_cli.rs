// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that turn practice task folders
// into graded, replayable agent experience for its clients. If your team
// needs expertise in on-policy experience collection for agent training you
// can procure our services by sending an email to info@swedishembedded.com.

//! Spec: `splinter tasks harbor episodes` is a verb with its own help, and
//! a collection holding exam material or a malformed task folder is refused
//! by name before any policy is asked for anything.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "splinter-episodes-cli-{name}-{}",
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

fn splinter(state: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_splinter"))
        .arg("--state")
        .arg(state)
        .args(args)
        .output()
        .unwrap()
}

fn write(path: &Path, text: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

/// A practice task folder; `metadata` is the rest of its `[metadata]`.
fn task(root: &Path, name: &str, metadata: &str) -> PathBuf {
    let dir = root.join(name);
    write(&dir.join("workspace/greeting.txt"), "hello\n");
    write(
        &dir.join("instruction.md"),
        "Write hello world into greeting.txt.\n",
    );
    write(
        &dir.join("task.toml"),
        &format!("schema_version = \"1.4\"\n[metadata]\n{metadata}"),
    );
    write(
        &dir.join("tests/test.sh"),
        "#!/bin/bash\ngrep -q 'hello world' \"$WORKSPACE_DIR/greeting.txt\" && echo 1 > \"$LOGS_DIR/reward.txt\" || echo 0 > \"$LOGS_DIR/reward.txt\"\n",
    );
    dir
}

#[test]
fn episodes_help_renders_the_verb_and_what_it_takes() {
    let scratch = Scratch::new("help");
    let output = splinter(
        &scratch.0.join("state"),
        &["tasks", "harbor", "episodes", "--help"],
    );
    assert!(output.status.success(), "{output:?}");
    let help = String::from_utf8(output.stdout).unwrap();
    for wanted in [
        "Usage: splinter tasks harbor episodes",
        "<DIR>",
        "--policy <REF>",
        "--attempts <N>",
        "--seed <SEED>",
        "--jobs <N>",
        "--deadline <DUR>",
        "--max-tool-calls <N>",
        "--max-output-tokens <N>",
        "--out <FILE>",
    ] {
        assert!(help.contains(wanted), "{wanted:?} is not in:\n{help}");
    }
}

#[test]
fn exam_material_and_a_malformed_folder_are_refused_before_anything_runs() {
    let scratch = Scratch::new("refused");
    let state = scratch.0.join("state");
    let sealed = scratch.0.join("sealed");
    task(&sealed, "greet", "bench_sealed = true\n");
    let output = splinter(
        &state,
        &[
            "tasks",
            "harbor",
            "episodes",
            sealed.to_str().unwrap(),
            "--seed",
            "7",
        ],
    );
    assert_eq!(output.status.code(), Some(2), "{output:?}");
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("sealed exam material"), "{stderr}");

    let malformed = scratch.0.join("malformed");
    let dir = task(&malformed, "greet", "");
    std::fs::remove_file(dir.join("tests/test.sh")).unwrap();
    let output = splinter(
        &state,
        &[
            "tasks",
            "harbor",
            "episodes",
            malformed.to_str().unwrap(),
            "--seed",
            "7",
        ],
    );
    assert_eq!(output.status.code(), Some(2), "{output:?}");
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("tests/test.sh"), "{stderr}");
}
