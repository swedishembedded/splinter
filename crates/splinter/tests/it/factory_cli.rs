// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning pipelines whose first steps work on
// a machine nobody configured, for its clients. If your team needs expertise
// in making a new user's first run work, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Spec: the factory's commands, run as a new user runs them - the real
//! binary, an empty home, no environment - in a task repository: a recipe
//! validates with the connections the repository ships; a model no
//! connections file reaches is refused naming the files read; a file that is
//! not a recipe says what it is; and every command given a run directory that
//! is not there says so, exits non-zero and creates nothing.

#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};
use std::process::Command;

/// An empty home holding a task repository of one task.
struct Repository {
    home: PathBuf,
    root: PathBuf,
}

fn write(path: &Path, text: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

impl Repository {
    fn new(name: &str) -> Self {
        let home = std::env::temp_dir().join(format!(
            "splinter-factory-cli-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&home);
        let root = home.join("bench");
        let task = root.join("tasks/demo/train/001-one");
        write(&task.join("task.toml"), "[task]\nname = \"demo/one\"\n");
        write(&task.join("instruction.md"), "Do it.\n");
        write(&task.join("tests/test.sh"), "exit 0\n");
        write(
            &root.join("collections/demo.yaml"),
            "schema: splinter.collection/v1\nname: demo\nroot: tasks/demo\n",
        );
        write(
            &root.join("environments/demo.yaml"),
            "schema: splinter.environment/v1\nnetwork: { mode: model-gateway-only }\n",
        );
        // seed: a recipe states its pass's seed; nothing runs here.
        write(
            &root.join("recipes/demo.yaml"),
            "schema: splinter.factory/v1\nname: demo\ntasks: { collection: collections/demo.yaml, split: train }\n\
             rollouts:\n  - { name: hosted, agent: sven, model: \"hosted:some/model\", \
             environment: environments/demo.yaml, attempts: 1, seed: 1 }\n",
        );
        Self { home, root }
    }

    /// `splinter args...` in the repository with this home and nothing else.
    fn run(&self, args: &[&str]) -> (i32, String) {
        let output = Command::new(env!("CARGO_BIN_EXE_splinter"))
            .args(args)
            .env_clear()
            .env("HOME", &self.home)
            .current_dir(&self.root)
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&output.stdout).into_owned()
            + &String::from_utf8_lossy(&output.stderr);
        (output.status.code().unwrap_or(-1), text)
    }
}

impl Drop for Repository {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.home);
    }
}

#[test]
fn a_recipe_validates_with_the_connections_its_repository_ships() {
    let repo = Repository::new("connections");
    let (code, text) = repo.run(&["validate", "recipes/demo.yaml"]);
    assert_ne!(code, 0, "{text}");
    assert!(
        text.contains("hosted:some/model") && text.contains("connections.yaml"),
        "the refusal names where connections were looked for: {text}"
    );
    write(
        &repo.root.join("connections.yaml"),
        "connections:\n  hosted: { protocol: openai, url: \"https://example.invalid/v1\", token: \"env:KEY\" }\n",
    );
    let (code, text) = repo.run(&["validate", "recipes/demo.yaml"]);
    assert_eq!(code, 0, "{text}");
}

#[test]
fn a_file_that_is_not_a_recipe_says_what_it_is() {
    let repo = Repository::new("not-a-recipe");
    let (code, text) = repo.run(&["validate", "collections/demo.yaml"]);
    assert_ne!(code, 0);
    assert!(
        text.contains("splinter.collection/v1 file, not a recipe"),
        "{text}"
    );
}

#[test]
fn a_run_directory_that_is_not_there_is_refused_and_nothing_is_created() {
    let repo = Repository::new("no-run");
    for command in [
        vec!["status", "runs/none"],
        vec!["inspect", "runs/none"],
        vec!["resume", "runs/none"],
        vec!["cancel", "runs/none"],
        vec!["report", "runs/none"],
        vec!["verify", "runs/none"],
        vec!["export", "runs/none", "--out", "x.tar.zst"],
        vec!["index", "runs/none"],
    ] {
        let (code, text) = repo.run(&command);
        assert_ne!(code, 0, "{command:?}: {text}");
        assert!(
            text.contains("runs/none does not exist"),
            "{command:?}: {text}"
        );
    }
    assert!(!repo.root.join("runs").exists(), "no command created a run");
}
