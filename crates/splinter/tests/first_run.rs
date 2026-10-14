// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning pipelines whose documented first
// steps are tested, not trusted, for its clients. If your team needs expertise
// in making a new user's first run work the way the README says, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: the commands the first-run example documents, run as a new user would
//! run them - the real binary, an empty home, no environment configuration -
//! do what the README says. No model is loaded and no weights are needed, so
//! this runs anywhere; the steps that need a GPU are not here.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

/// A scratch directory standing in for a user's whole home.
struct Home(PathBuf);

impl Home {
    fn new(name: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("splinter-first-run-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    /// `splinter --json args...` with this home and none of the variables a
    /// configured machine would set; its exit code, stdout and stderr.
    fn run(&self, args: &[&str]) -> (i32, String, String) {
        let output = Command::new(env!("CARGO_BIN_EXE_splinter"))
            .arg("--json")
            .args(args)
            .env("HOME", &self.0)
            .env_remove("SPLINTER_STATE")
            .env_remove("BRAIN_MODELS_DIR")
            .env_remove("BRAIN_QWEN_WEIGHTS")
            .env_remove("SPLINTER_ALLOW_REMOTE")
            .current_dir(&self.0)
            .output()
            .unwrap();
        (
            output.status.code().unwrap_or(-1),
            String::from_utf8(output.stdout).unwrap(),
            String::from_utf8(output.stderr).unwrap(),
        )
    }

    fn json(&self, args: &[&str]) -> (i32, Value) {
        let (code, stdout, stderr) = self.run(args);
        let json = serde_json::from_str(&stdout)
            .unwrap_or_else(|e| panic!("{args:?} printed no JSON ({e}): {stdout}\n{stderr}"));
        (code, json)
    }
}

impl Drop for Home {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn example(file: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/first-run")
        .join(file)
}

fn check<'a>(report: &'a Value, name: &str) -> &'a Value {
    report["checks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == name)
        .unwrap_or_else(|| panic!("no check {name} in {report}"))
}

/// Step 1: on a machine with no models, `doctor` says what is missing and how
/// to fix it, exits 1, and leaves the home as it found it.
#[test]
fn doctor_on_an_empty_home_fails_with_the_fix_and_creates_nothing() {
    let home = Home::new("doctor-empty");
    let (code, report) = home.json(&["doctor"]);
    assert_eq!(code, 1, "{report}");
    assert_eq!(check(&report, "state-writable")["verdict"], "ok");
    let policy = check(&report, "role:policy");
    assert_eq!(policy["verdict"], "fail");
    assert!(policy["fix"].as_str().unwrap().len() > 10, "{policy}");
    assert!(
        !home.0.join(".splinter").exists(),
        "checking creates nothing"
    );
}

/// The state lives in `~/.splinter`; state left in the former location by an
/// earlier version is used until `~/.splinter` exists.
#[test]
fn the_default_state_directory_is_splinters_own_and_the_former_one_is_kept_until_then() {
    let home = Home::new("state");
    let former = home.0.join(".sven/splinter");
    std::fs::create_dir_all(&former).unwrap();
    let (_, report) = home.json(&["doctor"]);
    let detail = check(&report, "state-writable")["detail"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(detail.contains(".sven/splinter"), "{detail}");

    std::fs::create_dir_all(home.0.join(".splinter")).unwrap();
    let (_, report) = home.json(&["doctor"]);
    let detail = check(&report, "state-writable")["detail"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(
        detail.contains(".splinter") && !detail.contains(".sven"),
        "{detail}"
    );
}

/// Step 2: the anchor suite freezes from the file the example ships.
#[test]
fn the_shipped_anchor_suite_freezes_into_twenty_tasks() {
    let home = Home::new("anchor");
    let anchor = example("anchor.jsonl");
    let (code, stdout, stderr) = home.run(&[
        "eval",
        "--suite",
        "anchor",
        "--freeze",
        anchor.to_str().unwrap(),
    ]);
    assert_eq!(code, 0, "{stdout}\n{stderr}");
    let frozen: Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(frozen["anchor"]["tasks"], 20, "{frozen}");
    assert_eq!(frozen["anchor"]["version"], 1, "{frozen}");
}

/// Step 3: `learn --dry-run` on the shipped manual plans the run and writes
/// nothing under the state directory.
#[test]
fn a_dry_run_on_the_shipped_manual_plans_and_writes_nothing() {
    let home = Home::new("dry-run");
    let manual = example("manual.md");
    let (code, plan) = home.json(&[
        "learn",
        manual.to_str().unwrap(),
        "--kinds",
        "denoise",
        "--dry-run",
    ]);
    assert_eq!(code, 0, "{plan}");
    assert_eq!(plan["dry_run"], true);
    assert_eq!(plan["kinds"], serde_json::json!(["denoise"]));
    assert!(
        !home.0.join(".splinter").exists(),
        "a dry run writes nothing"
    );
}

/// The help tells a newcomer where to start.
#[test]
fn the_help_says_where_to_start() {
    let home = Home::new("help");
    let (code, stdout, _) = home.run(&["--help"]);
    assert_eq!(code, 0);
    assert!(
        stdout.contains("splinter doctor") && stdout.contains("examples/first-run"),
        "{stdout}"
    );
}
