// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning pipelines whose documented first
// steps are tested, not trusted, for its clients. If your team needs expertise
// in making a new user's first run work the way the README says, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: the commands the first-learning-run page documents, run as a new user
//! would run them - the real binary, an empty home, no environment
//! configuration - do what the page says, and the output the page quotes is
//! the output they print. The commands are read from the page, so it cannot
//! show a command or an output the program does not produce. No model is loaded
//! and no weights are needed, so this runs anywhere; the steps that need a GPU
//! are not here.

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

/// The repository root: the commands on the page are written to run from it.
fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The walk-through page.
fn page() -> String {
    // doc-reference: the test fails when the page moves, so it cannot drift from the program
    let path = repository_root().join("docs/getting-started/first-learning-run.md");
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The lines of the page's fenced blocks of one language.
fn block_lines(page: &str, language: &str) -> Vec<String> {
    let mut lines = Vec::new();
    let mut inside = false;
    for line in page.lines() {
        if let Some(info) = line.strip_prefix("```") {
            inside = !inside && info.trim() == language;
        } else if inside {
            lines.push(line.to_string());
        }
    }
    lines
}

/// The one command the page shows that starts `splinter` and contains `fragment`.
fn page_command(fragment: &str) -> String {
    let commands: Vec<String> = block_lines(&page(), "bash")
        .into_iter()
        .filter(|l| l.starts_with("splinter ") && l.contains(fragment))
        .collect();
    assert_eq!(commands.len(), 1, "{fragment:?} in {commands:?}");
    commands[0].clone()
}

/// A line the page quotes as output (a `text` block) that starts with `prefix`.
fn quoted_output(prefix: &str) -> String {
    let found: Vec<String> = block_lines(&page(), "text")
        .into_iter()
        .filter(|l| l.trim_start().starts_with(prefix))
        .collect();
    assert_eq!(found.len(), 1, "{prefix:?} in {found:?}");
    found[0].clone()
}

impl Home {
    /// A command from the page, run from the repository root with its state in
    /// this home and `flags` after the program name; its exit code, stdout and
    /// stderr.
    fn run_page(&self, command: &str, flags: &str) -> (i32, String, String) {
        let program = format!(
            "{} {flags} --state {}/state",
            env!("CARGO_BIN_EXE_splinter"),
            self.0.display()
        );
        let output = Command::new("sh")
            .arg("-c")
            .arg(command.replacen("splinter", &program, 1))
            .env("HOME", &self.0)
            .env_remove("SPLINTER_STATE")
            .env_remove("BRAIN_MODELS_DIR")
            .env_remove("BRAIN_QWEN_WEIGHTS")
            .env_remove("SPLINTER_ALLOW_REMOTE")
            .current_dir(repository_root())
            .output()
            .unwrap();
        (
            output.status.code().unwrap_or(-1),
            String::from_utf8(output.stdout).unwrap(),
            String::from_utf8(output.stderr).unwrap(),
        )
    }
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

/// Step 1: the failure the page quotes for an empty model store is what the
/// page's `doctor` command prints, down to the fix.
#[test]
fn doctor_prints_the_failure_the_page_quotes() {
    let home = Home::new("doctor-text");
    let (code, stdout, stderr) = home.run_page(&page_command("doctor"), "");
    assert_eq!(code, 1, "{stdout}{stderr}");
    let failure = quoted_output("FAIL role:policy:");
    let needs = "FAIL role:policy: policy:default needs ";
    assert!(failure.starts_with(needs), "{failure}");
    assert!(stdout.lines().any(|l| l.starts_with(needs)), "{stdout}");
    for exact in [quoted_output("fix:"), quoted_output("not ready")] {
        assert!(
            stdout.lines().any(|l| l == exact),
            "{exact:?} is not in {stdout}"
        );
    }
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

/// Step 2: the anchor suite freezes from the file the page names, into the
/// twenty tasks and the version the page quotes.
#[test]
fn the_shipped_anchor_suite_freezes_into_the_suite_the_page_quotes() {
    let home = Home::new("anchor");
    let command = page_command("--freeze");
    let (code, stdout, stderr) = home.run_page(&command, "--json");
    assert_eq!(code, 0, "{stdout}\n{stderr}");
    let frozen: Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(frozen["anchor"]["tasks"], 20, "{frozen}");
    assert_eq!(frozen["anchor"]["version"], 1, "{frozen}");

    let (code, stdout, stderr) = home.run_page(&command, "");
    assert_eq!(code, 0, "{stdout}\n{stderr}");
    let quoted = quoted_output("anchor suite version");
    assert_eq!(stdout.lines().next(), Some(quoted.as_str()), "{stdout}");
}

/// Step 5: `learn --dry-run` on the shipped manual plans the run, prints the
/// stages the page quotes and writes nothing under the state directory.
#[test]
fn a_dry_run_on_the_shipped_manual_plans_and_writes_nothing() {
    let home = Home::new("dry-run");
    let command = page_command("--dry-run");
    let (code, stdout, stderr) = home.run_page(&command, "");
    assert_eq!(code, 0, "{stdout}\n{stderr}");
    let stages = quoted_output("stages:");
    assert!(
        stdout.lines().any(|l| l == stages),
        "{stages:?} is not in {stdout}"
    );
    assert!(
        stdout.starts_with("dry run - nothing written under "),
        "{stdout}"
    );
    assert!(
        !home.0.join("state").exists() && !home.0.join(".splinter").exists(),
        "a dry run writes nothing"
    );

    let (code, plan, stderr) = home.run_page(&command, "--json");
    assert_eq!(code, 0, "{plan}\n{stderr}");
    let plan: Value = serde_json::from_str(&plan).unwrap();
    assert_eq!(plan["dry_run"], true);
    assert_eq!(plan["kinds"], serde_json::json!(["denoise"]));
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
