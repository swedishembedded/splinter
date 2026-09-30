// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: with `--json`, a command prints one JSON document on stdout whose
//! fields are the ones the README documents, and a refusal prints
//! `{"error", "refused"}` and exits 2. No model is loaded for any of this.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!("splinter-cli-{name}-{}", std::process::id()));
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

/// Runs `splinter --json --state <state> args...`; its exit code and stdout
/// as JSON.
fn splinter(state: &Path, args: &[&str]) -> (i32, Value) {
    let output = Command::new(env!("CARGO_BIN_EXE_splinter"))
        .arg("--json")
        .arg("--state")
        .arg(state)
        .args(args)
        .env_remove("SPLINTER_ALLOW_REMOTE")
        .output()
        .unwrap();
    let stdout = String::from_utf8(output.stdout).unwrap();
    let json = serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("{args:?} printed no JSON ({e}): {stdout}"));
    (output.status.code().unwrap_or(-1), json)
}

fn keys(value: &Value) -> Vec<&str> {
    let mut keys: Vec<&str> = value
        .as_object()
        .unwrap_or_else(|| panic!("{value} is not an object"))
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();
    keys
}

#[test]
fn status_and_source_list_print_their_documented_fields() {
    let scratch = Scratch::new("json");
    let state = scratch.0.join("state");
    let doc = scratch.0.join("manual.md");
    std::fs::write(&doc, "# Manual\n\nThe console runs at 115200 baud.\n").unwrap();

    let (code, added) = splinter(&state, &["source", "add", doc.to_str().unwrap()]);
    assert_eq!(code, 0, "{added}");
    assert_eq!(keys(&added), ["new", "run", "source"]);
    assert_eq!(added["new"], true);

    let (code, list) = splinter(&state, &["source", "list"]);
    assert_eq!(code, 0);
    assert_eq!(keys(&list), ["sources"]);
    let source = &list["sources"][0];
    assert_eq!(
        keys(source),
        ["bytes", "captured_at", "id", "kind", "origin", "parts"]
    );
    assert_eq!(source["kind"], "document");
    assert_eq!(source["origin"]["kind"], "document");
    assert_eq!(source["parts"], 1);
    assert!(source["id"].as_str().unwrap().starts_with("sha256:"));

    let (code, status) = splinter(&state, &["status"]);
    assert_eq!(code, 0, "{status}");
    assert_eq!(keys(&status), ["counts", "policy", "recent_runs", "state"]);
    assert_eq!(
        keys(&status["policy"]),
        ["adapter", "base", "model", "reference", "release"]
    );
    assert_eq!(status["policy"]["reference"], "policy:default");
    assert_eq!(status["policy"]["adapter"], Value::Null);
    assert_eq!(status["policy"]["release"], Value::Null);
    assert_eq!(
        keys(&status["counts"]),
        [
            "candidates",
            "datasets",
            "experience_sets",
            "experiences",
            "sources",
            "task_sets",
            "tasks"
        ]
    );
    assert_eq!(status["counts"]["sources"], 1);
    let run = &status["recent_runs"][0];
    assert_eq!(
        keys(run),
        ["command", "id", "started_at", "status", "updated_at"]
    );
    assert_eq!(run["command"], "source add");
    assert_eq!(run["status"], "completed");
}

#[test]
fn a_refusal_is_json_and_exits_two() {
    let scratch = Scratch::new("refusal");
    let state = scratch.0.join("state");
    let (code, refused) = splinter(&state, &["source", "add", "https://example.com/doc.md"]);
    assert_eq!(code, 2);
    assert_eq!(keys(&refused), ["error", "refused"]);
    assert_eq!(refused["refused"], true);

    let (code, remote) = splinter(
        &state,
        &["ask", "q", "--policy", "remote:openrouter/z-ai/glm"],
    );
    assert_eq!(code, 2, "{remote}");
    assert!(remote["error"].as_str().unwrap().contains("--allow-remote"));
}

#[test]
fn releases_rollback_and_the_anchor_suite_speak_json() {
    let scratch = Scratch::new("releases");
    let state = scratch.0.join("state");

    let (code, list) = splinter(&state, &["release", "list"]);
    assert_eq!(code, 0, "{list}");
    assert_eq!(keys(&list), ["releases"]);
    assert_eq!(list["releases"], serde_json::json!([]));

    let (code, refused) = splinter(&state, &["rollback", "default"]);
    assert_eq!(code, 2, "{refused}");
    assert!(refused["error"].as_str().unwrap().contains("no release"));

    let (code, refused) = splinter(&state, &["release", "candidate-none"]);
    assert_eq!(code, 2, "{refused}");

    let anchor = scratch.0.join("anchor.jsonl");
    std::fs::write(
        &anchor,
        "{\"instruction\": \"What is 2 + 2?\", \"reference\": \"4\"}\n",
    )
    .unwrap();
    let (code, frozen) = splinter(
        &state,
        &[
            "eval",
            "--suite",
            "anchor",
            "--freeze",
            anchor.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0, "{frozen}");
    assert_eq!(
        keys(&frozen),
        ["anchor", "model", "reference", "run", "scores"]
    );
    assert_eq!(keys(&frozen["anchor"]), ["digest", "tasks", "version"]);
    assert_eq!(frozen["anchor"]["version"], 1);

    let (code, shown) = splinter(&state, &["eval", "--suite", "anchor"]);
    assert_eq!(code, 0, "{shown}");
    assert_eq!(shown["anchor"]["digest"], frozen["anchor"]["digest"]);
}
