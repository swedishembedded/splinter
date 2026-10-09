// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: `session add` takes a recorded session in once and names what it
//! refuses and why; `session list`, `claims list`, `claims ledger` and `runs`
//! inspect the result as JSON; a claim set that does not exist is a refusal.
//! No model is loaded for any of this.

use std::path::Path;
use std::process::Command;

use serde_json::{json, Value};

type Outcome = Result<(), Box<dyn std::error::Error>>;

fn splinter(state: &Path, args: &[&str]) -> Result<(i32, Value), Box<dyn std::error::Error>> {
    let output = Command::new(env!("CARGO_BIN_EXE_splinter"))
        .arg("--json")
        .arg("--state")
        .arg(state)
        .args(args)
        .env_remove("SPLINTER_ALLOW_REMOTE")
        .output()?;
    let stdout = String::from_utf8(output.stdout)?;
    let json = serde_json::from_str(&stdout)
        .map_err(|e| format!("{args:?} printed no JSON ({e}): {stdout}"))?;
    Ok((output.status.code().unwrap_or(-1), json))
}

fn session(steps: Value) -> Value {
    json!({
        "schema_version": "ATIF-v1.7",
        "session_id": "monday",
        "agent": {"name": "scripted", "version": "1"},
        "steps": steps,
    })
}

fn step(id: u64, source: &str, message: &str) -> Value {
    json!({"step_id": id, "source": source, "message": message})
}

#[test]
fn sessions_are_taken_in_once_and_refusals_are_named() -> Outcome {
    let dir = tempfile::tempdir()?;
    let state = dir.path().join("state");
    let good = dir.path().join("good.atif.json");
    std::fs::write(
        &good,
        serde_json::to_vec(&session(json!([
            step(1, "user", "Which port does the dashboard use?"),
            step(2, "agent", "Port 8080."),
            step(3, "user", "No, it is 9090."),
            step(4, "agent", "Understood."),
        ])))?,
    )?;
    let good_arg = good.to_string_lossy().into_owned();

    let (code, first) = splinter(&state, &["session", "add", &good_arg])?;
    assert_eq!(code, 0, "{first}");
    assert_eq!(first["new"], 1);
    assert!(first["run"].as_str().is_some_and(|r| !r.is_empty()));
    let (_, again) = splinter(&state, &["session", "add", &good_arg])?;
    assert_eq!(again["new"], 0, "the same file again is a no-op");

    let (_, listed) = splinter(&state, &["session", "list"])?;
    assert_eq!(listed["sessions"].as_array().map(Vec::len), Some(1));
    let (_, runs) = splinter(&state, &["runs", "list"])?;
    assert!(runs["runs"]
        .as_array()
        .is_some_and(|r| r.iter().any(|run| run["command"] == "session add")));

    let broken = dir.path().join("broken.atif.json");
    std::fs::write(&broken, b"{ not json")?;
    let (code, refused) = splinter(&state, &["session", "add", &broken.to_string_lossy()])?;
    assert_eq!(code, 1, "a refused file fails the command: {refused}");
    assert_eq!(
        refused["refused"][0]["reason"]
            .as_str()
            .map(|r| r.contains("not an ATIF")),
        Some(true)
    );
    Ok(())
}

#[test]
fn the_claim_commands_inspect_the_stores_and_refuse_what_is_not_there() -> Outcome {
    let dir = tempfile::tempdir()?;
    let state = dir.path().join("state");
    let (_, sets) = splinter(&state, &["claims", "list"])?;
    assert_eq!(sets["claim_sets"].as_array().map(Vec::len), Some(0));
    let (_, ledger) = splinter(&state, &["claims", "ledger"])?;
    for part in ["live", "superseded", "refused"] {
        assert_eq!(ledger[part].as_array().map(Vec::len), Some(0), "{part}");
    }
    let (code, missing) = splinter(&state, &["claims", "gate", "blake3:abcdef012345"])?;
    assert_eq!(code, 2, "{missing}");
    assert_eq!(missing["refused"], true);
    Ok(())
}
