// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The server a run owns: started with the store, the adapter directory and
//! the training workload, reachable through endpoints that offer adapters by
//! writing them into the directory it serves, and gone when dropped.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::os::unix::fs::PermissionsExt;
use std::path::Path;

use splinter_model::service::{ManagedBrain, ServiceSettings};
use sven_sdk::CancelToken;

/// A stand-in `brain`: records its arguments, writes the key file and the
/// ready marker the way `brain serve` does, then waits to be stopped.
fn stand_in(dir: &Path) -> std::path::PathBuf {
    let script = dir.join("brain");
    std::fs::write(
        &script,
        r#"#!/bin/sh
echo "$@" > "$0.args"
while [ $# -gt 0 ]; do
  case "$1" in
    --api-keys-out) printf '{"openai":"sk-stand-in"}' > "$2"; shift ;;
    --ready-file) : > "$2"; shift ;;
  esac
  shift
done
exec sleep 600
"#,
    )
    .unwrap();
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
    script
}

fn settings(dir: &Path, binary: std::path::PathBuf) -> ServiceSettings {
    ServiceSettings {
        binary,
        models: dir.join("models"),
        state: dir.join("state"),
        device: Some("gpu1".into()),
        backend: None,
        max_batch: Some(24),
        serve_adapters: true,
        adapter_base: None,
    }
}

#[test]
fn the_server_a_run_owns_serves_the_store_the_adapters_and_training_and_is_asked_as_wide_as_it_decodes(
) {
    let dir = tempfile::tempdir().unwrap();
    let binary = stand_in(dir.path());

    let base = dir.path().join("models/Qwen/Qwen3-4B");
    let brain = ManagedBrain::start(
        &ServiceSettings {
            adapter_base: Some(base.clone()),
            ..settings(dir.path(), binary.clone())
        },
        &CancelToken::default(),
    )
    .unwrap();

    let args = std::fs::read_to_string(dir.path().join("brain.args")).unwrap();
    // The adapters sit on the base the run trains, on every layer it trains.
    let checkpoint = format!("--checkpoint {}", base.display());
    for wanted in [
        checkpoint.as_str(),
        "--adapter-targets all",
        "--device gpu1",
        "serve --openai 127.0.0.1:",
        "--max-batch 24",
        "--models-dir",
        "--serve-adapters",
        "--training",
        "--artifact-store",
    ] {
        assert!(args.contains(wanted), "{wanted} in {args}");
    }
    assert_eq!(brain.endpoints().key(), "sk-stand-in");
    assert_eq!(brain.endpoints().attempt_concurrency(), 24);
    assert!(brain.endpoints().artifact_store().is_ok());
    assert!(brain.endpoints().urls()[0].starts_with("http://127.0.0.1:"));
}

#[test]
fn dropping_the_server_removes_its_scratch_directory_and_keys() {
    let dir = tempfile::tempdir().unwrap();
    let binary = stand_in(dir.path());
    let brain =
        ManagedBrain::start(&settings(dir.path(), binary), &CancelToken::default()).unwrap();
    let run = dir
        .path()
        .join("state")
        .join(format!("run-{}", std::process::id()));
    assert!(run.join("keys.json").is_file());

    drop(brain);

    assert!(!run.exists(), "the keys do not outlive the server");
    assert!(
        dir.path().join("state/adapters").is_dir(),
        "the adapters stay"
    );
}

#[test]
fn a_binary_that_cannot_start_is_refused_by_name() {
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("no-such-brain");

    let refused = ManagedBrain::start(
        &settings(dir.path(), missing.clone()),
        &CancelToken::default(),
    )
    .err()
    .expect("refused")
    .to_string();

    assert!(refused.contains("no-such-brain"), "{refused}");
    assert!(!dir
        .path()
        .join("state")
        .join(format!("run-{}", std::process::id()))
        .exists());
}
