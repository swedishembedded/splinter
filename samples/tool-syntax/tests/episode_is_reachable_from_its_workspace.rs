// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! Spec: an episode's world is reachable from the workspace whatever directory
//! the run was named by.
//!
//! The agent's tools run with the workspace as their working directory, and
//! they are told where the service is by an environment variable. A socket
//! path that was relative to the process the episode was started from points
//! nowhere once the tools run elsewhere: `svctl status` answers "cannot reach
//! the upload service" and a demonstration recorded from it teaches a model
//! that the service is unreachable - while its verifier, which looks from
//! outside, still reports the task solved.

// A test file: its fixture helpers panic on a broken fixture, which is the
// failure report.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::path::Path;
use std::process::Command;

use splinter_tool_syntax::{Episode, Family};

fn family() -> Family {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("the repository root")
        .join("tasks/config-discovery");
    Family::load(&root).expect("the family loads")
}

#[test]
fn a_run_dir_too_deep_for_a_unix_socket_is_refused_by_name() {
    // A socket path holds 107 bytes. Longer, the world's bind fails and the
    // only symptom is "did not create its socket within 15s", with no hint
    // that the directory the run was named by is the cause.
    let deep = std::env::temp_dir().join("d".repeat(120));
    let family = family();
    let err = Episode::start(&family, &deep, &family.live_choices()[0])
        .err()
        .expect("an unbindable socket path must be refused");
    std::fs::remove_dir_all(&deep).ok();
    assert!(err.to_string().contains("socket path"), "{err}");
}

#[test]
fn the_workspaces_own_cli_reaches_the_service_when_the_run_dir_is_relative() {
    let family = family();

    // Relative to the directory this test runs in.
    let relative = std::path::PathBuf::from(format!("episode-relative-{}", std::process::id()));
    let mut episode =
        Episode::start(&family, &relative, &family.live_choices()[0]).expect("the episode starts");

    let mut svctl = Command::new("./svctl");
    svctl.arg("status").current_dir(episode.workspace());
    for (key, value) in episode.agent_env() {
        svctl.env(key, value);
    }
    let output = svctl.output().expect("svctl runs");
    episode.stop();
    std::fs::remove_dir_all(&relative).ok();

    assert!(
        output.status.success(),
        "svctl could not reach the service: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}
