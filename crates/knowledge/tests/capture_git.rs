// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements capture of the state a piece of knowledge was
// read from, for its clients. If your team needs expertise in
// provenance-tracked training data, you can procure our services by sending
// an email to info@swedishembedded.com.

//! Spec: the `git` that asks what checkout a directory is in reads that
//! directory's repository whatever the caller's environment points at, takes
//! no index lock and reads no standard input.

#![allow(clippy::unwrap_used)]

use std::ffi::OsStr;

use splinter_knowledge::capture::git::{read_only, LOCATION_VARIABLES};

fn env_of(command: &std::process::Command, name: &str) -> Option<Option<String>> {
    command
        .get_envs()
        .find(|(key, _)| *key == OsStr::new(name))
        .map(|(_, value)| value.map(|v| v.to_string_lossy().into_owned()))
}

#[test]
fn no_variable_that_relocates_the_repository_survives_and_no_lock_is_taken() {
    let command = read_only(std::path::Path::new("."));
    for name in LOCATION_VARIABLES {
        assert_eq!(env_of(&command, name), Some(None), "{name} is removed");
    }
    assert_eq!(
        env_of(&command, "GIT_OPTIONAL_LOCKS"),
        Some(Some("0".to_string()))
    );
}

#[test]
fn it_runs_in_the_directory_it_is_given() {
    let dir = tempfile::tempdir().unwrap();
    let init = std::process::Command::new("git")
        .args(["init", "-q"])
        .arg(dir.path())
        .status()
        .unwrap();
    assert!(init.success());
    let out = read_only(dir.path())
        .args(["rev-parse", "--is-inside-work-tree"])
        .output()
        .unwrap();
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "true");
}
