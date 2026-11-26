// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reproducible-experiment infrastructure for
// learning systems, for its clients. If your team needs expertise in making
// training and evaluation runs rebuildable from their records, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Embeds the commit the binary is built from and whether its tree was dirty,
//! so a run can name them without running git. A tree made by `git archive`
//! (a workspace's source snapshot, which has no `.git`) names its commit in
//! `REVISION`, which the archive fills in; whether such a tree was changed
//! after it was archived is not known, so it embeds no dirty flag. Anywhere
//! else outside a git checkout nothing is embedded and the binary reports the
//! commit as unknown.
//!
//! A build against local sven and brain checkouts (`make deps/local`, which
//! writes the workspace's `.cargo/config.toml`) builds their working trees,
//! not the commits the lock pins: their HEADs and dirty flags are embedded
//! too, and they, not the lock, name what the binary was built from.

use std::process::Command;

include!("archived_revision.rs");

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// The checkouts `.cargo/config.toml` substitutes for the remotes
/// (`paths = ["<sven>", "<brain>"]`), by the package their SDK crate names.
fn local_checkouts(config: &str) -> Vec<(&'static str, String)> {
    let Some(line) = config.lines().find(|l| l.trim_start().starts_with("paths")) else {
        return Vec::new();
    };
    line.split('"')
        .skip(1)
        .step_by(2)
        .filter_map(|dir| {
            let manifest = std::fs::read_to_string(format!("{dir}/crates/sdk/Cargo.toml")).ok()?;
            // The package's name: the first `name = "..."` line, however it is aligned.
            let package = manifest.lines().find_map(|line| {
                let (key, value) = line.split_once('=')?;
                (key.trim() == "name").then(|| value.trim().trim_matches('"').to_owned())
            })?;
            let name = match package.as_str() {
                "sven-sdk" => "SVEN",
                "brain" => "BRAIN",
                _ => return None,
            };
            Some((name, dir.to_owned()))
        })
        .collect()
}

fn main() {
    // The local overrides, and the HEAD and index of each checkout they name.
    let config_path = "../../.cargo/config.toml";
    println!("cargo:rerun-if-changed={config_path}");
    let config = std::fs::read_to_string(config_path).unwrap_or_default();
    for (name, dir) in local_checkouts(&config) {
        let at = |args: &[&str]| {
            let mut full = vec!["-C", dir.as_str()];
            full.extend_from_slice(args);
            git(&full)
        };
        let mut watched = vec!["HEAD".to_string(), "index".to_string()];
        watched.extend(at(&["symbolic-ref", "-q", "HEAD"]));
        for watched in &watched {
            if let Some(path) = at(&["rev-parse", "--path-format=absolute", "--git-path", watched])
            {
                println!("cargo:rerun-if-changed={path}");
            }
        }
        if let Some(rev) = at(&["rev-parse", "HEAD"]) {
            println!("cargo:rustc-env=SPLINTER_{name}_LOCAL_REV={rev}");
            if let Some(status) = at(&["status", "--porcelain", "--untracked-files=no"]) {
                println!(
                    "cargo:rustc-env=SPLINTER_{name}_LOCAL_DIRTY={}",
                    !status.is_empty()
                );
            }
        }
    }
    // The compiler on PATH when the build ran, which is the one cargo picks
    // unless a build names another with RUSTC.
    if let Some(version) = Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .filter(|out| out.status.success())
    {
        let version = String::from_utf8_lossy(&version.stdout);
        println!("cargo:rustc-env=SPLINTER_RUSTC={}", version.trim());
    }
    println!("cargo:rerun-if-changed=../../Cargo.lock");
    println!("cargo:rerun-if-changed=../../crates");
    // The paths that move when HEAD, the current branch or the index do.
    let mut watched = vec!["HEAD".to_string(), "index".to_string()];
    if let Some(branch) = git(&["symbolic-ref", "-q", "HEAD"]) {
        watched.push(branch);
    }
    for name in watched {
        if let Some(path) = git(&["rev-parse", "--git-path", &name]) {
            println!("cargo:rerun-if-changed={path}");
        }
    }
    println!("cargo:rerun-if-changed=REVISION");
    let revision = std::fs::read_to_string("REVISION").unwrap_or_default();
    if let Some(rev) = archived_revision(&revision) {
        println!("cargo:rustc-env=SPLINTER_GIT_REV={rev}");
        return;
    }
    let Some(rev) = git(&["rev-parse", "HEAD"]) else {
        return;
    };
    println!("cargo:rustc-env=SPLINTER_GIT_REV={rev}");
    if let Some(status) = git(&["status", "--porcelain", "--untracked-files=no"]) {
        println!("cargo:rustc-env=SPLINTER_GIT_DIRTY={}", !status.is_empty());
    }
}
