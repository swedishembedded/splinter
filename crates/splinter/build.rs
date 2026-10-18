// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reproducible-experiment infrastructure for
// learning systems, for its clients. If your team needs expertise in making
// training and evaluation runs rebuildable from their records, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Embeds the commit the binary is built from and whether its tree was dirty,
//! so a run can name them without running git. Outside a git checkout nothing
//! is embedded and the binary reports the commit as unknown.

use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn main() {
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
    let Some(rev) = git(&["rev-parse", "HEAD"]) else {
        return;
    };
    println!("cargo:rustc-env=SPLINTER_GIT_REV={rev}");
    if let Some(status) = git(&["status", "--porcelain", "--untracked-files=no"]) {
        println!("cargo:rustc-env=SPLINTER_GIT_DIRTY={}", !status.is_empty());
    }
}
