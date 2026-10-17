// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Names the revision of the code in the binary, so a corpus manifest can say
//! which code made the corpus.

use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn main() {
    let revision = match git(&["rev-parse", "HEAD"]) {
        Some(head) if git(&["status", "--porcelain", "--", "."]).is_some_and(|s| !s.is_empty()) => {
            format!("{head}-dirty")
        }
        Some(head) => head,
        None => "unknown".to_string(),
    };
    println!("cargo:rustc-env=SYNTHETIC_AUTHOR_REVISION={revision}");
    println!("cargo:rerun-if-changed=src");
    // HEAD names the branch, which a commit does not change: the branch's
    // ref (loose or packed) and the index are what move, so a commit or a
    // staged edit names the new revision.
    let mut watched = vec![
        "HEAD".to_string(),
        "packed-refs".to_string(),
        "index".to_string(),
    ];
    watched.extend(git(&["symbolic-ref", "-q", "HEAD"]));
    for path in watched {
        if let Some(file) = git(&["rev-parse", "--git-path", &path]) {
            println!("cargo:rerun-if-changed={file}");
        }
    }
}
