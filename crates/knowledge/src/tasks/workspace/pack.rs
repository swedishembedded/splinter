// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reproducible snapshots of source trees that
// agents are trained and measured on, for its clients. If your team needs
// expertise in turning a codebase into executable training tasks, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Archives made from a directory: the whole tree (a snapshot) or some of
//! its files (a hidden overlay), and an overlay laid over a tree.
//!
//! An archive is a function of the files it holds: entries are sorted by
//! name, owned by root and stamped with one fixed time, so the same files
//! are the same archive and so the same environment.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Duration;

use splinter_sandbox::host::{DEFAULT_LANG, DEFAULT_SEARCH_PATH};
use splinter_sandbox::process::{run, ProcessSpec};

use super::WorkspaceError;

/// How long making or laying out one archive may take.
const TAR_TIMEOUT: Duration = Duration::from_secs(120);

/// The most output kept of the archive tool.
const TAR_OUTPUT_CAP: usize = 64 * 1024;

/// How much of the tool's standard error an error carries.
const DETAIL_CHARS: usize = 300;

/// Archives all of `dir` except the entries named `exclude` (at any depth)
/// into the new file `out`.
pub(super) fn pack_dir(dir: &Path, exclude: &[&str], out: &Path) -> Result<(), WorkspaceError> {
    let options: Vec<String> = exclude.iter().map(|e| format!("--exclude={e}")).collect();
    tar_create(dir, out, options, vec![".".to_string()])
}

/// Archives the files `paths` (relative to `dir`) of `dir` into the new
/// file `out`.
pub(super) fn pack_files(dir: &Path, paths: &[String], out: &Path) -> Result<(), WorkspaceError> {
    let mut names = vec!["--".to_string()];
    names.extend(paths.iter().cloned());
    tar_create(dir, out, Vec::new(), names)
}

/// Lays the archive `archive` over `tree`, replacing the files it holds.
pub(super) fn lay_over(archive: &Path, tree: &Path) -> Result<(), WorkspaceError> {
    let argv = [
        "tar",
        "--extract",
        "--no-same-owner",
        "--no-same-permissions",
        "--file",
        &archive.to_string_lossy(),
        "--directory",
        &tree.to_string_lossy(),
    ]
    .map(String::from)
    .to_vec();
    tar(
        argv,
        tree,
        &format!("laying {} over the tree", archive.display()),
    )
}

fn tar_create(
    dir: &Path,
    out: &Path,
    options: Vec<String>,
    names: Vec<String>,
) -> Result<(), WorkspaceError> {
    let mut argv: Vec<String> = [
        "tar",
        "--create",
        "--format=gnu",
        "--sort=name",
        "--mtime=@0",
        "--owner=0",
        "--group=0",
        "--numeric-owner",
    ]
    .map(String::from)
    .to_vec();
    argv.push("--file".into());
    argv.push(out.to_string_lossy().into_owned());
    argv.push("--directory".into());
    argv.push(dir.to_string_lossy().into_owned());
    argv.extend(options);
    argv.extend(names);
    tar(argv, dir, &format!("archiving {}", dir.display()))
}

fn tar(argv: Vec<String>, cwd: &Path, what: &str) -> Result<(), WorkspaceError> {
    let env: BTreeMap<String, String> = [("PATH", DEFAULT_SEARCH_PATH), ("LANG", DEFAULT_LANG)]
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .into();
    let spec = ProcessSpec::new(argv, cwd, env, TAR_TIMEOUT, TAR_OUTPUT_CAP);
    let out = run(&spec).map_err(splinter_sandbox::SandboxError::from)?;
    if out.exit_code == Some(0) && !out.timed_out {
        return Ok(());
    }
    Err(WorkspaceError::Archive {
        what: what.to_string(),
        detail: format!(
            "tar ended with {:?}: {}",
            out.exit_code,
            String::from_utf8_lossy(&out.stderr.bytes)
                .chars()
                .take(DETAIL_CHARS)
                .collect::<String>()
        ),
    })
}
