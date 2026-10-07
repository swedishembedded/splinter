// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements self-improving coding agents whose every
// step is auditable, for its clients. If your team needs expertise in agent
// evaluation or locally operated coding agents, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The checks that decide whether a candidate is accepted.
//!
//! Acceptance commands are the supervisor's: they are named in the
//! contract, run by the loop in the candidate's checkout after the worker
//! has stopped, and live outside what the worker may edit (a command may
//! point at a script anywhere on the machine; the files the worker must not
//! change are listed as protected). A command that does not finish within
//! its time is killed and counts as a failure.

use std::io::Read;
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::redact::redact;

/// The most output of one stream kept for the record.
const MAX_CAPTURE: usize = 64 * 1024;

/// One named check.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Check {
    /// What the check is called in the record.
    pub name: String,
    /// The shell command, run in the candidate's checkout.
    pub command: String,
    /// Seconds it may take.
    pub timeout_secs: u64,
    /// Whether the worker is shown the command. A hidden check is named in
    /// the prompt and nothing more: its command and the script it runs stay
    /// out of the worker's sight.
    #[serde(default)]
    pub visible: bool,
}

impl Check {
    /// `NAME=COMMAND`, as the command line gives it.
    pub fn parse(text: &str, timeout_secs: u64, visible: bool) -> Result<Self> {
        let (name, command) = text
            .split_once('=')
            .filter(|(n, c)| !n.trim().is_empty() && !c.trim().is_empty())
            .with_context(|| format!("an acceptance check is NAME=COMMAND, not {text:?}"))?;
        Ok(Self {
            name: name.trim().to_string(),
            command: command.trim().to_string(),
            timeout_secs,
            visible,
        })
    }
}

/// What one check did.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CheckResult {
    /// The check's name.
    pub name: String,
    /// The command that ran.
    pub command: String,
    /// Whether it exited 0 within its time.
    pub passed: bool,
    /// The exit code; `None` when it was killed or could not start.
    pub exit_code: Option<i32>,
    /// Whether it was killed for taking too long.
    pub timed_out: bool,
    /// Wall-clock milliseconds.
    pub duration_ms: u64,
    /// The last of standard output, redacted.
    pub stdout_tail: String,
    /// The last of standard error, redacted.
    pub stderr_tail: String,
}

/// Reads `stream` to the end on its own thread, keeping `MAX_CAPTURE` bytes.
fn capture(mut stream: impl Read + Send + 'static) -> std::thread::JoinHandle<String> {
    std::thread::spawn(move || {
        let mut kept = Vec::new();
        let mut chunk = [0u8; 8192];
        while let Ok(n) = stream.read(&mut chunk) {
            if n == 0 {
                break;
            }
            kept.extend_from_slice(&chunk[..n]);
            if kept.len() > MAX_CAPTURE {
                kept.drain(..kept.len() - MAX_CAPTURE);
            }
        }
        String::from_utf8_lossy(&kept).into_owned()
    })
}

/// Kills every process in the process group led by `leader`: a command that
/// started others must not leave them holding its output open.
fn kill_group(leader: u32) {
    let _ = Command::new("kill")
        .args(["-KILL", "--", &format!("-{leader}")])
        .stderr(Stdio::null())
        .status();
}

/// Removes the Python bytecode caches under `dir`. A cache written by an
/// earlier run is keyed by the source's modification time and size, so a
/// file edited within the same second to the same length would be run from
/// its old bytecode, and the check would judge code that is no longer there.
fn clear_bytecode(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        if !path.is_dir() || path.is_symlink() || name == ".git" {
            continue;
        }
        if name == "__pycache__" {
            let _ = std::fs::remove_dir_all(&path);
        } else {
            clear_bytecode(&path);
        }
    }
}

/// Runs `check` in `dir`.
pub fn run_check(check: &Check, dir: &Path) -> CheckResult {
    clear_bytecode(dir);
    let started = Instant::now();
    let finish = |passed, exit_code, timed_out, stdout: String, stderr: String| CheckResult {
        name: check.name.clone(),
        command: check.command.clone(),
        passed,
        exit_code,
        timed_out,
        duration_ms: u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
        stdout_tail: redact(&stdout),
        stderr_tail: redact(&stderr),
    };
    let spawned = Command::new("sh")
        .arg("-c")
        .arg(&check.command)
        .current_dir(dir)
        // No bytecode is written for the next check to trust.
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // Its own process group, so a timeout can end everything it started.
        .process_group(0)
        .spawn();
    let mut child = match spawned {
        Ok(child) => child,
        Err(e) => {
            return finish(
                false,
                None,
                false,
                String::new(),
                format!("cannot start: {e}"),
            )
        }
    };
    let out = child.stdout.take().map(capture);
    let err = child.stderr.take().map(capture);
    let limit = Duration::from_secs(check.timeout_secs);
    let (code, timed_out) = loop {
        match child.try_wait() {
            Ok(Some(status)) => break (status.code(), false),
            Ok(None) if started.elapsed() >= limit => {
                kill_group(child.id());
                let _ = child.kill();
                let _ = child.wait();
                break (None, true);
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(25)),
            Err(_) => break (None, false),
        }
    };
    let stdout = out.and_then(|h| h.join().ok()).unwrap_or_default();
    let stderr = err.and_then(|h| h.join().ok()).unwrap_or_default();
    finish(
        code == Some(0) && !timed_out,
        code,
        timed_out,
        stdout,
        stderr,
    )
}

/// Runs every check, in order, and reports each: a later check still runs
/// after an earlier one fails, so the record shows everything that is wrong.
pub fn run_all(checks: &[Check], dir: &Path) -> Vec<CheckResult> {
    checks.iter().map(|c| run_check(c, dir)).collect()
}

/// The last `lines` lines of `text`.
#[must_use]
pub fn tail_lines(text: &str, lines: usize) -> String {
    let all: Vec<&str> = text.lines().collect();
    all[all.len().saturating_sub(lines)..].join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(command: &str, timeout_secs: u64) -> Check {
        Check {
            name: "c".into(),
            command: command.into(),
            timeout_secs,
            visible: true,
        }
    }

    #[test]
    fn a_command_is_judged_by_its_exit_code_and_its_output_is_kept() {
        let dir = tempfile::tempdir().unwrap();
        let ok = run_check(&check("echo fine", 5), dir.path());
        assert!(ok.passed && ok.exit_code == Some(0) && ok.stdout_tail.contains("fine"));
        let bad = run_check(&check("echo broken >&2; exit 3", 5), dir.path());
        assert!(!bad.passed && bad.exit_code == Some(3) && bad.stderr_tail.contains("broken"));
    }

    #[test]
    fn a_command_that_runs_too_long_is_killed_and_fails() {
        let dir = tempfile::tempdir().unwrap();
        let slow = run_check(&check("sleep 30", 1), dir.path());
        assert!(slow.timed_out && !slow.passed && slow.duration_ms < 10_000);
    }

    #[test]
    fn a_stale_bytecode_cache_cannot_hide_an_edit_from_the_next_check() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("m.py"), "VALUE = 1\n").unwrap();
        // Bytecode is written by whatever ran in the checkout before.
        let first = Command::new("python3")
            .args(["-c", "import m"])
            .current_dir(dir.path())
            .status()
            .unwrap();
        assert!(first.success());
        // Same length, same second: the cache still looks current.
        std::fs::write(dir.path().join("m.py"), "VALUE = 2\n").unwrap();
        let r = run_check(
            &check("python3 -c 'import m; assert m.VALUE == 2'", 30),
            dir.path(),
        );
        assert!(r.passed, "{}", r.stderr_tail);
    }

    #[test]
    fn a_check_is_name_equals_command() {
        assert!(Check::parse("unit=python3 -m unittest", 60, true).is_ok());
        assert!(Check::parse("no-equals", 60, true).is_err());
        assert!(Check::parse("=cmd", 60, true).is_err());
    }

    #[test]
    fn output_secrets_are_redacted_before_they_are_kept() {
        let dir = tempfile::tempdir().unwrap();
        let r = run_check(&check("echo password=hunter2hunter2", 5), dir.path());
        assert!(
            !r.stdout_tail.contains("hunter2hunter2"),
            "{}",
            r.stdout_tail
        );
    }
}
