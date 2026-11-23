// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agent execution backends whose every
// attempt is accounted for, for its clients. If your team needs expertise
// in reliable agent orchestration, you can procure our services by sending
// an email to info@swedishembedded.com.

//! What executes attempts: the seam between the controller and an
//! execution worker.
//!
//! A backend takes a batch of [`ExecutionSpec`]s and an output directory it
//! owns, and returns one [`AttemptReport`] per attempt that finished. It
//! never grades. [`ExecutionBackend::reconcile`] reads back what an earlier,
//! interrupted call to the same output directory finished, without running
//! anything again: the controller calls it before it launches replacements.
//!
//! [`HarborWorker`] is the production backend: the `splinter-harbor-worker`
//! program on Harbor's job API, run as a subprocess.

use std::path::{Path, PathBuf};
use std::process::Command;

use splinter_core::factory::attempt::{AttemptReport, ATTEMPT_SCHEMA};
use splinter_core::factory::execution::ExecutionSpec;

/// Why a backend could not execute or read back a batch.
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    /// The worker could not be started or failed itself.
    #[error("the execution worker failed: {0}")]
    Worker(String),
    /// A report is unreadable.
    #[error("{path}: {message}")]
    Report {
        /// The file.
        path: PathBuf,
        /// What is wrong.
        message: String,
    },
}

/// Where a batch runs.
#[derive(Clone, Copy, Debug)]
pub struct Batch<'a> {
    /// The specifications.
    pub specs: &'a [ExecutionSpec],
    /// The directory the specs' task directories are relative to.
    pub tasks: &'a Path,
    /// The directory the specs' agent files and mounts are relative to.
    pub files: &'a Path,
    /// The backend's output directory.
    pub out: &'a Path,
    /// Attempts at once.
    pub concurrency: u32,
}

/// What executes attempts.
pub trait ExecutionBackend: Send + Sync {
    /// The backend's identity, part of every rollout stage's implementation.
    fn identity(&self) -> String;

    /// Execute a batch; one report per finished attempt.
    ///
    /// # Errors
    /// The backend itself failed (not an attempt: a failed attempt is a
    /// report).
    fn execute(&self, batch: Batch<'_>) -> Result<Vec<AttemptReport>, BackendError>;

    /// The reports of every attempt an earlier execution into `out`
    /// finished, running nothing.
    ///
    /// # Errors
    /// The output cannot be read.
    fn reconcile(&self, out: &Path) -> Result<Vec<AttemptReport>, BackendError>;
}

/// Every attempt report under `out/attempts`, sorted by slot and attempt.
///
/// # Errors
/// A report that is not a `splinter.attempt/1` document.
pub fn read_reports(out: &Path) -> Result<Vec<AttemptReport>, BackendError> {
    let mut reports = Vec::new();
    let mut stack = vec![out.join("attempts")];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "json") {
                let fail = |message: String| BackendError::Report {
                    path: path.clone(),
                    message,
                };
                let text = std::fs::read_to_string(&path).map_err(|e| fail(e.to_string()))?;
                let report: AttemptReport =
                    serde_json::from_str(&text).map_err(|e| fail(e.to_string()))?;
                if report.schema != ATTEMPT_SCHEMA {
                    return Err(fail(format!(
                        "schema {:?} is not {ATTEMPT_SCHEMA}",
                        report.schema
                    )));
                }
                reports.push(report);
            }
        }
    }
    reports.sort_by(|a, b| (&a.slot, a.attempt).cmp(&(&b.slot, b.attempt)));
    Ok(reports)
}

/// The Harbor worker program.
#[derive(Clone, Debug)]
pub struct HarborWorker {
    /// The `splinter-harbor-worker` executable.
    pub program: PathBuf,
    /// Its version, as `--version` printed it.
    pub version: String,
}

impl HarborWorker {
    /// The worker at `program`, asking it for its version.
    ///
    /// # Errors
    /// It cannot be run.
    pub fn new(program: impl Into<PathBuf>) -> Result<Self, BackendError> {
        let program = program.into();
        let output = Command::new(&program)
            .arg("--version")
            .output()
            .map_err(|e| BackendError::Worker(format!("{}: {e}", program.display())))?;
        if !output.status.success() {
            return Err(BackendError::Worker(format!(
                "{} --version failed",
                program.display()
            )));
        }
        let version = String::from_utf8_lossy(&output.stdout).trim().to_owned();
        Ok(Self { program, version })
    }

    fn run(&self, command: &mut Command) -> Result<(), BackendError> {
        let status = command
            .status()
            .map_err(|e| BackendError::Worker(format!("{}: {e}", self.program.display())))?;
        if status.success() {
            Ok(())
        } else {
            Err(BackendError::Worker(format!(
                "{} exited with {status}",
                self.program.display()
            )))
        }
    }
}

impl ExecutionBackend for HarborWorker {
    fn identity(&self) -> String {
        format!("harbor-worker {}", self.version)
    }

    fn execute(&self, batch: Batch<'_>) -> Result<Vec<AttemptReport>, BackendError> {
        std::fs::create_dir_all(batch.out).map_err(|e| BackendError::Worker(e.to_string()))?;
        let specs = batch.out.join("specs.jsonl");
        let mut lines = String::new();
        for spec in batch.specs {
            lines.push_str(
                &serde_json::to_string(spec).map_err(|e| BackendError::Worker(e.to_string()))?,
            );
            lines.push('\n');
        }
        std::fs::write(&specs, lines).map_err(|e| BackendError::Worker(e.to_string()))?;
        self.run(
            Command::new(&self.program)
                .arg("run")
                .arg("--specs")
                .arg(&specs)
                .arg("--tasks")
                .arg(batch.tasks)
                .arg("--files")
                .arg(batch.files)
                .arg("--out")
                .arg(batch.out)
                .arg("--concurrency")
                .arg(batch.concurrency.to_string()),
        )?;
        read_reports(batch.out)
    }

    fn reconcile(&self, out: &Path) -> Result<Vec<AttemptReport>, BackendError> {
        if out.join("harbor").is_dir() {
            self.run(
                Command::new(&self.program)
                    .arg("reconcile")
                    .arg("--out")
                    .arg(out),
            )?;
        }
        read_reports(out)
    }
}
