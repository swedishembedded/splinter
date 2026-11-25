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
//! [`ExecutionBackend::import`] reports the trials of Harbor jobs run
//! elsewhere the same way, running nothing.
//!
//! [`HarborWorker`] is the production backend: the `splinter-harbor-worker`
//! program on Harbor's job API, run as a subprocess.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};
use splinter_core::factory::attempt::{AttemptReport, ATTEMPT_SCHEMA};
use splinter_core::factory::execution::ExecutionSpec;
use splinter_core::factory::task::TaskRef;

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

/// What an import reads and where it writes.
#[derive(Clone, Copy, Debug)]
pub struct Import<'a> {
    /// The pass whose slots the trials fill.
    pub pass: &'a str,
    /// The Harbor job directories, in order: a task's trials fill its slots
    /// in this order, then by trial name.
    pub jobs: &'a [PathBuf],
    /// The tasks a trial may be of.
    pub tasks: &'a [TaskRef],
    /// The directory the tasks' ids are relative to.
    pub task_root: &'a Path,
    /// The backend's output directory: new or empty. The trial of job `n`
    /// named `t` is copied to `harbor/<n>/<t>` below it.
    pub out: &'a Path,
}

/// A trial an import did not take, and why.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportRefusal {
    /// The job's index in [`Import::jobs`].
    pub job: usize,
    /// The trial's directory name in the job.
    pub trial: String,
    /// Why.
    pub reason: String,
}

/// What an import found.
#[derive(Clone, Debug, PartialEq)]
pub struct Imported {
    /// One report per trial taken, sorted by slot.
    pub reports: Vec<AttemptReport>,
    /// The trials refused, in the order met.
    pub refused: Vec<ImportRefusal>,
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

    /// Report the trials of Harbor jobs run elsewhere as attempts of
    /// `import.tasks`, running nothing: each trial is matched to its task by
    /// Harbor's own content digest of the task directory, copied under
    /// `import.out`, and reported exactly as a trial the backend ran would
    /// be. A trial that matches no task, or that Harbor never finished, is
    /// refused with its reason.
    ///
    /// # Errors
    /// The backend cannot import, or failed itself.
    fn import(&self, import: Import<'_>) -> Result<Imported, BackendError> {
        let _ = import;
        Err(BackendError::Worker(format!(
            "{} cannot import Harbor jobs",
            self.identity()
        )))
    }

    /// Run one attempt interactively on the caller's terminal: a person
    /// drives the agent, and the finished session is graded like any
    /// attempt. `continue_from` is an earlier played trial's directory whose
    /// conversation the session continues (the earlier trial is left as it
    /// is).
    ///
    /// # Errors
    /// The backend cannot play, refused the spec, or failed.
    fn play(
        &self,
        spec: &ExecutionSpec,
        batch: Batch<'_>,
        continue_from: Option<&Path>,
    ) -> Result<AttemptReport, BackendError> {
        let _ = (spec, batch, continue_from);
        Err(BackendError::Worker(format!(
            "{} cannot play interactively",
            self.identity()
        )))
    }
}

/// The backend of a machine that offers no execution worker: every
/// execution is refused with why, so a recipe whose stages execute no
/// attempts (an online stage runs its own rollouts) still runs there.
#[derive(Clone, Debug)]
pub struct NoWorker(pub String);

impl ExecutionBackend for NoWorker {
    fn identity(&self) -> String {
        "no execution worker".into()
    }

    fn execute(&self, _batch: Batch<'_>) -> Result<Vec<AttemptReport>, BackendError> {
        Err(BackendError::Worker(self.0.clone()))
    }

    fn reconcile(&self, _out: &Path) -> Result<Vec<AttemptReport>, BackendError> {
        Err(BackendError::Worker(self.0.clone()))
    }
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

/// The refusals an import wrote to `out/refused.jsonl`.
///
/// # Errors
/// The file is missing or a line is not a refusal.
pub fn read_refusals(out: &Path) -> Result<Vec<ImportRefusal>, BackendError> {
    let path = out.join("refused.jsonl");
    let fail = |message: String| BackendError::Report {
        path: path.clone(),
        message,
    };
    std::fs::read_to_string(&path)
        .map_err(|e| fail(e.to_string()))?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).map_err(|e| fail(e.to_string())))
        .collect()
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

    fn play(
        &self,
        spec: &ExecutionSpec,
        batch: Batch<'_>,
        continue_from: Option<&Path>,
    ) -> Result<AttemptReport, BackendError> {
        std::fs::create_dir_all(batch.out.parent().unwrap_or(batch.out))
            .map_err(|e| BackendError::Worker(e.to_string()))?;
        let spec_file = batch.out.with_extension("spec.json");
        let text = serde_json::to_string(spec).map_err(|e| BackendError::Worker(e.to_string()))?;
        std::fs::write(&spec_file, text).map_err(|e| BackendError::Worker(e.to_string()))?;
        let mut command = Command::new(&self.program);
        command
            .arg("play")
            .arg("--spec")
            .arg(&spec_file)
            .arg("--tasks")
            .arg(batch.tasks)
            .arg("--files")
            .arg(batch.files)
            .arg("--out")
            .arg(batch.out);
        if let Some(trial) = continue_from {
            command.arg("--continue").arg(trial);
        }
        self.run(&mut command)?;
        read_reports(batch.out)?
            .into_iter()
            .next()
            .ok_or_else(|| BackendError::Worker("the played session left no report".into()))
    }

    fn import(&self, import: Import<'_>) -> Result<Imported, BackendError> {
        let worker = |e: std::io::Error| BackendError::Worker(e.to_string());
        std::fs::create_dir_all(import.out).map_err(worker)?;
        // The task list lives beside the output, which the worker wants empty.
        let listing = import.out.with_extension("tasks.jsonl");
        let mut lines = String::new();
        for task in import.tasks {
            lines.push_str(
                &serde_json::to_string(task).map_err(|e| BackendError::Worker(e.to_string()))?,
            );
            lines.push('\n');
        }
        std::fs::write(&listing, lines).map_err(worker)?;
        let mut command = Command::new(&self.program);
        command
            .arg("import")
            .arg("--pass")
            .arg(import.pass)
            .arg("--tasks")
            .arg(import.task_root)
            .arg("--task-list")
            .arg(&listing)
            .arg("--out")
            .arg(import.out);
        for job in import.jobs {
            command.arg("--job").arg(job);
        }
        self.run(&mut command)?;
        Ok(Imported {
            reports: read_reports(import.out)?,
            refused: read_refusals(import.out)?,
        })
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
