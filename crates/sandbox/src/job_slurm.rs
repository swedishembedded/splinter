// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements scheduler-backed execution of multi-node
// jobs for its clients. If your team needs expertise in batch-system
// integration or scheduler-independent job control, you can procure our
// services by sending an email to info@swedishembedded.com.

//! A [`JobLauncher`] that submits a job to Slurm.
//!
//! The specification becomes a batch script: one node per machine, one task
//! per rank and the accelerators each rank asks for, with the wall clock as
//! the scheduler's time limit. `sbatch --parsable` submits it and its job id
//! is recorded in the launch directory, so observing and stopping the job
//! read the same durable record a restarted controller reopens.
//!
//! The binaries and any further `sbatch` arguments are values the caller
//! passes, never the process environment: a test points the launcher at
//! stand-in programs and rehearses the whole lifecycle without a scheduler.

use std::ffi::{OsStr, OsString};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use splinter_core::job::JobSpec;

use super::{claim_attempt, write_launch, JobError, JobLauncher, JobOutcome, Launch, Observation};

/// The batch script written into the launch directory.
const SCRIPT_FILE: &str = "job.sbatch";

/// The file recording the scheduler's job id, beside the launch.
const SCHEDULER_JOB_FILE: &str = "scheduler.job";

/// The scheduler's view of a job.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlurmState {
    /// Accepted by the controller, not yet placed.
    Pending,
    /// Placed, and its tasks may run.
    Running,
    /// All tasks exited and the job is being cleaned up.
    Completing,
    /// Every task exited with code zero.
    Completed,
    /// A task failed, or the job could not run.
    Failed,
    /// Stopped by a cancel request.
    Cancelled,
    /// Exceeded its time limit.
    Timeout,
    /// Taken back by the scheduler for higher-priority work.
    Preempted,
    /// A node it ran on failed.
    NodeFail,
}

impl SlurmState {
    /// The state `squeue` spelled, as the value it names.
    pub fn parse(text: &str) -> Result<SlurmState, JobError> {
        match text.trim() {
            "PENDING" => Ok(SlurmState::Pending),
            "RUNNING" => Ok(SlurmState::Running),
            "COMPLETING" => Ok(SlurmState::Completing),
            "COMPLETED" => Ok(SlurmState::Completed),
            "FAILED" => Ok(SlurmState::Failed),
            "CANCELLED" => Ok(SlurmState::Cancelled),
            "TIMEOUT" => Ok(SlurmState::Timeout),
            "PREEMPTED" => Ok(SlurmState::Preempted),
            "NODE_FAIL" => Ok(SlurmState::NodeFail),
            state => Err(JobError::UnknownState {
                state: state.to_string(),
            }),
        }
    }

    /// What the launcher trait reports for this state.
    ///
    /// A job still in the scheduler's hands reads as running - with no rank
    /// placed yet while it is pending - so a controller keeps reconciling it.
    /// A job the scheduler finished reads as finished with an outcome; only
    /// `COMPLETED` proves the exit code, since the scheduler's state does not
    /// carry a code for the others.
    fn observation(self, world_size: u32) -> Observation {
        match self {
            SlurmState::Pending => Observation::Running {
                ranks_alive: Vec::new(),
            },
            SlurmState::Running | SlurmState::Completing => Observation::Running {
                ranks_alive: (0..world_size).collect(),
            },
            SlurmState::Completed => Observation::Finished {
                outcome: JobOutcome::Completed,
                exit_code: Some(0),
            },
            SlurmState::Failed | SlurmState::Timeout | SlurmState::NodeFail => {
                Observation::Finished {
                    outcome: JobOutcome::Failed,
                    exit_code: None,
                }
            }
            SlurmState::Cancelled => Observation::Finished {
                outcome: JobOutcome::Cancelled,
                exit_code: None,
            },
            SlurmState::Preempted => Observation::Finished {
                outcome: JobOutcome::Preempted,
                exit_code: None,
            },
        }
    }
}

/// Submits jobs to Slurm through `sbatch`, observes them with `squeue` and
/// stops them with `scancel`.
#[derive(Clone, Debug)]
pub struct SlurmLauncher {
    sbatch: PathBuf,
    squeue: PathBuf,
    scancel: PathBuf,
    sbatch_args: Vec<String>,
}

impl SlurmLauncher {
    /// A launcher running the given programs.
    #[must_use]
    pub fn new(
        sbatch: impl Into<PathBuf>,
        squeue: impl Into<PathBuf>,
        scancel: impl Into<PathBuf>,
    ) -> SlurmLauncher {
        SlurmLauncher {
            sbatch: sbatch.into(),
            squeue: squeue.into(),
            scancel: scancel.into(),
            sbatch_args: Vec::new(),
        }
    }

    /// Further arguments every submission passes before the script, such as a
    /// partition or an account.
    #[must_use]
    pub fn with_sbatch_args(mut self, arguments: Vec<String>) -> SlurmLauncher {
        self.sbatch_args = arguments;
        self
    }

    /// The scheduler's job id recorded for `launch`.
    pub fn job_id(&self, launch: &Launch) -> Result<String, JobError> {
        let path = launch.dir.join(SCHEDULER_JOB_FILE);
        match fs::read_to_string(&path) {
            Ok(text) => {
                let id = text.trim();
                if id.is_empty() {
                    return Err(JobError::NoSchedulerJob {
                        dir: launch.dir.clone(),
                    });
                }
                Ok(id.to_string())
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Err(JobError::NoSchedulerJob {
                dir: launch.dir.clone(),
            }),
            Err(e) => Err(JobError::io(format!("reading {}", path.display()), e)),
        }
    }

    /// The state the scheduler reports for `launch`.
    ///
    /// [`JobLauncher::observe`] cannot carry an error, so a caller that needs
    /// to know why the scheduler could not be asked uses this instead.
    pub fn state(&self, launch: &Launch) -> Result<SlurmState, JobError> {
        let job_id = self.job_id(launch)?;
        let arguments = [
            OsStr::new("-h"),
            OsStr::new("-j"),
            OsStr::new(job_id.as_str()),
            OsStr::new("-o"),
            OsStr::new("%T"),
        ];
        let output = run(&self.squeue, "squeue", &arguments, None)?;
        let state = output.trim();
        if state.is_empty() {
            return Err(JobError::NotListed { job_id });
        }
        if state.split_whitespace().count() != 1 {
            return Err(JobError::Unparseable {
                what: "squeue".to_string(),
                output: state.to_string(),
            });
        }
        SlurmState::parse(state)
    }

    fn cancel(&self, job_id: &str) -> Result<(), JobError> {
        run(&self.scancel, "scancel", &[OsStr::new(job_id)], None).map(|_| ())
    }
}

impl JobLauncher for SlurmLauncher {
    fn name(&self) -> &'static str {
        "slurm"
    }

    fn launch(&self, spec: &JobSpec, attempt: u32, dir: &Path) -> Result<Launch, JobError> {
        claim_attempt(spec, attempt, dir)?;
        let script = dir.join(SCRIPT_FILE);
        fs::write(&script, batch_script(spec))
            .map_err(|e| JobError::io(format!("writing {}", script.display()), e))?;

        let mut arguments: Vec<OsString> = self.sbatch_args.iter().map(OsString::from).collect();
        arguments.push(OsString::from("--parsable"));
        arguments.push(script.into_os_string());
        let arguments: Vec<&OsStr> = arguments.iter().map(OsString::as_os_str).collect();
        let output = run(&self.sbatch, "sbatch", &arguments, Some(dir))?;
        let job_id = parse_job_id(&output)?;

        let launch = Launch {
            digest: spec.digest(),
            attempt,
            world_size: spec.workers.world_size(),
            pids: Vec::new(),
            dir: dir.to_path_buf(),
        };
        let recording = fs::write(dir.join(SCHEDULER_JOB_FILE), format!("{job_id}\n"))
            .map_err(|e| JobError::io(format!("recording job {job_id} in {}", dir.display()), e));
        if let Err(error) = recording.and_then(|()| write_launch(&launch)) {
            // The scheduler now holds a job the launch directory does not
            // name: cancel it rather than leave it owned by nobody. A cancel
            // that fails is reported with the recording failure, not instead
            // of it.
            return match self.cancel(&job_id) {
                Ok(()) => Err(error),
                Err(cancel) => Err(JobError::Unrecorded {
                    job_id,
                    record: error.to_string(),
                    cancel: cancel.to_string(),
                }),
            };
        }
        Ok(launch)
    }

    fn observe(&self, launch: &Launch) -> Observation {
        // Nothing can be claimed when the scheduler cannot be asked; a caller
        // that needs the reason calls `state`, whose error is typed.
        match self.state(launch) {
            Ok(state) => state.observation(launch.world_size),
            Err(_) => Observation::Finished {
                outcome: JobOutcome::Lost,
                exit_code: None,
            },
        }
    }

    fn stop(&self, launch: &Launch, rank: Option<u32>) -> Result<(), JobError> {
        if let Some(rank) = rank {
            return Err(JobError::Unsupported {
                action: format!("stop rank {rank} alone: the scheduler cancels the job as a whole"),
            });
        }
        let job_id = self.job_id(launch)?;
        self.cancel(&job_id)
    }
}

/// The scheduler's job id from `sbatch --parsable`, which prints
/// `<job id>` or `<job id>;<cluster>`.
fn parse_job_id(output: &str) -> Result<String, JobError> {
    let line = output.trim();
    let id = line.split(';').next().unwrap_or(line).trim();
    if id.is_empty() || !id.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(JobError::Unparseable {
            what: "sbatch".to_string(),
            output: line.to_string(),
        });
    }
    Ok(id.to_string())
}

/// Run a scheduler command, returning its standard output.
fn run(
    binary: &Path,
    name: &str,
    arguments: &[&OsStr],
    dir: Option<&Path>,
) -> Result<String, JobError> {
    let mut command = Command::new(binary);
    command.args(arguments);
    if let Some(dir) = dir {
        command.current_dir(dir);
    }
    let output = command
        .output()
        .map_err(|source| JobError::SchedulerCommand {
            command: format!("{name} ({})", binary.display()),
            source,
        })?;
    if !output.status.success() {
        return Err(JobError::SchedulerFailed {
            command: name.to_string(),
            status: output.status.to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
        });
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// The batch script that runs one task per rank of `spec`.
fn batch_script(spec: &JobSpec) -> String {
    let mut script = String::from("#!/bin/sh\n");
    script.push_str(&format!("#SBATCH --job-name={}\n", job_name(spec)));
    script.push_str(&format!("#SBATCH --nodes={}\n", spec.workers.nodes));
    script.push_str(&format!(
        "#SBATCH --ntasks-per-node={}\n",
        spec.workers.ranks_per_node
    ));
    if spec.workers.gpus_per_rank > 0 {
        script.push_str(&format!(
            "#SBATCH --gpus-per-task={}\n",
            spec.workers.gpus_per_rank
        ));
    }
    if spec.wall_clock_seconds > 0 {
        script.push_str(&format!(
            "#SBATCH --time={}\n",
            wall_clock(spec.wall_clock_seconds)
        ));
    }
    script.push_str("#SBATCH --output=slurm-%j.out\n");
    let command: Vec<String> = spec.command.iter().map(|part| shell_word(part)).collect();
    script.push_str(&format!("exec srun {}\n", command.join(" ")));
    script
}

/// A job name that is the same for the same specification and valid in a
/// batch system.
fn job_name(spec: &JobSpec) -> String {
    let digest = spec.digest();
    let hex = digest
        .as_str()
        .rsplit(':')
        .next()
        .unwrap_or(digest.as_str());
    format!("splinter-{}", &hex[..hex.len().min(12)])
}

/// A wall clock in the scheduler's `[days-]hours:minutes:seconds` form.
fn wall_clock(seconds: u64) -> String {
    let days = seconds / 86_400;
    let hours = (seconds % 86_400) / 3_600;
    let minutes = (seconds % 3_600) / 60;
    let rest = seconds % 60;
    if days > 0 {
        format!("{days}-{hours:02}:{minutes:02}:{rest:02}")
    } else {
        format!("{hours:02}:{minutes:02}:{rest:02}")
    }
}

/// One command word for the batch script's shell.
///
/// The rank and world placeholders become the scheduler's own task variables,
/// so one script gives every rank its own values. Every other character is
/// quoted: the job's command is a program and its arguments, never a shell
/// line, and must not gain one here.
fn shell_word(argument: &str) -> String {
    const RANK: &str = "{rank}";
    const WORLD: &str = "{world}";
    let mut word = String::new();
    let mut rest = argument;
    loop {
        let rank = rest.find(RANK);
        let world = rest.find(WORLD);
        let (position, placeholder, expansion) = match (rank, world) {
            (Some(rank), Some(world)) if world < rank => (world, WORLD, "${SLURM_NTASKS}"),
            (Some(rank), _) => (rank, RANK, "${SLURM_PROCID}"),
            (None, Some(world)) => (world, WORLD, "${SLURM_NTASKS}"),
            (None, None) => {
                word.push_str(&quote(rest));
                return word;
            }
        };
        word.push_str(&quote(&rest[..position]));
        word.push_str(expansion);
        rest = &rest[position + placeholder.len()..];
    }
}

/// `text` as one single-quoted shell word, empty when there is nothing to
/// quote so placeholders compose with their surroundings.
fn quote(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    format!("'{}'", text.replace('\'', "'\\''"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_wall_clock_is_formatted_for_the_scheduler() {
        assert_eq!(wall_clock(0), "00:00:00");
        assert_eq!(wall_clock(3_723), "01:02:03");
        assert_eq!(wall_clock(90_061), "1-01:01:01");
    }

    #[test]
    fn placeholders_become_the_schedulers_task_variables() {
        assert_eq!(shell_word("--rank={rank}"), "'--rank='${SLURM_PROCID}");
        assert_eq!(shell_word("--world={world}"), "'--world='${SLURM_NTASKS}");
        assert_eq!(shell_word("{rank}"), "${SLURM_PROCID}");
        assert_eq!(shell_word("plain"), "'plain'");
        assert_eq!(shell_word("two words"), "'two words'");
        assert_eq!(shell_word("it's"), "'it'\\''s'");
        assert_eq!(
            shell_word("{rank}{world}"),
            "${SLURM_PROCID}${SLURM_NTASKS}"
        );
    }

    #[test]
    fn the_job_id_is_read_past_the_cluster_suffix() {
        assert_eq!(parse_job_id("4242\n").unwrap(), "4242");
        assert_eq!(parse_job_id("4242;alpha\n").unwrap(), "4242");
        assert!(parse_job_id("Submitted batch job 4242").is_err());
        assert!(parse_job_id("").is_err());
    }
}
