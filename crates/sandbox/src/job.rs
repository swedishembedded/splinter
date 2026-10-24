// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements detached job execution that outlives the
// process that submitted it, for its clients. If your team needs expertise in
// process supervision, work stealing or multi-node launch simulation, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Jobs that run outside the process that submitted them.
//!
//! A [`JobLauncher`] starts a job's workers, observes them, and stops them. Its
//! launch is a directory: `launch.json` names the attempt and the worker
//! processes, and each worker's output is kept beside it. The directory is
//! written once, so submitting the same attempt twice fails instead of starting
//! a second set of workers; a controller that restarted reopens the directory
//! ([`Launch::open`]) rather than launching again.
//!
//! [`LocalLauncher`] starts every rank as a separate process on this machine.
//! One rank is a plain local job. More than one is how a cluster is simulated
//! without a scheduler: the ranks are real processes with their own process
//! groups, they can be killed one at a time, and the controller can be killed
//! and reopened while they run. A scheduler-backed launcher implements the same
//! trait.
//!
//! Workers are started in their own session (`setsid`), so they survive the
//! controller and a terminal's process-group kill. An exit code is known when
//! the submitting process is still alive; a launch started under a supervisor
//! (see [`LocalLauncher::supervised`]) writes each worker's exit into the
//! launch directory, so a controller that restarted reads the same completed
//! or failed attempt instead of a lost one. A reopened launch whose workers
//! were not supervised reports that they are gone, and the durable job record
//! says what their artifacts prove.

use std::collections::HashMap;
use std::ffi::{OsStr, OsString};
use std::fs;
use std::io::{self, Write};
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use splinter_core::digest::Digest;
use splinter_core::job::JobSpec;

/// How long a stop waits for a worker to leave after `SIGTERM`.
pub const STOP_GRACE: Duration = Duration::from_secs(2);

/// The file naming a launch, beside the workers' output.
const LAUNCH_FILE: &str = "launch.json";

/// The file claiming an attempt before anything is started for it.
const INTENT_FILE: &str = "launch.intent";

#[path = "job_slurm.rs"]
mod slurm;

#[path = "job_plan.rs"]
mod plan;

pub use slurm::{SlurmLauncher, SlurmState};

/// The first argument of a process the local launcher starts to supervise one
/// worker: a launcher that records exit status durably runs this program in
/// place of the worker, and it runs the worker itself.
pub const SUPERVISOR_MARKER: &str = "__splinter_job_supervisor";

/// Why a launch could not be started, observed or stopped.
#[derive(Debug, thiserror::Error)]
pub enum JobError {
    /// The specification cannot describe a job.
    #[error("the job specification is invalid: {0}")]
    Invalid(#[from] splinter_core::job::JobSpecError),
    /// The attempt directory already names a launch; submitting twice must not
    /// start a second set of workers.
    #[error("{} already holds a launch of attempt {attempt}; a job is launched once per attempt", dir.display())]
    AlreadyLaunched {
        /// The attempt directory.
        dir: PathBuf,
        /// The attempt already launched there.
        attempt: u32,
    },
    /// The launch directory could not be read or written.
    #[error("{context}: {source}")]
    Io {
        /// What was being done.
        context: String,
        /// The operating-system failure.
        source: io::Error,
    },
    /// A worker process could not be started.
    #[error("starting rank {rank} of {}: {source}", .world)]
    Spawn {
        /// The worker's rank.
        rank: u32,
        /// The job's world size.
        world: u32,
        /// The operating-system failure.
        source: io::Error,
    },
    /// A scheduler command could not be run at all: it is missing, or not
    /// executable.
    #[error("could not run {command}: {source}")]
    SchedulerCommand {
        /// The command, with the binary it was to run.
        command: String,
        /// The operating-system failure.
        source: io::Error,
    },
    /// A scheduler command ran and failed.
    #[error("{command} failed with {status}: {stderr}")]
    SchedulerFailed {
        /// The command that failed.
        command: String,
        /// How it ended.
        status: String,
        /// What it said on stderr.
        stderr: String,
    },
    /// A scheduler command succeeded but printed something this launcher
    /// cannot read.
    #[error("{what} printed output a scheduler job cannot be read from: {output:?}")]
    Unparseable {
        /// What was being read.
        what: String,
        /// The output that was read.
        output: String,
    },
    /// The scheduler reported a state this launcher does not know.
    #[error("the scheduler reported an unknown job state: {state:?}")]
    UnknownState {
        /// The state as the scheduler spelled it.
        state: String,
    },
    /// The scheduler no longer lists the job, so its terminal state is not
    /// knowable from it.
    #[error("the scheduler no longer lists job {job_id}")]
    NotListed {
        /// The scheduler's job id.
        job_id: String,
    },
    /// A launch directory holds no scheduler job id.
    #[error("{} holds no scheduler job id", dir.display())]
    NoSchedulerJob {
        /// The launch directory.
        dir: PathBuf,
    },
    /// The launcher cannot carry out this request.
    #[error("the launcher cannot {action}")]
    Unsupported {
        /// What was asked for.
        action: String,
    },
    /// A scheduler accepted a job the launch directory could not record, and
    /// cancelling it failed as well.
    #[error("the scheduler accepted job {job_id} but recording it failed: {record}; cancelling it failed: {cancel}")]
    Unrecorded {
        /// The scheduler's job id.
        job_id: String,
        /// Why the launch directory could not record the job.
        record: String,
        /// Why the cancelling failed.
        cancel: String,
    },
}

impl JobError {
    fn io(context: impl Into<String>, source: io::Error) -> JobError {
        JobError::Io {
            context: context.into(),
            source,
        }
    }
}

/// A submitted job's launch directory.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Launch {
    /// The specification this launch runs.
    pub digest: Digest,
    /// The attempt, counting from one.
    pub attempt: u32,
    /// How many workers were started.
    pub world_size: u32,
    /// The workers' process ids, in rank order.
    pub pids: Vec<u32>,
    /// The directory holding the launch and the workers' output.
    pub dir: PathBuf,
}

impl Launch {
    /// The launch a previous process wrote, if there is one.
    pub fn open(dir: &Path) -> Result<Option<Launch>, JobError> {
        let path = dir.join(LAUNCH_FILE);
        match fs::read(&path) {
            Ok(bytes) => {
                let mut launch: Launch = serde_json::from_slice(&bytes).map_err(|e| {
                    JobError::io(
                        format!("parsing {}", path.display()),
                        io::Error::new(io::ErrorKind::InvalidData, e),
                    )
                })?;
                launch.dir = dir.to_path_buf();
                Ok(Some(launch))
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(JobError::io(format!("reading {}", path.display()), e)),
        }
    }

    /// The worker of `rank`, when the launch has one.
    #[must_use]
    pub fn pid(&self, rank: u32) -> Option<u32> {
        usize::try_from(rank)
            .ok()
            .and_then(|rank| self.pids.get(rank).copied())
    }

    /// Which ranks are still running, by asking the operating system rather
    /// than by trusting the submitting process's memory.
    #[must_use]
    pub fn ranks_alive(&self) -> Vec<u32> {
        (0..self.world_size)
            .filter(|rank| self.pid(*rank).is_some_and(process_alive))
            .collect()
    }

    /// What happened to the job as the operating system sees it.
    ///
    /// Once no worker is alive, the exit is the one a supervisor recorded for
    /// every rank, so a controller that restarted reads the same terminal
    /// state the submitting process would have. A launch whose workers were
    /// not supervised, or whose supervisor died before it could record the
    /// exit, reports no exit code rather than inventing one, and its outcome
    /// is [`JobOutcome::Lost`].
    #[must_use]
    pub fn observe(&self) -> Observation {
        let alive = self.ranks_alive();
        if alive.is_empty() {
            let exit_code = self.recorded_exit();
            Observation::Finished {
                outcome: outcome_of(exit_code),
                exit_code,
            }
        } else {
            Observation::Running { ranks_alive: alive }
        }
    }

    /// The exit every rank's supervisor recorded, or `None` when one is
    /// missing. A job of several ranks has one exit code: the first non-zero
    /// one, so any failed rank fails the job.
    fn recorded_exit(&self) -> Option<i32> {
        let mut codes = Vec::with_capacity(self.world_size as usize);
        for rank in 0..self.world_size {
            codes.push(read_exit(&self.dir, rank)?);
        }
        Some(codes.into_iter().find(|code| *code != 0).unwrap_or(0))
    }

    /// Stop one worker (the failure a rank crash is simulated with), or every
    /// worker when `rank` is `None`.
    pub fn stop(&self, rank: Option<u32>) -> Result<(), JobError> {
        let targets: Vec<u32> = match rank {
            Some(rank) => self.pid(rank).into_iter().collect(),
            None => self.pids.clone(),
        };
        for pid in targets {
            signal(pid, SIGTERM)?;
        }
        let deadline = Instant::now() + STOP_GRACE;
        while Instant::now() < deadline {
            let alive: Vec<u32> = match rank {
                Some(rank) => self
                    .pid(rank)
                    .filter(|pid| process_alive(*pid))
                    .into_iter()
                    .collect(),
                None => self
                    .pids
                    .iter()
                    .copied()
                    .filter(|pid| process_alive(*pid))
                    .collect(),
            };
            if alive.is_empty() {
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        // A worker that ignored the polite request is killed.
        let remaining: Vec<u32> = match rank {
            Some(rank) => self
                .pid(rank)
                .filter(|pid| process_alive(*pid))
                .into_iter()
                .collect(),
            None => self
                .pids
                .iter()
                .copied()
                .filter(|pid| process_alive(*pid))
                .collect(),
        };
        for pid in remaining {
            signal(pid, SIGKILL)?;
        }
        Ok(())
    }
}

/// What a job looks like from outside its processes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Observation {
    /// Workers are still running.
    Running {
        /// The ranks still alive.
        ranks_alive: Vec<u32>,
    },
    /// No worker is alive, and the launcher says how the attempt ended.
    Finished {
        /// What the launcher observed.
        outcome: JobOutcome,
        /// The exit code, when the launcher saw one; `None` when a scheduler
        /// does not carry one or a controller restart cannot recover it.
        exit_code: Option<i32>,
    },
}

/// How an attempt ended, as its launcher reports it.
///
/// A scheduler distinguishes outcomes a process exit code cannot: it may take
/// a job back for higher-priority work, fail the node it ran on, or cancel it
/// on request. The controller decides from these whether an attempt may be
/// retried.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JobOutcome {
    /// Every worker ended with a zero exit.
    Completed,
    /// A worker failed, or the job could not run.
    Failed,
    /// Stopped by a cancel request.
    Cancelled,
    /// Taken back by the scheduler for other work.
    Preempted,
    /// The workers are gone and no exit or scheduler state says how.
    Lost,
}

/// The outcome an observed exit code implies: zero completed, nonzero failed,
/// and an unknown exit is lost rather than a claim of success.
fn outcome_of(exit_code: Option<i32>) -> JobOutcome {
    match exit_code {
        Some(0) => JobOutcome::Completed,
        Some(_) => JobOutcome::Failed,
        None => JobOutcome::Lost,
    }
}

/// Where a job runs. One implementation starts child processes; a scheduler's
/// implementation submits the same specification and reports the same states.
pub trait JobLauncher: Send + Sync {
    /// The launcher's name, for the job record ("local", a scheduler's kind).
    fn name(&self) -> &'static str;

    /// Start `spec` as attempt `attempt` under `dir`, or report that the
    /// attempt was already launched.
    fn launch(&self, spec: &JobSpec, attempt: u32, dir: &Path) -> Result<Launch, JobError>;

    /// What the launcher knows of `launch` now.
    fn observe(&self, launch: &Launch) -> Observation;

    /// Stop the job, or one of its workers when a rank crash is being
    /// simulated.
    fn stop(&self, launch: &Launch, rank: Option<u32>) -> Result<(), JobError>;
}

/// Start every rank of a job on this machine and watch them.
///
/// One rank is a plain local job; more than one is a cluster simulated without
/// a scheduler. The children are held for the life of the launcher, so their
/// exit codes are known; a launcher reopened after a controller restart has no
/// children, and [`Launch::observe`] answers from the operating system.
///
/// A launcher built with [`LocalLauncher::supervised`] starts each worker under
/// a supervisor program instead of directly: the supervisor runs the worker,
/// records its exit code in the launch directory, and exits. The exit then
/// outlives the process that submitted the job, so a controller that restarted
/// reads a completed or failed attempt instead of a lost one. It also enforces
/// the specification's wall clock: a worker that outlives it is stopped with
/// its process group and recorded as [`TIMEOUT_EXIT`].
#[derive(Debug, Default)]
pub struct LocalLauncher {
    children: Mutex<HashMap<PathBuf, Vec<Child>>>,
    supervisor: Option<PathBuf>,
}

impl LocalLauncher {
    /// A launcher that runs every worker under `supervisor`, a program that
    /// understands [`SUPERVISOR_MARKER`] (see [`supervise`]).
    #[must_use]
    pub fn supervised(supervisor: impl Into<PathBuf>) -> LocalLauncher {
        LocalLauncher {
            children: Mutex::new(HashMap::new()),
            supervisor: Some(supervisor.into()),
        }
    }

    fn children(&self) -> std::sync::MutexGuard<'_, HashMap<PathBuf, Vec<Child>>> {
        self.children
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The process a worker is started as: the supervisor with its protocol
    /// arguments, or the worker's own command.
    fn worker_command(
        &self,
        command: &[String],
        dir: &Path,
        rank: u32,
        wall_clock_seconds: u64,
    ) -> Command {
        match &self.supervisor {
            Some(supervisor) => {
                let mut process = Command::new(supervisor);
                process
                    .arg(SUPERVISOR_MARKER)
                    .arg(dir)
                    .arg(rank.to_string())
                    .arg(wall_clock_seconds.to_string())
                    .args(command);
                process
            }
            None => {
                let mut process = Command::new(&command[0]);
                process.args(&command[1..]);
                process
            }
        }
    }
}

impl JobLauncher for LocalLauncher {
    fn name(&self) -> &'static str {
        "local"
    }

    fn launch(&self, spec: &JobSpec, attempt: u32, dir: &Path) -> Result<Launch, JobError> {
        claim_attempt(spec, attempt, dir)?;
        let world = spec.workers.world_size();
        let mut children = Vec::new();
        for rank in 0..world {
            let command = spec.command_for(rank);
            let out = fs::File::create(dir.join(format!("rank-{rank}.out")))
                .map_err(|e| JobError::io("creating a worker's stdout", e))?;
            let err = fs::File::create(dir.join(format!("rank-{rank}.err")))
                .map_err(|e| JobError::io("creating a worker's stderr", e))?;
            let mut process = self.worker_command(&command, dir, rank, spec.wall_clock_seconds);
            process
                .current_dir(dir)
                .stdin(Stdio::null())
                .stdout(Stdio::from(out))
                .stderr(Stdio::from(err));
            // A worker is its own session: a terminal's process-group kill, or
            // the controller exiting, must not take a training run down.
            // SAFETY: `setsid` is async-signal-safe and touches no Rust state.
            unsafe {
                process.pre_exec(|| {
                    if libc::setsid() == -1 {
                        return Err(io::Error::last_os_error());
                    }
                    Ok(())
                });
            }
            match process.spawn() {
                Ok(child) => children.push(child),
                Err(source) => {
                    for child in &mut children {
                        child.kill().ok();
                        child.wait().ok();
                    }
                    return Err(JobError::Spawn {
                        rank,
                        world,
                        source,
                    });
                }
            }
        }
        let pids: Vec<u32> = children.iter().map(Child::id).collect();
        let launch = Launch {
            digest: spec.digest(),
            attempt,
            world_size: world,
            pids,
            dir: dir.to_path_buf(),
        };
        if let Err(error) = write_launch(&launch) {
            // The workers started, but nothing records them: stop them rather
            // than leave processes no controller can find.
            for child in &mut children {
                child.kill().ok();
                child.wait().ok();
            }
            return Err(error);
        }
        self.children().insert(dir.to_path_buf(), children);
        Ok(launch)
    }

    fn observe(&self, launch: &Launch) -> Observation {
        {
            let mut held = self.children();
            if let Some(children) = held.get_mut(&launch.dir) {
                let mut alive = Vec::new();
                let mut exit = None;
                for (rank, child) in children.iter_mut().enumerate() {
                    match child.try_wait() {
                        Ok(Some(status)) => exit = exit.or(Some(status.code().unwrap_or(-1))),
                        Ok(None) => alive.push(rank as u32),
                        Err(_) => alive.push(rank as u32),
                    }
                }
                if alive.is_empty() {
                    // A supervisor's record is what the worker ended with; a
                    // child killed with its group reports no code to this
                    // process.
                    let exit_code = launch.recorded_exit().or(exit);
                    return Observation::Finished {
                        outcome: outcome_of(exit_code),
                        exit_code,
                    };
                }
                return Observation::Running { ranks_alive: alive };
            }
        }
        launch.observe()
    }

    fn stop(&self, launch: &Launch, rank: Option<u32>) -> Result<(), JobError> {
        launch.stop(rank)
    }
}

/// Claim `dir` for attempt `attempt` before anything is started for it.
///
/// The claim is a file created exclusively, so a controller that crashed
/// between starting a job and recording it cannot have its retry start the
/// attempt a second time.
fn claim_attempt(spec: &JobSpec, attempt: u32, dir: &Path) -> Result<(), JobError> {
    spec.validate()?;
    if let Some(existing) = Launch::open(dir)? {
        return Err(JobError::AlreadyLaunched {
            dir: dir.to_path_buf(),
            attempt: existing.attempt,
        });
    }
    fs::create_dir_all(dir).map_err(|e| JobError::io(format!("creating {}", dir.display()), e))?;
    let intent = dir.join(INTENT_FILE);
    match fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&intent)
    {
        Ok(mut file) => {
            let body = format!("{}\n{}\n", spec.digest().as_str(), attempt);
            file.write_all(body.as_bytes())
                .map_err(|e| JobError::io(format!("writing {}", intent.display()), e))?;
            // The plan is part of what the attempt was launched under, so it is
            // written before the claim is complete and before any worker starts.
            plan::write(spec, dir)
        }
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => Err(JobError::AlreadyLaunched {
            dir: dir.to_path_buf(),
            attempt,
        }),
        Err(e) => Err(JobError::io(format!("claiming {}", intent.display()), e)),
    }
}

/// The exit code a worker that outlived its wall clock is recorded with, as
/// `timeout(1)` uses.
pub const TIMEOUT_EXIT: i32 = 124;

/// Run one supervised worker and record its exit code, when `args` is a
/// supervisor invocation
/// (`SUPERVISOR_MARKER DIR RANK WALL_CLOCK_SECONDS PROGRAM [ARG...]`).
///
/// Returns `Some(code)` when the arguments name such an invocation and `None`
/// when they do not, so a program can call this before parsing its own command
/// line. The worker runs with the supervisor's standard streams (the launcher
/// points them at the rank's output files), and its exit is written to
/// `DIR/rank-RANK.exit` before this returns, so a launcher in another process
/// reads it. A program that cannot be started is recorded as 127, and one that
/// outlives a nonzero wall clock is stopped and recorded as [`TIMEOUT_EXIT`].
pub fn supervise(mut args: impl Iterator<Item = OsString>) -> Option<i32> {
    if args.next().as_deref() != Some(OsStr::new(SUPERVISOR_MARKER)) {
        return None;
    }
    let Some(dir) = args.next().map(PathBuf::from) else {
        eprintln!("splinter: a job supervisor names no launch directory");
        return Some(2);
    };
    let rank = args
        .next()
        .and_then(|rank| rank.into_string().ok())
        .and_then(|rank| rank.parse::<u32>().ok());
    let Some(rank) = rank else {
        eprintln!("splinter: a job supervisor names no rank");
        return Some(2);
    };
    let wall_clock = args
        .next()
        .and_then(|seconds| seconds.into_string().ok())
        .and_then(|seconds| seconds.parse::<u64>().ok());
    let Some(wall_clock) = wall_clock else {
        eprintln!("splinter: a job supervisor names no wall clock");
        record_exit(&dir, rank, 2);
        return Some(2);
    };
    let Some(program) = args.next() else {
        record_exit(&dir, rank, 2);
        return Some(2);
    };
    let rest: Vec<OsString> = args.collect();
    let mut worker = match Command::new(&program).args(&rest).spawn() {
        Ok(worker) => worker,
        Err(error) => {
            eprintln!(
                "splinter: starting {} for rank {rank}: {error}",
                program.to_string_lossy()
            );
            record_exit(&dir, rank, 127);
            return Some(127);
        }
    };
    let code = wait_for_worker(&mut worker, wall_clock, &dir, rank);
    record_exit(&dir, rank, code);
    Some(code)
}

/// Wait for `worker`, stopping it at `wall_clock` seconds if it outlives the
/// bound. A launcher made this process the leader of its own session, so the
/// worker shares its process group: the exit is recorded first, then the group
/// is taken down, so nothing is orphaned and a launcher in another process
/// still reads why the attempt ended.
fn wait_for_worker(worker: &mut Child, wall_clock: u64, dir: &Path, rank: u32) -> i32 {
    if wall_clock == 0 {
        return status_code(worker.wait());
    }
    let deadline = Instant::now() + Duration::from_secs(wall_clock);
    loop {
        match worker.try_wait() {
            Ok(Some(status)) => return status_code(Ok(status)),
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(20)),
            Ok(None) => {
                // Record before stopping: the group kill can take this process
                // with it when it is a session leader.
                record_exit(dir, rank, TIMEOUT_EXIT);
                // SAFETY: signal 0 names this process's own group; the ids are
                // the ones the operating system assigned.
                if unsafe { libc::getsid(0) } == unsafe { libc::getpid() } {
                    unsafe { libc::kill(0, SIGKILL) };
                }
                worker.kill().ok();
                worker.wait().ok();
                return TIMEOUT_EXIT;
            }
            Err(_) => {
                worker.kill().ok();
                worker.wait().ok();
                return TIMEOUT_EXIT;
            }
        }
    }
}

/// The code a worker ended with, a signal being reported as shells report it.
fn status_code(status: io::Result<std::process::ExitStatus>) -> i32 {
    match status {
        Ok(status) => status
            .code()
            .unwrap_or_else(|| 128 + status.signal().unwrap_or(0)),
        Err(_) => 127,
    }
}

/// The file a worker's supervisor records its exit code in.
fn exit_file(dir: &Path, rank: u32) -> PathBuf {
    dir.join(format!("rank-{rank}.exit"))
}

/// The exit code recorded for `rank`, when one was.
fn read_exit(dir: &Path, rank: u32) -> Option<i32> {
    fs::read_to_string(exit_file(dir, rank))
        .ok()?
        .trim()
        .parse()
        .ok()
}

/// Record `rank`'s exit code where a launcher in another process reads it.
fn record_exit(dir: &Path, rank: u32, code: i32) {
    let path = exit_file(dir, rank);
    let temporary = path.with_extension("exit.tmp");
    let written =
        fs::write(&temporary, format!("{code}\n")).and_then(|()| fs::rename(&temporary, &path));
    if let Err(error) = written {
        eprintln!(
            "splinter: recording the exit of rank {rank} in {}: {error}",
            dir.display()
        );
    }
}

fn write_launch(launch: &Launch) -> Result<(), JobError> {
    let path = launch.dir.join(LAUNCH_FILE);
    let temporary = launch.dir.join(format!("{LAUNCH_FILE}.tmp"));
    let body = serde_json::to_vec_pretty(launch).map_err(|e| {
        JobError::io(
            "serialising the launch",
            io::Error::new(io::ErrorKind::InvalidData, e),
        )
    })?;
    fs::write(&temporary, &body)
        .map_err(|e| JobError::io(format!("writing {}", temporary.display()), e))?;
    fs::rename(&temporary, &path)
        .map_err(|e| JobError::io(format!("renaming {}", temporary.display()), e))
}

/// `SIGTERM` and `SIGKILL` are the two stops a worker gets.
const SIGTERM: libc::c_int = 15;
/// The stop that cannot be ignored.
const SIGKILL: libc::c_int = 9;

fn signal(pid: u32, signal: libc::c_int) -> Result<(), JobError> {
    let Ok(pid) = libc::pid_t::try_from(pid) else {
        return Err(JobError::io(
            "signalling a worker",
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "the process id does not fit pid_t",
            ),
        ));
    };
    // The worker was made its own session leader, so the negative pid names its
    // whole group: a wrapper script's children stop with it instead of being
    // orphaned while the launch reads as stopped.
    // SAFETY: `kill` with a process group the launch created and a constant signal.
    let rc = unsafe { libc::kill(-pid, signal) };
    if rc == 0 {
        return Ok(());
    }
    let error = io::Error::last_os_error();
    // A worker that already left is not a failure of the stop.
    if error.raw_os_error() == Some(libc::ESRCH) {
        return Ok(());
    }
    Err(JobError::io(format!("signalling worker {pid}"), error))
}

/// Whether `pid` is a live process (the launch's own pid, never a reused one
/// within the window a job lives).
///
/// A worker the launcher started but did not wait for is a zombie: it answers
/// signal 0 although it has exited. `/proc` reports its state, so a stopped
/// worker reads as gone even before its parent reaps it.
fn process_alive(pid: u32) -> bool {
    let Ok(native) = libc::pid_t::try_from(pid) else {
        return false;
    };
    // SAFETY: signal 0 checks for existence and permission; it sends nothing.
    let rc = unsafe { libc::kill(native, 0) };
    if rc != 0 && io::Error::last_os_error().raw_os_error() != Some(libc::EPERM) {
        return false;
    }
    match fs::read_to_string(format!("/proc/{pid}/stat")) {
        Ok(stat) => {
            // The state is the first field after the parenthesised command name.
            let state = stat
                .rsplit_once(')')
                .and_then(|(_, rest)| rest.split_whitespace().next())
                .unwrap_or("R");
            state != "Z" && state != "X"
        }
        // No procfs: the signal check is the best answer available.
        Err(_) => true,
    }
}
