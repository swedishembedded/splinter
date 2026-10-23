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
//! the submitting process is still alive; a reopened launch reports that a
//! worker is gone and the durable job record says what its artifacts prove.

use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::os::unix::process::CommandExt;
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
    #[must_use]
    pub fn observe(&self) -> Observation {
        let alive = self.ranks_alive();
        if alive.is_empty() {
            Observation::Exited { exit_code: None }
        } else {
            Observation::Running { ranks_alive: alive }
        }
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
    /// No worker is alive.
    Exited {
        /// The exit code, when the observing process started the workers and
        /// saw one; `None` after a controller restart, where only the durable
        /// job record can say what the run proved.
        exit_code: Option<i32>,
    },
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
#[derive(Debug, Default)]
pub struct LocalLauncher {
    children: Mutex<HashMap<PathBuf, Vec<Child>>>,
}

impl LocalLauncher {
    fn children(&self) -> std::sync::MutexGuard<'_, HashMap<PathBuf, Vec<Child>>> {
        self.children
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl JobLauncher for LocalLauncher {
    fn name(&self) -> &'static str {
        "local"
    }

    fn launch(&self, spec: &JobSpec, attempt: u32, dir: &Path) -> Result<Launch, JobError> {
        spec.validate()?;
        if let Some(existing) = Launch::open(dir)? {
            return Err(JobError::AlreadyLaunched {
                dir: dir.to_path_buf(),
                attempt: existing.attempt,
            });
        }
        fs::create_dir_all(dir)
            .map_err(|e| JobError::io(format!("creating {}", dir.display()), e))?;
        // Claim the attempt before starting anything: a crash between spawning
        // a worker and writing the launch record must not let a retry start a
        // second set of workers.
        let intent = dir.join("launch.intent");
        match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&intent)
        {
            Ok(mut file) => {
                let body = format!("{}\n{}\n", spec.digest().as_str(), attempt);
                file.write_all(body.as_bytes())
                    .map_err(|e| JobError::io(format!("writing {}", intent.display()), e))?;
            }
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
                return Err(JobError::AlreadyLaunched {
                    dir: dir.to_path_buf(),
                    attempt,
                });
            }
            Err(e) => return Err(JobError::io(format!("claiming {}", intent.display()), e)),
        }
        let world = spec.workers.world_size();
        let mut children = Vec::new();
        for rank in 0..world {
            let command = spec.command_for(rank);
            let out = fs::File::create(dir.join(format!("rank-{rank}.out")))
                .map_err(|e| JobError::io("creating a worker's stdout", e))?;
            let err = fs::File::create(dir.join(format!("rank-{rank}.err")))
                .map_err(|e| JobError::io("creating a worker's stderr", e))?;
            let mut process = Command::new(&command[0]);
            process
                .args(&command[1..])
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
                    return Observation::Exited { exit_code: exit };
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
