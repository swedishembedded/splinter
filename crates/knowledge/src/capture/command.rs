// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements bounded, reproducible capture of tool runs
// as learning evidence, for its clients. If your team needs expertise in
// learning command-line tools or process supervision, you can procure our
// services by sending an email to info@swedishembedded.com.

//! One run of a program captured as a command source: how Splinter will
//! learn a command-line tool, the captured output being the evidence later
//! answers are checked against.
//!
//! The program runs directly (no shell) with exactly the environment it is
//! given, standard input closed, in its own process group. Each output
//! stream is kept up to a cap and read to its end past it, so the program
//! never blocks on a full pipe; a stream cut at the cap is recorded as
//! truncated. A run that outlives its timeout has its whole process group
//! killed and is recorded as timed out.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use splinter_store::clock::Clock;
use splinter_store::source::{CapturedSource, Origin, PartContent};

use super::{io, is_text, utf8, CaptureError};

/// The default time a run may take before it is killed.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// The default cap on each captured output stream, in bytes.
pub const DEFAULT_OUTPUT_CAP: usize = 1024 * 1024;

/// The variables [`default_environment`] passes through from the caller's
/// environment.
pub const DEFAULT_ENV_ALLOWLIST: &[&str] = &["PATH", "HOME", "LANG"];

/// The `TERM` [`default_environment`] sets: no colour, no cursor control,
/// so captured output is plain text.
pub const DEFAULT_TERM: &str = "dumb";

/// How long a killed run's output may take to close before the capture
/// gives up on it.
const KILL_GRACE: Duration = Duration::from_secs(2);

/// How often a running capture checks whether the run has finished.
const POLL: Duration = Duration::from_millis(10);

/// One run to capture.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommandSpec {
    /// The program and its arguments; the program is looked up on the
    /// `PATH` in [`CommandSpec::env`] when it has no `/`.
    pub argv: Vec<String>,
    /// The working directory.
    pub cwd: PathBuf,
    /// The complete environment of the run; nothing else is inherited.
    pub env: BTreeMap<String, String>,
    /// How long the run may take before its process group is killed.
    pub timeout: Duration,
    /// The most bytes kept of each output stream.
    pub output_cap: usize,
}

impl CommandSpec {
    /// A run of `argv` in `cwd` with environment `env`, under
    /// [`DEFAULT_TIMEOUT`] and [`DEFAULT_OUTPUT_CAP`].
    #[must_use]
    pub fn new(argv: Vec<String>, cwd: impl Into<PathBuf>, env: BTreeMap<String, String>) -> Self {
        Self {
            argv,
            cwd: cwd.into(),
            env,
            timeout: DEFAULT_TIMEOUT,
            output_cap: DEFAULT_OUTPUT_CAP,
        }
    }
}

/// The default environment of a run: the [`DEFAULT_ENV_ALLOWLIST`]
/// variables found in `parent` (the caller's environment, as the caller
/// reads it) and `TERM` set to [`DEFAULT_TERM`]. Everything else in
/// `parent` is dropped.
#[must_use]
pub fn default_environment(
    parent: impl IntoIterator<Item = (String, String)>,
) -> BTreeMap<String, String> {
    let mut env: BTreeMap<String, String> = parent
        .into_iter()
        .filter(|(name, _)| DEFAULT_ENV_ALLOWLIST.contains(&name.as_str()))
        .collect();
    env.insert("TERM".into(), DEFAULT_TERM.into());
    env
}

/// One output stream as captured: at most the cap, and whether it was cut.
struct Stream {
    bytes: Vec<u8>,
    truncated: bool,
}

/// Reads `reader` to its end, keeping the first `cap` bytes.
fn read_bounded(mut reader: impl Read, cap: usize) -> std::io::Result<Stream> {
    let mut bytes = Vec::new();
    let mut truncated = false;
    let mut buf = [0u8; 8192];
    loop {
        let n = match reader.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        };
        let room = cap - bytes.len();
        if n > room {
            truncated = true;
        }
        bytes.extend_from_slice(&buf[..n.min(room)]);
    }
    if truncated {
        // A cut through a multi-byte character would make text output read
        // as binary; drop the partial character instead.
        if let Err(e) = std::str::from_utf8(&bytes) {
            if e.error_len().is_none() {
                bytes.truncate(e.valid_up_to());
            }
        }
    }
    Ok(Stream { bytes, truncated })
}

fn reader(stream: impl Read + Send + 'static, cap: usize) -> JoinHandle<std::io::Result<Stream>> {
    std::thread::spawn(move || read_bounded(stream, cap))
}

/// Kills the run's whole process group, so a process it started cannot
/// keep running, or keep its output open, past the timeout.
fn kill_group(child: &mut Child) {
    #[cfg(unix)]
    if let Ok(group) = libc::pid_t::try_from(child.id()) {
        // SAFETY: `killpg` takes plain integers and only sends a signal.
        // `group` is the process group the child leads (it was spawned with
        // `process_group(0)`); it stays reserved while any member lives,
        // and an error (the group already gone) leaves nothing to do.
        unsafe {
            libc::killpg(group, libc::SIGKILL);
        }
    }
    // The child itself, for a platform without process groups; an error
    // means it has already exited.
    let _ = child.kill();
}

/// Captures one run of `spec` as a command source with two parts,
/// `stdout` and `stderr` (`text/plain` when UTF-8 text, else
/// `application/octet-stream`). The origin records the argv, the absolute
/// working directory, the exit code, whether the run timed out, and
/// whether either stream was truncated at the cap. A program that exits
/// on time but leaves a process holding its output open is timed out too.
pub fn capture_command(
    spec: &CommandSpec,
    clock: &dyn Clock,
) -> Result<CapturedSource, CaptureError> {
    let Some(program) = spec.argv.first() else {
        return Err(CaptureError::EmptyArgv);
    };
    if spec.timeout.is_zero() {
        return Err(CaptureError::ZeroTimeout);
    }
    let cwd = spec.cwd.canonicalize().map_err(io(&spec.cwd))?;
    let mut command = Command::new(program);
    command
        .args(&spec.argv[1..])
        .current_dir(&cwd)
        .env_clear()
        .envs(&spec.env)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    std::os::unix::process::CommandExt::process_group(&mut command, 0);

    let started = Instant::now();
    let mut child = command.spawn().map_err(|source| CaptureError::Spawn {
        program: program.clone(),
        source,
    })?;
    let (Some(stdout), Some(stderr)) = (child.stdout.take(), child.stderr.take()) else {
        unreachable!("both output streams were configured as pipes")
    };
    let readers = [
        reader(stdout, spec.output_cap),
        reader(stderr, spec.output_cap),
    ];
    let failed = |source| CaptureError::Wait {
        program: program.clone(),
        source,
    };

    let deadline = started + spec.timeout;
    let mut status: Option<ExitStatus> = None;
    let mut timed_out = false;
    loop {
        if status.is_none() {
            status = child.try_wait().map_err(failed)?;
        }
        if status.is_some() && readers.iter().all(JoinHandle::is_finished) {
            break;
        }
        let now = Instant::now();
        if now >= deadline {
            kill_group(&mut child);
            timed_out = true;
            break;
        }
        std::thread::sleep(POLL.min(deadline - now));
    }
    if status.is_none() {
        status = Some(child.wait().map_err(failed)?);
    }
    let grace = Instant::now() + KILL_GRACE;
    while !readers.iter().all(JoinHandle::is_finished) {
        if Instant::now() >= grace {
            // The reader threads are left to finish when the process that
            // holds the pipe does.
            return Err(CaptureError::OutputNotClosed {
                program: program.clone(),
            });
        }
        std::thread::sleep(POLL);
    }
    let [stdout, stderr] = readers.map(|handle| {
        handle
            .join()
            .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
    });
    let stdout = stdout.map_err(failed)?;
    let stderr = stderr.map_err(failed)?;

    let origin = Origin::Command {
        argv: spec.argv.clone(),
        cwd: utf8(&cwd)?.to_string(),
        exit_code: status.and_then(|s| s.code()),
        timed_out,
        stdout_truncated: stdout.truncated,
        stderr_truncated: stderr.truncated,
    };
    let part = |name: &str, stream: Stream| PartContent {
        name: name.to_string(),
        media_type: if is_text(&stream.bytes) {
            "text/plain"
        } else {
            "application/octet-stream"
        }
        .to_string(),
        bytes: stream.bytes,
    };
    Ok(CapturedSource::new(
        origin,
        vec![part("stdout", stdout), part("stderr", stderr)],
        clock,
    )?)
}
