// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements bounded, supervised execution of untrusted
// programs for its clients. If your team needs expertise in process
// supervision or agent sandboxing, you can procure our services by sending
// an email to info@swedishembedded.com.

//! One bounded run of a program: the process-running core every caller in
//! Splinter shares (a command capture, a sandboxed code call, a runtime's
//! version probe).
//!
//! The program runs directly (no shell) with exactly the environment it is
//! given, in its own process group, with its standard input either closed
//! or fed from bytes the caller supplies. Each output stream is kept up to
//! a cap and read to its end past it, so the program never blocks on a
//! full pipe; a stream cut at the cap is reported as truncated. A run that
//! outlives its timeout is timed out. Either way, once the run ends its
//! whole process group is killed, and so is every descendant of the program
//! still linked to it by parent (a process that moved to a session or group of
//! its own is found that way). What a run cannot stop is a process that was
//! orphaned before the end (its parent exited first): this bounds a run, it
//! does not confine one.
//!
//! [`ResourceLimits`], when given, are applied with `setrlimit` in the
//! child between `fork` and `exec`, so they bind the program and everything
//! it starts.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::landlock::Confinement;

/// How long a killed run's output may take to close before the run gives
/// up on it.
const KILL_GRACE: Duration = Duration::from_secs(2);

/// How often a running program is checked for having finished.
const POLL: Duration = Duration::from_millis(10);

/// Kernel resource limits for a run, each applied as both the soft and the
/// hard limit except the CPU time, whose hard limit is one second above the
/// soft one so the program is sent `SIGXCPU` before `SIGKILL`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceLimits {
    /// CPU time, in seconds (`RLIMIT_CPU`), per process.
    pub cpu_seconds: u64,
    /// Virtual address space, in bytes (`RLIMIT_AS`), per process.
    pub address_space_bytes: u64,
    /// The largest file a process may write, in bytes (`RLIMIT_FSIZE`).
    pub file_size_bytes: u64,
}

impl ResourceLimits {
    /// Applies the limits to the calling process, plus a zero core-file
    /// size so a program killed by a limit leaves no core dump behind.
    ///
    /// Only async-signal-safe calls (`setrlimit`), so it may run in a child
    /// between `fork` and `exec`.
    fn apply(&self) -> std::io::Result<()> {
        let cpu_hard = self.cpu_seconds.saturating_add(1);
        for (resource, soft, hard) in [
            (libc::RLIMIT_CPU, self.cpu_seconds, cpu_hard),
            (
                libc::RLIMIT_AS,
                self.address_space_bytes,
                self.address_space_bytes,
            ),
            (
                libc::RLIMIT_FSIZE,
                self.file_size_bytes,
                self.file_size_bytes,
            ),
            (libc::RLIMIT_CORE, 0, 0),
        ] {
            let limit = libc::rlimit {
                rlim_cur: soft,
                rlim_max: hard,
            };
            // SAFETY: `setrlimit` reads the struct it is handed and changes
            // only this process's limits.
            if unsafe { libc::setrlimit(resource, &limit) } != 0 {
                return Err(std::io::Error::last_os_error());
            }
        }
        Ok(())
    }
}

/// One run of a program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProcessSpec {
    /// The program and its arguments; the program is looked up on the
    /// `PATH` in [`ProcessSpec::env`] when it has no `/`.
    pub argv: Vec<String>,
    /// The working directory.
    pub cwd: PathBuf,
    /// The complete environment of the run; nothing else is inherited.
    pub env: BTreeMap<String, String>,
    /// What the program reads on its standard input; `None` closes it.
    pub stdin: Option<Vec<u8>>,
    /// How long the run may take before its process group is killed.
    pub timeout: Duration,
    /// The most bytes kept of each output stream (head and tail; see
    /// [`Stream`]), plus a short marker line when the middle is elided.
    pub output_cap: usize,
    /// Kernel limits applied to the program, when set.
    pub resources: Option<ResourceLimits>,
    /// The only paths the program may reach, when set; the run fails where
    /// the kernel cannot confine it.
    pub confinement: Option<Confinement>,
}

impl ProcessSpec {
    /// A run of `argv` in `cwd` with environment `env`, standard input
    /// closed and no resource limits.
    #[must_use]
    pub fn new(
        argv: Vec<String>,
        cwd: impl Into<PathBuf>,
        env: BTreeMap<String, String>,
        timeout: Duration,
        output_cap: usize,
    ) -> Self {
        Self {
            argv,
            cwd: cwd.into(),
            env,
            stdin: None,
            timeout,
            output_cap,
            resources: None,
            confinement: None,
        }
    }
}

/// One output stream as captured: at most the cap, kept as a head and a
/// tail with a marker line between them when the program wrote more.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stream {
    /// The bytes kept, the marker line included.
    pub bytes: Vec<u8>,
    /// Whether the program wrote more than was kept.
    pub truncated: bool,
    /// How many bytes the marker line says were left out of the middle.
    pub elided_bytes: u64,
}

/// How a run ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProcessOutput {
    /// Standard output.
    pub stdout: Stream,
    /// Standard error.
    pub stderr: Stream,
    /// The exit code, when the program exited rather than being killed.
    pub exit_code: Option<i32>,
    /// The signal that ended the program, when one did.
    pub signal: Option<i32>,
    /// Whether the timeout ended the run. A program that exits on time
    /// but leaves a process holding its output open is timed out too.
    pub timed_out: bool,
}

/// Why a run could not be carried out. A program that fails, or is killed
/// at its timeout, is an outcome ([`ProcessOutput`]), not an error.
#[derive(Debug, thiserror::Error)]
pub enum ProcessError {
    /// A run with no program.
    #[error("the command to run is empty")]
    EmptyArgv,
    /// A timeout of zero.
    #[error("the timeout must be longer than zero")]
    ZeroTimeout,
    /// The program could not be started.
    #[error("cannot start {program:?}: {source}")]
    Spawn {
        /// The program.
        program: String,
        /// The underlying error.
        source: std::io::Error,
    },
    /// The program could not be confined to its paths.
    #[error("cannot confine {program:?} to its paths: {source}")]
    Confine {
        /// The program.
        program: String,
        /// The underlying error.
        source: std::io::Error,
    },
    /// Waiting for the program, or reading its output, failed.
    #[error("running {program:?}: {source}")]
    Wait {
        /// The program.
        program: String,
        /// The underlying error.
        source: std::io::Error,
    },
    /// The program's output stayed open after it and its process group
    /// were killed: a process it started left the group and holds it.
    #[error("the output of {program:?} stayed open after its process group was killed")]
    OutputNotClosed {
        /// The program.
        program: String,
    },
}

/// The variables of `parent` (the caller's environment, as the caller reads
/// it) whose names are in `allowlist`; everything else is dropped.
#[must_use]
pub fn environment_from(
    parent: impl IntoIterator<Item = (String, String)>,
    allowlist: &[&str],
) -> BTreeMap<String, String> {
    parent
        .into_iter()
        .filter(|(name, _)| allowlist.contains(&name.as_str()))
        .collect()
}

/// Reads `reader` to its end, keeping at most `cap` bytes: all of it when it
/// fits, otherwise the first and last halves of the cap with a marker line
/// between them that says how many bytes were elided.
///
/// The end of a test runner's output is where its summary and the
/// failures it lists are, and the start is where its per-test lines begin, so
/// both are kept; memory is bounded by `cap` whatever the program writes.
fn read_bounded(mut reader: impl Read, cap: usize) -> std::io::Result<Stream> {
    let head_cap = cap / 2;
    let tail_cap = cap - head_cap;
    let mut head = Vec::new();
    let mut tail = std::collections::VecDeque::new();
    let mut dropped = 0_u64;
    let mut buf = [0u8; 8192];
    loop {
        let n = match reader.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        };
        let chunk = &buf[..n];
        let into_head = (head_cap - head.len()).min(n);
        head.extend_from_slice(&chunk[..into_head]);
        tail.extend(&chunk[into_head..]);
        if tail.len() > tail_cap {
            let excess = tail.len() - tail_cap;
            tail.drain(..excess);
            dropped += excess as u64;
        }
    }
    let tail: Vec<u8> = tail.into();
    if dropped == 0 {
        head.extend_from_slice(&tail);
        return Ok(Stream {
            bytes: head,
            truncated: false,
            elided_bytes: 0,
        });
    }
    // Cut between lines, so the reader of the text sees whole lines only;
    // output with no line end in reach is cut at a character boundary
    // instead, so text never reads as binary for a split character.
    let head_end = head
        .iter()
        .rposition(|b| *b == b'\n')
        .map_or_else(|| complete_prefix(&head), |at| at + 1);
    let tail_start = tail.iter().position(|b| *b == b'\n').map_or_else(
        || tail.iter().take_while(|b| (**b & 0xC0) == 0x80).count(),
        |at| at + 1,
    );
    let elided_bytes = dropped + (head.len() - head_end) as u64 + tail_start as u64;
    let mut bytes = head[..head_end].to_vec();
    bytes.extend_from_slice(format!("[... {elided_bytes} bytes elided ...]\n").as_bytes());
    bytes.extend_from_slice(&tail[tail_start..]);
    Ok(Stream {
        bytes,
        truncated: true,
        elided_bytes,
    })
}

/// The length of the longest prefix of `bytes` that does not end inside a
/// multi-byte character.
fn complete_prefix(bytes: &[u8]) -> usize {
    match std::str::from_utf8(bytes) {
        Err(e) if e.error_len().is_none() => e.valid_up_to(),
        _ => bytes.len(),
    }
}

fn reader(stream: impl Read + Send + 'static, cap: usize) -> JoinHandle<std::io::Result<Stream>> {
    std::thread::spawn(move || read_bounded(stream, cap))
}

/// Feeds `bytes` to the program's standard input and closes it. A program
/// that exits (or is killed) without reading it all closes the pipe, which
/// is not an error of the run.
fn writer(mut stdin: impl Write + Send + 'static, bytes: Vec<u8>) {
    std::thread::spawn(move || {
        let _ = stdin.write_all(&bytes);
    });
}

/// Whether the child has exited, without reaping it: while it is an
/// unreaped zombie its process id, and so its process group id, cannot be
/// reused, which is what makes [`kill_group`] safe after it exits.
fn has_exited(child: &Child) -> std::io::Result<bool> {
    let pid = libc::id_t::from(child.id());
    loop {
        // SAFETY: an all-zero `siginfo_t` is a valid value of the plain C
        // struct `waitid` fills in.
        let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
        // SAFETY: `waitid` writes only into `info`; `WNOWAIT` leaves the
        // child waitable, so `Child::wait` still reaps it.
        let rc = unsafe {
            libc::waitid(
                libc::P_PID,
                pid,
                &mut info,
                libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
            )
        };
        if rc == 0 {
            // SAFETY: `waitid` succeeded, so `info` holds a child-state
            // record, whose `si_pid` is zero when no child has changed state.
            return Ok(unsafe { info.si_pid() } != 0);
        }
        let err = std::io::Error::last_os_error();
        if err.kind() != std::io::ErrorKind::Interrupted {
            return Err(err);
        }
    }
}

/// Every process that descends from `root` by parent links, read from
/// `/proc` (none where it is not mounted). A descendant that left the run's
/// process group (`setsid`, `timeout`) is found this way as long as the chain
/// of parents to `root` is unbroken.
fn descendants(root: libc::pid_t) -> Vec<libc::pid_t> {
    let Ok(entries) = std::fs::read_dir("/proc") else {
        return Vec::new();
    };
    let mut children: std::collections::HashMap<libc::pid_t, Vec<libc::pid_t>> = Default::default();
    for entry in entries.flatten() {
        let Some(pid) = entry
            .file_name()
            .to_str()
            .and_then(|n| n.parse::<libc::pid_t>().ok())
        else {
            continue;
        };
        let Ok(stat) = std::fs::read_to_string(entry.path().join("stat")) else {
            continue;
        };
        // `pid (comm) state ppid ...`: the name may hold spaces and
        // parentheses, so the fields are read after the last `)`.
        let parent = stat
            .rsplit_once(") ")
            .and_then(|(_, rest)| rest.split_whitespace().nth(1))
            .and_then(|field| field.parse::<libc::pid_t>().ok());
        if let Some(parent) = parent {
            children.entry(parent).or_default().push(pid);
        }
    }
    let mut found = Vec::new();
    let mut frontier = vec![root];
    while let Some(parent) = frontier.pop() {
        for &child in children.get(&parent).into_iter().flatten() {
            found.push(child);
            frontier.push(child);
        }
    }
    found
}

/// Kills the run's whole process group and every descendant of its leader, so
/// a process it started cannot keep running, or keep its output open, past
/// the run, even one that moved to a session or group of its own.
///
/// The descendants are listed before anything is killed, and the group
/// (leader included) is killed first: killing a leader's children first would
/// let it finish on its own and be recorded as exited, not killed.
fn kill_group(child: &Child) {
    let Ok(group) = libc::pid_t::try_from(child.id()) else {
        return;
    };
    let descendants = descendants(group);
    // SAFETY: `killpg` takes plain integers and only sends a signal.
    // `group` is the process group the child leads (it was spawned with
    // `process_group(0)`), and the child is not reaped yet, so the id still
    // names that group; an error (no member left) leaves nothing to do.
    unsafe {
        libc::killpg(group, libc::SIGKILL);
    }
    for pid in descendants {
        // SAFETY: `kill` takes plain integers and only sends a signal; a
        // process that is already gone is an error with nothing to do.
        unsafe {
            libc::kill(pid, libc::SIGKILL);
        }
    }
}

/// Runs `spec` to its end, or to its timeout.
pub fn run(spec: &ProcessSpec) -> Result<ProcessOutput, ProcessError> {
    let Some(program) = spec.argv.first() else {
        return Err(ProcessError::EmptyArgv);
    };
    if spec.timeout.is_zero() {
        return Err(ProcessError::ZeroTimeout);
    }
    let mut command = Command::new(program);
    command
        .args(&spec.argv[1..])
        .current_dir(&spec.cwd)
        .env_clear()
        .envs(&spec.env)
        .stdin(if spec.stdin.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0);
    if let Some(resources) = spec.resources {
        // SAFETY: the closure runs in the child between fork and exec and
        // makes only async-signal-safe calls (`setrlimit`), touching no
        // memory shared with the parent.
        unsafe {
            command.pre_exec(move || resources.apply());
        }
    }

    // Built here, where it may allocate; the child only enforces it.
    let ruleset = match &spec.confinement {
        Some(confinement) => {
            Some(
                confinement
                    .prepare()
                    .map_err(|source| ProcessError::Confine {
                        program: program.clone(),
                        source,
                    })?,
            )
        }
        None => None,
    };
    if let Some(ruleset) = &ruleset {
        let fd = ruleset.raw_fd();
        // SAFETY: the closure makes only system calls (`prctl`,
        // `landlock_restrict_self`) on a descriptor the parent keeps open
        // until the child is spawned.
        unsafe {
            command.pre_exec(move || crate::landlock::enforce_fd(fd));
        }
    }

    let started = Instant::now();
    let mut child = command.spawn().map_err(|source| ProcessError::Spawn {
        program: program.clone(),
        source,
    })?;
    if let (Some(stdin), Some(bytes)) = (child.stdin.take(), spec.stdin.clone()) {
        writer(stdin, bytes);
    }
    let (Some(stdout), Some(stderr)) = (child.stdout.take(), child.stderr.take()) else {
        unreachable!("both output streams were configured as pipes")
    };
    let readers = [
        reader(stdout, spec.output_cap),
        reader(stderr, spec.output_cap),
    ];
    let failed = |source| ProcessError::Wait {
        program: program.clone(),
        source,
    };

    let deadline = started + spec.timeout;
    let mut exited = false;
    let mut timed_out = false;
    loop {
        if !exited {
            exited = has_exited(&child).map_err(failed)?;
        }
        if exited && readers.iter().all(JoinHandle::is_finished) {
            break;
        }
        let now = Instant::now();
        if now >= deadline {
            timed_out = true;
            break;
        }
        std::thread::sleep(POLL.min(deadline - now));
    }
    // The leader is not reaped yet, so the group id is still its own.
    kill_group(&child);
    let status = child.wait().map_err(failed)?;
    let grace = Instant::now() + KILL_GRACE;
    while !readers.iter().all(JoinHandle::is_finished) {
        if Instant::now() >= grace {
            // The reader threads are left to finish when the process that
            // holds the pipe does.
            return Err(ProcessError::OutputNotClosed {
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
    Ok(ProcessOutput {
        stdout: stdout.map_err(failed)?,
        stderr: stderr.map_err(failed)?,
        exit_code: status.code(),
        signal: status.signal(),
        timed_out,
    })
}
