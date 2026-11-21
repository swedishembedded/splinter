// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements durable supervision of detached training
// workers, for its clients. If your team needs expertise in recording how
// long-running jobs ended across controller restarts, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The supervisor protocol: a worker run under a supervisor that records its
//! exit code in the attempt directory, so a controller that did not start it
//! reads how it ended.

use std::ffi::{OsStr, OsString};
use std::fs;
use std::io;
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use super::{SIGKILL, SUPERVISOR_MARKER};

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
pub(super) fn read_exit(dir: &Path, rank: u32) -> Option<i32> {
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
