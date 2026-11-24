// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements restart-safe controllers whose leases hold
// for as long as their work runs, for its clients. If your team needs
// expertise in durable workflow control then you can procure our services by
// sending an email to info@swedishembedded.com.

//! The controller's heartbeat: a stage may run for longer than one lease (a
//! training job runs for hours), so while a controller lives its lease is
//! renewed through its fence every third of the lease. Without it a second
//! controller would take a lapsed lease from a live one and both would
//! compute the same stage. A renewal refused because another controller
//! took the lease ends the heartbeat; the live controller then meets the
//! stale fence at its next write and stops.

use std::path::Path;
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::Duration;

use splinter_store::factory::runstate::{Fence, RunState};

use splinter_core::clock::unix_now;

use super::RunError;

/// The heartbeat of one controller's lease; stopped when dropped.
pub(crate) struct Heartbeat {
    stop: Option<mpsc::Sender<()>>,
    thread: Option<JoinHandle<()>>,
}

impl Heartbeat {
    /// Starts renewing `fence` on the run state of `run` for `lease_secs`
    /// seconds at a time.
    pub(crate) fn start(run: &Path, fence: Fence, lease_secs: u64) -> Result<Self, RunError> {
        // A connection of its own: the controller's is not shared across
        // threads.
        let state = RunState::open(run)?;
        let (stop, stopped) = mpsc::channel::<()>();
        let period = Duration::from_millis(lease_secs.saturating_mul(1000) / 3)
            .max(Duration::from_millis(50));
        let thread = std::thread::Builder::new()
            .name("splinter-lease".into())
            .spawn(move || {
                while let Err(mpsc::RecvTimeoutError::Timeout) = stopped.recv_timeout(period) {
                    if state.renew(&fence, lease_secs, unix_now()).is_err() {
                        return;
                    }
                }
            })
            .map_err(|e| RunError::Io {
                path: run.to_path_buf(),
                message: format!("start the lease heartbeat: {e}"),
            })?;
        Ok(Self {
            stop: Some(stop),
            thread: Some(thread),
        })
    }
}

impl Drop for Heartbeat {
    fn drop(&mut self) {
        drop(self.stop.take());
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// This controller's name on a lease: `host:pid@pidns:start`, the host the
/// caller names, this process, the pid namespace it is numbered in and when
/// it started (clock ticks since boot): what a later controller needs to
/// prove the holder gone (`holder_gone`). Where `/proc` cannot say, the
/// namespace and start are `?` and no takeover is ever judged.
#[must_use]
pub fn controller_owner(host: &str) -> String {
    let pid = std::process::id();
    let (namespace, start) = (pid_namespace(), start_of(pid));
    format!(
        "{host}:{pid}@{}:{}",
        namespace.as_deref().unwrap_or("?"),
        start.map_or_else(|| "?".to_owned(), |s| s.to_string())
    )
}

/// The pid namespace this process is numbered in (`pid:[INODE]`).
fn pid_namespace() -> Option<String> {
    std::fs::read_link("/proc/self/ns/pid")
        .ok()
        .map(|p| p.to_string_lossy().into_owned())
}

/// When process `pid` of this namespace started, in clock ticks since boot
/// (field 22 of `/proc/PID/stat`); `None` when there is no such process.
fn start_of(pid: u32) -> Option<u64> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    // The command name is parenthesised and may hold spaces: fields count
    // from after its closing parenthesis (field 3 onwards).
    let after = stat.rsplit_once(')')?.1;
    after.split_whitespace().nth(19)?.parse().ok()
}

/// A lease owner's parts: host, pid, pid namespace, start.
fn parts(owner: &str) -> Option<(&str, u32, &str, &str)> {
    let (host, rest) = owner.rsplit_once(':').and_then(|(head, start)| {
        let (host_pid, namespace) = head.split_once('@')?;
        let (host, pid) = host_pid.rsplit_once(':')?;
        Some((host, (pid, namespace, start)))
    })?;
    let (pid, namespace, start) = rest;
    Some((host, pid.parse().ok()?, namespace, start))
}

/// Whether the lease `holder` is provably a controller of this host that no
/// longer runs, judged by this controller `owner`: the same host and pid
/// namespace, and its pid either gone or now another process (its start
/// differs). Anything less (another namespace, a start `/proc` could not
/// read, an owner of another form) is not proof: the lease is waited out.
pub(crate) fn holder_gone(owner: &str, holder: &str) -> bool {
    holder_gone_by(owner, holder, &|pid| start_of(pid))
}

fn holder_gone_by(owner: &str, holder: &str, start: &dyn Fn(u32) -> Option<u64>) -> bool {
    let (Some(mine), Some(theirs)) = (parts(owner), parts(holder)) else {
        return false;
    };
    let (host, pid, namespace, started) = theirs;
    if host != mine.0 || namespace != mine.2 || namespace == "?" || started == "?" {
        return false;
    }
    match start(pid) {
        None => true,
        Some(now) => started.parse::<u64>().is_ok_and(|then| then != now),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A holder is taken over only when it is provably gone: the same host
    /// and namespace with its pid absent or reused; another namespace, an
    /// unreadable start or a foreign owner is waited out.
    #[test]
    fn a_holder_is_taken_over_only_when_provably_gone() {
        let me = "node:10@pid:[7]:500";
        let alive = |pid: u32| (pid == 42).then_some(900);
        assert!(
            !holder_gone_by(me, "node:42@pid:[7]:900", &alive),
            "the same process runs"
        );
        assert!(
            holder_gone_by(me, "node:42@pid:[7]:100", &alive),
            "its pid is another process now"
        );
        assert!(
            holder_gone_by(me, "node:43@pid:[7]:100", &alive),
            "its pid is gone"
        );
        assert!(
            !holder_gone_by(me, "node:43@pid:[9]:100", &alive),
            "another namespace: not ours to judge"
        );
        assert!(
            !holder_gone_by(me, "other:43@pid:[7]:100", &alive),
            "another host"
        );
        assert!(
            !holder_gone_by(me, "node:43@?:?", &alive),
            "nothing to prove it by"
        );
        assert!(
            !holder_gone_by(me, "node:43", &alive),
            "an owner of another form"
        );
        assert!(!holder_gone_by("node:10@?:?", "node:43@?:?", &alive));
    }

    /// This process's own name proves it alive to itself.
    #[test]
    fn this_controller_is_never_judged_gone() {
        let me = controller_owner("node");
        assert!(!holder_gone(&me, &me), "{me}");
    }
}
