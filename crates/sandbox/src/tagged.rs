// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements sandboxes that leave no process of a run
// behind, for its clients. If your team needs expertise in running untrusted
// builds and tests safely, you can procure our services by sending an email
// to info@swedishembedded.com.

//! A run's processes found by the token in their environment, where no
//! private pid namespace holds them.
//!
//! A process that leaves the run's process group and is orphaned (`setsid`,
//! a daemon's double fork) is linked to the run neither by group nor by
//! parent any more, and outlives it; with a pid namespace of its own the run
//! takes it down when its init ends ([`crate::namespace`]), but not every
//! host offers one. What every descendant still carries is the environment
//! it was started with. A run is tagged with [`TOKEN_VAR`], and:
//!
//! * when it ends, every process whose environment holds its tag is killed
//!   ([`kill_tagged`]), orphans included;
//! * while it runs, a watchdog ([`Watchdog`]) waits on a pipe only the
//!   supervisor holds; when the supervisor dies (killed, crashed) the pipe
//!   closes and the watchdog kills the tagged processes before it exits.
//!
//! A program that clears its own environment before it detaches escapes the
//! tag: this guards against the accidents of ordinary programs (a test
//! runner's orphaned interpreter), not against a program that hides.

use std::io::Read;
use std::os::unix::process::CommandExt as _;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

/// The environment variable a run's processes carry its token in.
pub const TOKEN_VAR: &str = "SPLINTER_RUN";

/// Rounds of scanning and killing: a process may fork while its siblings
/// are killed, and its child carries the tag too.
const ROUNDS: usize = 5;

/// A token no other run of this host holds now: this process, a counter and
/// the time.
#[must_use]
pub fn token() -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    format!(
        "{}-{}-{nanos}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    )
}

/// The processes whose environment holds `TOKEN_VAR=token`, as far as this
/// process may read their environments.
fn tagged(token: &str) -> Vec<libc::pid_t> {
    let wanted = format!("{TOKEN_VAR}={token}");
    let Ok(entries) = std::fs::read_dir("/proc") else {
        return Vec::new();
    };
    let mut found = Vec::new();
    for entry in entries.flatten() {
        let Some(pid) = entry
            .file_name()
            .to_str()
            .and_then(|n| n.parse::<libc::pid_t>().ok())
        else {
            continue;
        };
        let mut environ = Vec::new();
        let read = std::fs::File::open(entry.path().join("environ"))
            .and_then(|mut f| f.read_to_end(&mut environ));
        if read.is_ok()
            && environ
                .split(|b| *b == 0)
                .any(|pair| pair == wanted.as_bytes())
        {
            found.push(pid);
        }
    }
    found
}

/// Kills every process tagged with `token`, until none is left or the
/// rounds are spent.
pub fn kill_tagged(token: &str) {
    for _ in 0..ROUNDS {
        let found = tagged(token);
        if found.is_empty() {
            return;
        }
        for pid in found {
            // SAFETY: `kill` takes plain integers and only sends a signal; a
            // process already gone is an error with nothing to do.
            unsafe {
                libc::kill(pid, libc::SIGKILL);
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
}

/// A process that kills a run's tagged processes when the supervisor dies:
/// it reads a pipe only the supervisor holds, and a closed pipe is its
/// order. [`Watchdog::stop`] closes it after the supervisor's own cleanup.
pub struct Watchdog {
    child: Child,
    order: Option<ChildStdin>,
}

/// The watchdog's program: wait for the pipe; a supervisor that cleaned up
/// itself says `done`, and a closed pipe without it (the supervisor died)
/// is the order to kill every process whose environment holds the run's tag.
const WATCH: &str = r#"read -r said
[ "$said" = done ] && exit 0
for e in /proc/[0-9]*/environ; do
    if tr '\0' '\n' < "$e" 2>/dev/null | grep -qxF "$WATCHED"; then
        p=${e#/proc/}
        kill -9 "${p%/environ}" 2>/dev/null
    fi
done"#;

impl Watchdog {
    /// Starts the watchdog of the run tagged `token`; `None` where no shell
    /// can be started (the run then relies on the supervisor's own cleanup).
    #[must_use]
    pub fn start(token: &str) -> Option<Self> {
        let mut child = Command::new("/bin/sh")
            .args(["-c", WATCH])
            .env_clear()
            .env("PATH", crate::host::DEFAULT_SEARCH_PATH)
            .env("WATCHED", format!("{TOKEN_VAR}={token}"))
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .process_group(0)
            .spawn()
            .ok()?;
        let order = child.stdin.take();
        Some(Self { child, order })
    }

    /// The run is over and the supervisor cleaned it up: the watchdog ends
    /// without scanning.
    pub fn stop(mut self) {
        if let Some(mut order) = self.order.take() {
            use std::io::Write as _;
            let _ = order.write_all(b"done\n");
        }
        let _ = self.child.wait();
    }
}
