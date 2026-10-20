// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements process isolation for model-written code,
// for its clients. If your team needs expertise in keeping a language-model
// agent from seeing the other processes of its host then you can procure our
// services by sending an email to info@swedishembedded.com.

//! A private process view for a command: a user and pid namespace with its
//! own `/proc`, so that no other process of the host (its environment,
//! command line, memory maps) is visible to the command at all.
//!
//! Landlock cannot do this: system tools read `/proc/self`, which names a
//! different directory in every process, so `/proc` cannot be allowed for the
//! command and withheld for the other processes of its user. The namespace
//! can: the command's `/proc` holds only the command's own processes.
//!
//! [`enter_private_proc`] runs in the child between `fork` and `exec` and
//! makes only system calls (no allocation). The process that calls it becomes
//! a relay and never returns: it waits for a reaper that is process 1 of the
//! new namespace, which in turn waits for the process that goes on to `exec`
//! (so the command is never process 1, which would ignore signals it has no
//! handler for). When the reaper exits, the kernel kills everything left in
//! the namespace, so a command leaves no daemon behind. The command's fate is
//! relayed to the supervisor: its exit status, or, for a command killed by a
//! signal, an exit status of 128 plus the signal that the relay turns back
//! into that signal on itself (the shell's convention; a command that itself
//! exits with such a status is reported as the signal).
//!
//! The same chain contains a command that is not confined: whatever it
//! starts, however it detaches (`setsid`, double fork, a parent that exits
//! first), is in the namespace and ends with it, so nothing can be orphaned
//! out of a run. The reaper also dies with the supervisor (`PR_SET_PDEATHSIG`
//! on the relay's child), which takes the namespace with it. A host that
//! forbids user namespaces runs an unconfined command without one
//! ([`Need::IfAvailable`]); a confined command is refused instead.
//!
//! The user in the namespace has the user's own numeric id, so files look as
//! they do outside, and loses its capabilities when the command is executed.

use std::io;

const PROC: &[u8] = b"/proc\0";

fn last_error() -> io::Error {
    io::Error::last_os_error()
}

/// Writes `content` to the file `path` (NUL-terminated).
fn write_file(path: &[u8], content: &[u8]) -> io::Result<()> {
    // SAFETY: `path` is NUL-terminated and `content` is a valid buffer.
    unsafe {
        let fd = libc::open(path.as_ptr().cast(), libc::O_WRONLY | libc::O_CLOEXEC);
        if fd < 0 {
            return Err(last_error());
        }
        let written = libc::write(fd, content.as_ptr().cast(), content.len());
        let result = if written < 0 || written as usize != content.len() {
            Err(last_error())
        } else {
            Ok(())
        };
        libc::close(fd);
        result
    }
}

/// Writes `"<id> <id> 1\n"` into `buf` without allocating, returning its length.
fn identity_map(id: u32, buf: &mut [u8; 48]) -> usize {
    let mut digits = [0u8; 10];
    let (mut n, mut count) = (id, 0);
    while n > 0 || count == 0 {
        digits[count] = b'0' + (n % 10) as u8;
        n /= 10;
        count += 1;
    }
    let mut at = 0;
    for _ in 0..2 {
        for digit in digits[..count].iter().rev() {
            buf[at] = *digit;
            at += 1;
        }
        buf[at] = b' ';
        at += 1;
    }
    buf[at] = b'1';
    buf[at + 1] = b'\n';
    at + 2
}

/// Waits for `child`, ending the calling process with its fate.
///
/// First it lets go of every descriptor but the standard three. The one that
/// reports a failed `exec` to the supervisor is closed by the `exec` itself,
/// and the supervisor's `spawn` waits for that; a relay that kept a copy open
/// would hold `spawn` until the command had ended, and a command that waits
/// for the input the supervisor writes after `spawn` would never end.
fn relay_and_exit(child: libc::pid_t, reaper: bool) -> ! {
    // SAFETY: only close_range, wait, kill, signal and exit calls on plain
    // integers.
    unsafe {
        libc::syscall(
            libc::SYS_close_range,
            3 as libc::c_uint,
            libc::c_uint::MAX,
            0 as libc::c_uint,
        );
        let mut status = 0;
        loop {
            // The reaper waits for anyone, so orphans in its namespace are
            // collected; the relay waits for its one child.
            let waited = libc::waitpid(if reaper { -1 } else { child }, &mut status, 0);
            if waited == child {
                break;
            }
            if waited < 0 && *libc::__errno_location() != libc::EINTR {
                libc::_exit(127);
            }
        }
        let code = if libc::WIFEXITED(status) {
            libc::WEXITSTATUS(status)
        } else if libc::WIFSIGNALED(status) {
            128 + libc::WTERMSIG(status)
        } else {
            127
        };
        if !reaper && code > 128 && code < 128 + 65 {
            libc::signal(code - 128, libc::SIG_DFL);
            let mut all: libc::sigset_t = std::mem::zeroed();
            libc::sigemptyset(&mut all);
            libc::sigaddset(&mut all, code - 128);
            libc::sigprocmask(libc::SIG_UNBLOCK, &all, std::ptr::null_mut());
            libc::kill(libc::getpid(), code - 128);
        }
        libc::_exit(code)
    }
}

/// Whether a host that cannot create the namespace is an error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Need {
    /// The command must run in the namespace or not at all.
    Always,
    /// The command runs without the namespace where the host forbids it.
    IfAvailable,
}

/// Enters a new user and pid namespace with a fresh `/proc`. Returns in the
/// process that goes on to `exec` the command; every other process of the
/// chain ends here. See the module documentation.
pub fn enter_private_proc(need: Need) -> io::Result<()> {
    // SAFETY: plain system calls on static, NUL-terminated strings and
    // stack buffers; nothing allocates.
    unsafe {
        let (uid, gid) = (libc::getuid(), libc::getgid());
        if libc::unshare(libc::CLONE_NEWUSER | libc::CLONE_NEWNS | libc::CLONE_NEWPID) != 0 {
            // Nothing has changed yet, so the command can go on without.
            return match need {
                Need::Always => Err(last_error()),
                Need::IfAvailable => Ok(()),
            };
        }
        let mut map = [0u8; 48];
        write_file(b"/proc/self/setgroups\0", b"deny")?;
        let len = identity_map(gid, &mut map);
        write_file(b"/proc/self/gid_map\0", &map[..len])?;
        let len = identity_map(uid, &mut map);
        write_file(b"/proc/self/uid_map\0", &map[..len])?;

        // The relay: the next fork is process 1 of the new namespace.
        match libc::fork() {
            -1 => return Err(last_error()),
            0 => {}
            child => relay_and_exit(child, false),
        }
        // Process 1: dies with the relay (the supervisor's own child, which
        // is killed when the supervisor dies), taking the namespace along.
        if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
            libc::_exit(126);
        }
        // A private mount of `/proc`, then the process that execs.
        if libc::mount(
            std::ptr::null(),
            c"/".as_ptr(),
            std::ptr::null(),
            libc::MS_REC | libc::MS_PRIVATE,
            std::ptr::null(),
        ) != 0
            || libc::mount(
                c"proc".as_ptr(),
                PROC.as_ptr().cast(),
                c"proc".as_ptr(),
                libc::MS_NOSUID | libc::MS_NODEV | libc::MS_NOEXEC,
                std::ptr::null(),
            ) != 0
        {
            libc::_exit(126);
        }
        match libc::fork() {
            -1 => libc::_exit(126),
            0 => Ok(()),
            command => relay_and_exit(command, true),
        }
    }
}
