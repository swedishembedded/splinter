// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements filesystem confinement for code-editing
// agents, for its clients. If your team needs expertise in keeping a
// language-model agent away from material it must not read, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Confinement of a child process with Landlock, through the kernel's three
//! system calls (no library): the child can read and run what
//! [`Confinement::read_only`] names, read and write what
//! [`Confinement::read_write`] names, and reach nothing else - and, unless
//! [`Confinement::allow_network`] says otherwise, no network: TCP `bind` and
//! `connect` are denied by Landlock (ABI 4) and the creation of every
//! internet-family or packet socket is refused by a seccomp filter (which
//! also closes UDP, that Landlock does not cover). Only local (unix-domain)
//! sockets remain. A kernel that can do neither refuses the confinement
//! instead of running the child with the network open.
//!
//! The ruleset is built in the parent ([`Confinement::prepare`]: it opens
//! paths and allocates), and the child only enforces it ([`Prepared::enforce`]:
//! two system calls, safe between `fork` and `exec`). Paths that do not exist
//! are skipped. The rights a kernel does not know are left out, so the
//! confinement is as strict as the running kernel can make it (ABI 1 from
//! Linux 5.13; 2 adds `REFER`, 3 adds `TRUNCATE`).

use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};

const SYS_CREATE_RULESET: libc::c_long = 444;
const SYS_ADD_RULE: libc::c_long = 445;
const SYS_RESTRICT_SELF: libc::c_long = 446;
const CREATE_RULESET_VERSION: libc::c_ulong = 1;
const RULE_PATH_BENEATH: libc::c_int = 1;

const EXECUTE: u64 = 1 << 0;
const WRITE_FILE: u64 = 1 << 1;
const READ_FILE: u64 = 1 << 2;
const READ_DIR: u64 = 1 << 3;
const REFER: u64 = 1 << 13;
/// Network rights of ABI 4: TCP `bind` and TCP `connect`.
const NET_BIND_TCP: u64 = 1 << 0;
const NET_CONNECT_TCP: u64 = 1 << 1;
const NET_ABI: u32 = 4;
const TRUNCATE: u64 = 1 << 14;
/// Rights ABI 1 knows: bits 0 to 12.
const ABI1: u64 = (1 << 13) - 1;
/// The rights that apply to a file (a rule on a file may carry no other).
const FILE_RIGHTS: u64 = EXECUTE | WRITE_FILE | READ_FILE | TRUNCATE;

#[repr(C)]
struct RulesetAttr {
    handled_access_fs: u64,
    /// Present from ABI 4; the kernel reads it only when told the larger size.
    handled_access_net: u64,
}

/// The size of the part of [`RulesetAttr`] ABI 1 to 3 kernels know.
const RULESET_ATTR_FS_ONLY: usize = std::mem::size_of::<u64>();

#[repr(C, packed)]
struct PathBeneathAttr {
    allowed_access: u64,
    parent_fd: i32,
}

/// The newest Landlock ABI the running kernel supports, or `None` where
/// Landlock is missing or switched off.
#[must_use]
pub fn abi_version() -> Option<u32> {
    // SAFETY: a null attribute with the VERSION flag only asks for the ABI.
    let version = unsafe {
        libc::syscall(
            SYS_CREATE_RULESET,
            std::ptr::null::<RulesetAttr>(),
            0 as libc::size_t,
            CREATE_RULESET_VERSION,
        )
    };
    u32::try_from(version).ok().filter(|v| *v >= 1)
}

/// What a confined process may reach.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Confinement {
    /// Paths it may read and run programs from, and everything beneath them.
    pub read_only: Vec<PathBuf>,
    /// Paths it may read, write, create and remove beneath.
    pub read_write: Vec<PathBuf>,
    /// Whether the child may use the network. Off by default: a confinement
    /// is network-less unless it says otherwise.
    pub allow_network: bool,
    /// Whether the child gets a private user and pid namespace with its own
    /// `/proc` ([`crate::namespace`]), so no other process of the host is
    /// visible to it. Where the namespace cannot be made the child is not
    /// run. Off by default here; the process sandbox turns it on.
    pub private_proc: bool,
}

/// A ruleset ready to be enforced in a child.
pub struct Prepared {
    ruleset: OwnedFd,
    socket_filter: Option<crate::seccomp::Filter>,
    private_proc: bool,
}

impl Confinement {
    /// Builds the kernel ruleset; fails where Landlock is unavailable.
    pub fn prepare(&self) -> io::Result<Prepared> {
        let abi = abi_version().ok_or_else(|| {
            io::Error::new(io::ErrorKind::Unsupported, "Landlock is not available")
        })?;
        let mut handled = ABI1;
        if abi >= 2 {
            handled |= REFER;
        }
        if abi >= 3 {
            handled |= TRUNCATE;
        }
        let deny_network = !self.allow_network;
        if deny_network && abi < NET_ABI {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!("denying the network needs Landlock ABI {NET_ABI}; the kernel has {abi}"),
            ));
        }
        let socket_filter = deny_network
            .then(crate::seccomp::Filter::deny_inet_sockets)
            .transpose()?;
        let attr = RulesetAttr {
            handled_access_fs: handled,
            handled_access_net: if deny_network {
                NET_BIND_TCP | NET_CONNECT_TCP
            } else {
                0
            },
        };
        let attr_size = if abi >= NET_ABI {
            std::mem::size_of::<RulesetAttr>()
        } else {
            RULESET_ATTR_FS_ONLY
        };
        // SAFETY: `attr` is a valid `landlock_ruleset_attr` of the size passed.
        let fd = unsafe {
            libc::syscall(
                SYS_CREATE_RULESET,
                &attr as *const RulesetAttr,
                attr_size,
                0 as libc::c_ulong,
            )
        };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: the call succeeded, so `fd` is a new descriptor we own.
        let ruleset = unsafe { OwnedFd::from_raw_fd(fd as RawFd) };
        let read = handled & (EXECUTE | READ_FILE | READ_DIR);
        for path in &self.read_only {
            add_rule(&ruleset, path, read)?;
        }
        for path in &self.read_write {
            add_rule(&ruleset, path, handled)?;
        }
        Ok(Prepared {
            ruleset,
            socket_filter,
            private_proc: self.private_proc,
        })
    }
}

/// Allows `access` beneath `path`; a path that is not there allows nothing.
fn add_rule(ruleset: &OwnedFd, path: &Path, access: u64) -> io::Result<()> {
    let Ok(c_path) = std::ffi::CString::new(path.as_os_str().as_bytes()) else {
        return Ok(());
    };
    // SAFETY: `c_path` is a NUL-terminated path.
    let fd = unsafe { libc::open(c_path.as_ptr(), libc::O_PATH | libc::O_CLOEXEC) };
    if fd < 0 {
        let err = io::Error::last_os_error();
        return if err.kind() == io::ErrorKind::NotFound {
            Ok(())
        } else {
            Err(err)
        };
    }
    // SAFETY: `fd` was just opened and is ours.
    let opened = unsafe { OwnedFd::from_raw_fd(fd) };
    let is_dir = std::fs::metadata(path).is_ok_and(|m| m.is_dir());
    let rule = PathBeneathAttr {
        allowed_access: if is_dir { access } else { access & FILE_RIGHTS },
        parent_fd: opened.as_raw_fd(),
    };
    // SAFETY: `rule` is a valid `landlock_path_beneath_attr`, the ruleset
    // descriptor is open, and the rule's descriptor stays open for the call.
    let rc = unsafe {
        libc::syscall(
            SYS_ADD_RULE,
            ruleset.as_raw_fd(),
            RULE_PATH_BENEATH,
            &rule as *const PathBeneathAttr,
            0 as libc::c_uint,
        )
    };
    if rc < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

impl Prepared {
    /// Confines the calling process (a child between `fork` and `exec`) to
    /// the ruleset, for good. Makes only system calls and allocates nothing.
    pub fn enforce(&self) -> io::Result<()> {
        if self.private_proc {
            crate::namespace::enter_private_proc(crate::namespace::Need::Always)?;
            // The fresh `/proc` is another filesystem than the one the rules
            // were built from, so it is allowed here, read only.
            allow_read_beneath(self.ruleset.as_raw_fd(), b"/proc\0")?;
        }
        // SAFETY: `prctl` takes plain integers.
        if unsafe { libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) } != 0 {
            return Err(io::Error::last_os_error());
        }
        if let Some(filter) = &self.socket_filter {
            filter.install()?;
        }
        // SAFETY: `landlock_restrict_self` takes a descriptor and flags; the
        // ruleset descriptor is open for as long as `self` is.
        if unsafe {
            libc::syscall(
                SYS_RESTRICT_SELF,
                self.ruleset.as_raw_fd(),
                0 as libc::c_uint,
            )
        } != 0
        {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }
}

/// Adds a rule letting the child read and run beneath `path` (NUL-terminated).
/// System calls only, no allocation: it runs in a child between `fork` and
/// `exec`.
fn allow_read_beneath(ruleset: RawFd, path: &[u8]) -> io::Result<()> {
    // SAFETY: `path` is NUL-terminated.
    let fd = unsafe { libc::open(path.as_ptr().cast(), libc::O_PATH | libc::O_CLOEXEC) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    let rule = PathBeneathAttr {
        allowed_access: READ_FILE | READ_DIR,
        parent_fd: fd,
    };
    // SAFETY: `rule` is a valid `landlock_path_beneath_attr` and both
    // descriptors are open for the call.
    let rc = unsafe {
        libc::syscall(
            SYS_ADD_RULE,
            ruleset,
            RULE_PATH_BENEATH,
            &rule as *const PathBeneathAttr,
            0 as libc::c_uint,
        )
    };
    let result = if rc < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    };
    // SAFETY: `fd` was opened above and is closed once.
    unsafe {
        libc::close(fd);
    }
    result
}
