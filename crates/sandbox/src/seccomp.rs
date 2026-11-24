// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements network denial for model-written code, for
// its clients. If your team needs expertise in keeping a language-model
// agent's processes off the network then you can procure our services by
// sending an email to info@swedishembedded.com.

//! A seccomp filter that refuses to create the sockets a command could
//! leave its confinement through.
//!
//! Landlock (ABI 4) denies TCP `bind` and `connect`, but not UDP and not raw
//! or packet sockets, and no Landlock right governs `connect` to a Unix
//! socket that exists (searching a path needs none): a confined command
//! reached any daemon's control socket it could name (a container runtime's,
//! the session bus, an SSH agent). This filter closes them: `socket(2)` with
//! domain `AF_INET`, `AF_INET6`, `AF_PACKET`, `AF_UNIX` or `AF_NETLINK`
//! (which, run as root, reconfigures the host's network) fails with `EPERM`.
//! `socketpair(2)`, the local pipes a program makes for itself, is left
//! alone, and so is everything else. The filter is
//! built in the parent (it allocates) and installed in the child with one
//! system call.
//!
//! A call made under another system-call ABI than the filter was written for
//! (32-bit on a 64-bit kernel, x32) would bypass the number check, so it
//! kills the process. On an architecture the filter was not written for,
//! building it fails, and the confinement is refused rather than weakened.

use std::io;

const BPF_LD_W_ABS: u16 = 0x20;
const BPF_JEQ_K: u16 = 0x15;
const BPF_JGE_K: u16 = 0x35;
const BPF_RET_K: u16 = 0x06;

const SECCOMP_RET_ALLOW: u32 = 0x7fff_0000;
const SECCOMP_RET_ERRNO: u32 = 0x0005_0000;
const SECCOMP_RET_KILL_PROCESS: u32 = 0x8000_0000;
const SECCOMP_SET_MODE_FILTER: libc::c_long = 1;

/// `struct seccomp_data` offsets.
const OFFSET_NR: u32 = 0;
const OFFSET_ARCH: u32 = 4;
const OFFSET_ARG0: u32 = 16;
const OFFSET_ARG1: u32 = 24;
const BPF_ALU_AND_K: u16 = 0x54;
const SOCK_STREAM: u32 = 1;
/// The bits of `socket(2)`'s type that name the type (the rest are
/// `SOCK_NONBLOCK`/`SOCK_CLOEXEC`).
const SOCK_TYPE_MASK: u32 = 0xf;

const AF_UNIX: u32 = 1;
const AF_INET: u32 = 2;
const AF_INET6: u32 = 10;
const AF_NETLINK: u32 = 16;
const AF_PACKET: u32 = 17;

/// The first system-call number of the x32 ABI, which no filter written for
/// the native numbers can classify.
const X32_FLAG: u32 = 0x4000_0000;
/// The number a tracer (proot) sets to skip a call: -1, which the kernel
/// answers with `ENOSYS`. It has the x32 bit set but is no x32 call, so it
/// is allowed before that check.
const SKIPPED_CALL: u32 = u32::MAX;

#[cfg(target_arch = "x86_64")]
const AUDIT_ARCH: u32 = 0xc000_003e;
#[cfg(target_arch = "x86_64")]
const SYS_SOCKET: u32 = 41;
#[cfg(target_arch = "aarch64")]
const AUDIT_ARCH: u32 = 0xc000_00b7;
#[cfg(target_arch = "aarch64")]
const SYS_SOCKET: u32 = 198;

#[repr(C)]
#[derive(Clone, Copy)]
struct SockFilter {
    code: u16,
    jt: u8,
    jf: u8,
    k: u32,
}

#[repr(C)]
struct SockFprog {
    len: u16,
    filter: *const SockFilter,
}

/// A compiled filter, ready to install in a child.
pub struct Filter {
    program: Vec<SockFilter>,
}

// SAFETY: the program is plain data, never mutated after it is built.
unsafe impl Send for Filter {}
// SAFETY: as above; installing reads it only.
unsafe impl Sync for Filter {}

const fn stmt(code: u16, k: u32) -> SockFilter {
    SockFilter {
        code,
        jt: 0,
        jf: 0,
        k,
    }
}

const fn jump(code: u16, k: u32, jt: u8, jf: u8) -> SockFilter {
    SockFilter { code, jt, jf, k }
}

impl Filter {
    /// The filter refusing `socket(2)` of the internet, packet, Unix and
    /// netlink families.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    pub fn deny_inet_sockets() -> io::Result<Self> {
        // Indices of the three outcomes at the end of the program.
        const ALLOW: u8 = 12;
        const DENY: u8 = 13;
        const KILL: u8 = 14;
        // A jump offset counts from the instruction after the jump.
        const fn to(target: u8, at: u8) -> u8 {
            target - at - 1
        }
        let program = vec![
            /* 0 */ stmt(BPF_LD_W_ABS, OFFSET_ARCH),
            /* 1 */ jump(BPF_JEQ_K, AUDIT_ARCH, 0, to(KILL, 1)),
            /* 2 */ stmt(BPF_LD_W_ABS, OFFSET_NR),
            /* 3 */ jump(BPF_JEQ_K, SKIPPED_CALL, to(ALLOW, 3), 0),
            /* 4 */ jump(BPF_JGE_K, X32_FLAG, to(KILL, 4), 0),
            /* 5 */ jump(BPF_JEQ_K, SYS_SOCKET, 0, to(ALLOW, 5)),
            /* 6 */ stmt(BPF_LD_W_ABS, OFFSET_ARG0),
            /* 7 */ jump(BPF_JEQ_K, AF_INET, to(DENY, 7), 0),
            /* 8 */ jump(BPF_JEQ_K, AF_INET6, to(DENY, 8), 0),
            /* 9 */ jump(BPF_JEQ_K, AF_PACKET, to(DENY, 9), 0),
            /* 10 */ jump(BPF_JEQ_K, AF_UNIX, to(DENY, 10), 0),
            /* 11 */ jump(BPF_JEQ_K, AF_NETLINK, to(DENY, 11), 0),
            /* 12 */ stmt(BPF_RET_K, SECCOMP_RET_ALLOW),
            /* 13 */ stmt(BPF_RET_K, SECCOMP_RET_ERRNO | libc::EPERM as u32),
            /* 14 */ stmt(BPF_RET_K, SECCOMP_RET_KILL_PROCESS),
        ];
        Ok(Self { program })
    }

    /// The filter refusing `socket(2)` of the IPv6, packet, Unix and netlink
    /// families and every IPv4 socket that is not a stream: what is left of
    /// the network is IPv4 TCP, which Landlock then holds to its allowed
    /// ports (a granted port is no grant of the host's daemons).
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    pub fn deny_inet_sockets_but_tcp4() -> io::Result<Self> {
        const ALLOW: u8 = 15;
        const DENY: u8 = 16;
        const KILL: u8 = 17;
        const fn to(target: u8, at: u8) -> u8 {
            target - at - 1
        }
        let program = vec![
            /* 0 */ stmt(BPF_LD_W_ABS, OFFSET_ARCH),
            /* 1 */ jump(BPF_JEQ_K, AUDIT_ARCH, 0, to(KILL, 1)),
            /* 2 */ stmt(BPF_LD_W_ABS, OFFSET_NR),
            /* 3 */ jump(BPF_JEQ_K, SKIPPED_CALL, to(ALLOW, 3), 0),
            /* 4 */ jump(BPF_JGE_K, X32_FLAG, to(KILL, 4), 0),
            /* 5 */ jump(BPF_JEQ_K, SYS_SOCKET, 0, to(ALLOW, 5)),
            /* 6 */ stmt(BPF_LD_W_ABS, OFFSET_ARG0),
            /* 7 */ jump(BPF_JEQ_K, AF_INET6, to(DENY, 7), 0),
            /* 8 */ jump(BPF_JEQ_K, AF_PACKET, to(DENY, 8), 0),
            /* 9 */ jump(BPF_JEQ_K, AF_UNIX, to(DENY, 9), 0),
            /* 10 */ jump(BPF_JEQ_K, AF_NETLINK, to(DENY, 10), 0),
            /* 11 */ jump(BPF_JEQ_K, AF_INET, 0, to(ALLOW, 11)),
            /* 12 */ stmt(BPF_LD_W_ABS, OFFSET_ARG1),
            /* 13 */ stmt(BPF_ALU_AND_K, SOCK_TYPE_MASK),
            /* 14 */ jump(BPF_JEQ_K, SOCK_STREAM, to(ALLOW, 14), to(DENY, 14)),
            /* 15 */ stmt(BPF_RET_K, SECCOMP_RET_ALLOW),
            /* 16 */ stmt(BPF_RET_K, SECCOMP_RET_ERRNO | libc::EPERM as u32),
            /* 17 */ stmt(BPF_RET_K, SECCOMP_RET_KILL_PROCESS),
        ];
        Ok(Self { program })
    }

    /// Where the filter was not written for the architecture.
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub fn deny_inet_sockets_but_tcp4() -> io::Result<Self> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "no socket filter exists for this architecture",
        ))
    }

    /// Where the filter was not written for the architecture.
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub fn deny_inet_sockets() -> io::Result<Self> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "no socket filter exists for this architecture",
        ))
    }

    /// Installs the filter on the calling process and its future children.
    /// Makes one system call and allocates nothing, so it may run between
    /// `fork` and `exec`. `no_new_privs` must already be set.
    pub fn install(&self) -> io::Result<()> {
        let program = SockFprog {
            len: self.program.len() as u16,
            filter: self.program.as_ptr(),
        };
        // SAFETY: `program` points at a valid filter that outlives the call.
        let rc = unsafe {
            libc::syscall(
                libc::SYS_seccomp,
                SECCOMP_SET_MODE_FILTER,
                0 as libc::c_ulong,
                &program as *const SockFprog,
            )
        };
        if rc != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }
}
