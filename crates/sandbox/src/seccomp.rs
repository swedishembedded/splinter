// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements network denial for model-written code, for
// its clients. If your team needs expertise in keeping a language-model
// agent's processes off the network then you can procure our services by
// sending an email to info@swedishembedded.com.

//! A seccomp filter that refuses to create sockets of the internet families.
//!
//! Landlock (ABI 4) denies TCP `bind` and `connect`, but not UDP and not raw
//! or packet sockets. This filter closes them: `socket(2)` with domain
//! `AF_INET`, `AF_INET6` or `AF_PACKET` fails with `EPERM`. Local
//! (unix-domain) sockets and everything else are untouched. The filter is
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

const AF_INET: u32 = 2;
const AF_INET6: u32 = 10;
const AF_PACKET: u32 = 17;

/// The first system-call number of the x32 ABI, which no filter written for
/// the native numbers can classify.
const X32_FLAG: u32 = 0x4000_0000;

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
    /// The filter refusing `socket(2)` of the internet and packet families.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    pub fn deny_inet_sockets() -> io::Result<Self> {
        // Indices of the three outcomes at the end of the program.
        const ALLOW: u8 = 9;
        const DENY: u8 = 10;
        const KILL: u8 = 11;
        // A jump offset counts from the instruction after the jump.
        const fn to(target: u8, at: u8) -> u8 {
            target - at - 1
        }
        let program = vec![
            /* 0 */ stmt(BPF_LD_W_ABS, OFFSET_ARCH),
            /* 1 */ jump(BPF_JEQ_K, AUDIT_ARCH, 0, to(KILL, 1)),
            /* 2 */ stmt(BPF_LD_W_ABS, OFFSET_NR),
            /* 3 */ jump(BPF_JGE_K, X32_FLAG, to(KILL, 3), 0),
            /* 4 */ jump(BPF_JEQ_K, SYS_SOCKET, 0, to(ALLOW, 4)),
            /* 5 */ stmt(BPF_LD_W_ABS, OFFSET_ARG0),
            /* 6 */ jump(BPF_JEQ_K, AF_INET, to(DENY, 6), 0),
            /* 7 */ jump(BPF_JEQ_K, AF_INET6, to(DENY, 7), 0),
            /* 8 */ jump(BPF_JEQ_K, AF_PACKET, to(DENY, 8), 0),
            /* 9 */ stmt(BPF_RET_K, SECCOMP_RET_ALLOW),
            /* 10 */ stmt(BPF_RET_K, SECCOMP_RET_ERRNO | libc::EPERM as u32),
            /* 11 */ stmt(BPF_RET_K, SECCOMP_RET_KILL_PROCESS),
        ];
        Ok(Self { program })
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
