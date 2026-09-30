// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements bounded execution of model-written code for
// its clients. If your team needs expertise in agent sandboxing, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The limits every code call in a sandbox runs under. They are part of an
//! environment's snapshot: changing any of them changes the environment.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::process::ResourceLimits;

/// The default wall-clock time of one call, in milliseconds.
pub const DEFAULT_WALL_TIME_MS: u64 = 10_000;

/// The default CPU time of one call, in seconds.
pub const DEFAULT_CPU_SECONDS: u64 = 5;

/// The default memory of one call, in bytes: its address space in the
/// process sandbox, its container's memory in the container sandbox.
pub const DEFAULT_MEMORY_BYTES: u64 = 1024 * 1024 * 1024;

/// The default size of the largest file one call may write, in bytes.
pub const DEFAULT_FILE_SIZE_BYTES: u64 = 16 * 1024 * 1024;

/// The default process (and thread) limit of one call.
pub const DEFAULT_MAX_PROCESSES: u64 = 1024;

/// The default cap on each output stream of one call, in bytes.
pub const DEFAULT_OUTPUT_CAP_BYTES: u64 = 64 * 1024;

/// What one code call may use. The defaults are the `DEFAULT_*` constants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Limits {
    /// Wall-clock time before the call is killed and reported timed out.
    pub wall_time_ms: u64,
    /// CPU time; a call that spends it is killed and reported timed out.
    pub cpu_seconds: u64,
    /// Memory: the address space of each process in the process sandbox
    /// (`RLIMIT_AS`), the container's memory (swap included) in the
    /// container sandbox.
    pub memory_bytes: u64,
    /// The largest file a call may write (`RLIMIT_FSIZE`); in the container
    /// sandbox also the size of its writable scratch directory.
    pub file_size_bytes: u64,
    /// Processes and threads. The container sandbox counts the container's
    /// (`--pids-limit`). The process sandbox can only set `RLIMIT_NPROC`,
    /// which counts every process and thread of the user id the call runs
    /// as - so there it is a ceiling on that user, not on the call, and a
    /// value below what the user already runs keeps the call from starting
    /// any process or thread at all.
    pub max_processes: u64,
    /// The most bytes kept of each of standard output and standard error.
    pub output_cap_bytes: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            wall_time_ms: DEFAULT_WALL_TIME_MS,
            cpu_seconds: DEFAULT_CPU_SECONDS,
            memory_bytes: DEFAULT_MEMORY_BYTES,
            file_size_bytes: DEFAULT_FILE_SIZE_BYTES,
            max_processes: DEFAULT_MAX_PROCESSES,
            output_cap_bytes: DEFAULT_OUTPUT_CAP_BYTES,
        }
    }
}

impl Limits {
    /// The wall-clock time as a duration.
    #[must_use]
    pub fn wall_time(&self) -> Duration {
        Duration::from_millis(self.wall_time_ms)
    }

    /// The output cap as a buffer size, saturating on a platform whose
    /// `usize` is narrower than the cap.
    #[must_use]
    pub fn output_cap(&self) -> usize {
        usize::try_from(self.output_cap_bytes).unwrap_or(usize::MAX)
    }

    /// The kernel limits a directly run process gets.
    #[must_use]
    pub fn resources(&self) -> ResourceLimits {
        ResourceLimits {
            cpu_seconds: self.cpu_seconds,
            address_space_bytes: self.memory_bytes,
            file_size_bytes: self.file_size_bytes,
            max_processes: self.max_processes,
        }
    }
}
