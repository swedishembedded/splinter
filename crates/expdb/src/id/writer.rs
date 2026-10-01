// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Writer ids: a process derives its own, so ingest never asks anyone for a
//! sequence number.

use std::fmt;

/// The id of one writer incarnation. Record sequence numbers are unique
/// within it, so `(writer, sequence)` is unique across the database.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WriterId(u64);

impl WriterId {
    /// A writer id from its raw value.
    pub fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// The raw value.
    pub fn raw(self) -> u64 {
        self.0
    }
}

impl fmt::Display for WriterId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:016x}", self.0)
    }
}

impl fmt::Debug for WriterId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WriterId({self})")
    }
}

/// Who a writer is: the experiment, the scheduler job, the host and the rank
/// within the job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriterIdentity {
    experiment: String,
    job: String,
    host: String,
    rank: u32,
}

impl WriterIdentity {
    /// An identity from its four parts.
    pub fn new(experiment: &str, job: &str, host: &str, rank: u32) -> Self {
        Self {
            experiment: experiment.into(),
            job: job.into(),
            host: host.into(),
            rank,
        }
    }

    /// The job part, which names the writer's ref.
    pub fn job(&self) -> &str {
        &self.job
    }

    /// The writer id for one incarnation of this identity. The same
    /// identity restarted must use a different `incarnation`, or it would
    /// reuse sequence numbers it already spent.
    pub fn writer_id(&self, incarnation: u64) -> WriterId {
        let mut hasher = blake3::Hasher::new();
        for part in [&self.experiment, &self.job, &self.host] {
            hasher.update(&(part.len() as u64).to_le_bytes());
            hasher.update(part.as_bytes());
        }
        hasher.update(&self.rank.to_le_bytes());
        hasher.update(&incarnation.to_le_bytes());
        let mut head = [0u8; 8];
        head.copy_from_slice(&hasher.finalize().as_bytes()[..8]);
        WriterId(u64::from_le_bytes(head))
    }
}
