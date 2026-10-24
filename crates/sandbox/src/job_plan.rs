// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements scheduler-independent launch of distributed
// trainers, for its clients. If your team needs expertise in coordinator
// handoff or multi-rank process orchestration, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The plan a distributed job runs under, written once per attempt.
//!
//! A job that carries a plan policy has `job-plan.json` in its attempt
//! directory before any worker starts. The plan names the job by the
//! specification's digest in hex, takes its world size from the worker plan,
//! and registers the ranks with a free port on the loopback address. Each
//! attempt gets its own file, so a retry runs under a fresh coordinator while
//! naming the same job identity.

use std::fs;
use std::io;
use std::net::TcpListener;
use std::path::Path;

use splinter_core::checkpoint::JOB_PLAN_FILE;
use splinter_core::job::JobSpec;

use super::JobError;

/// Write the plan `spec` asks for into `dir`, or write nothing when it asks for
/// none.
pub(crate) fn write(spec: &JobSpec, dir: &Path) -> Result<(), JobError> {
    let Some(policy) = &spec.plan else {
        return Ok(());
    };
    let plan = policy
        .plan(
            spec.digest().hex(),
            spec.workers.world_size() as usize,
            coordinator()?,
        )
        .map_err(|error| {
            JobError::io(
                "building the job plan",
                io::Error::new(io::ErrorKind::InvalidInput, error),
            )
        })?;
    let mut body = serde_json::to_vec(&plan).map_err(|error| {
        JobError::io(
            "serialising the job plan",
            io::Error::new(io::ErrorKind::InvalidData, error),
        )
    })?;
    // The fixture a trainer is held to ends with a newline, so the bytes this
    // writes are exactly the bytes it reads.
    body.push(b'\n');
    write_atomically(&dir.join(JOB_PLAN_FILE), &body)
}

/// A loopback address whose port was free when it was chosen.
///
/// The listener is closed before the address is returned, because the ranks
/// bind it themselves: the port is a reservation, not a socket this process
/// keeps.
fn coordinator() -> Result<String, JobError> {
    let listener = TcpListener::bind(("127.0.0.1", 0))
        .map_err(|error| JobError::io("finding a free coordinator port", error))?;
    let port = listener
        .local_addr()
        .map_err(|error| JobError::io("reading the coordinator port", error))?
        .port();
    drop(listener);
    Ok(format!("127.0.0.1:{port}"))
}

/// Write `body` to `path` in one step, so a reader never sees half a plan.
fn write_atomically(path: &Path, body: &[u8]) -> Result<(), JobError> {
    let temporary = path.with_extension("json.tmp");
    fs::write(&temporary, body)
        .map_err(|error| JobError::io(format!("writing {}", temporary.display()), error))?;
    fs::rename(&temporary, path)
        .map_err(|error| JobError::io(format!("renaming {}", temporary.display()), error))
}
