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
//! and registers the ranks with a port on the address of the machine rank 0
//! runs on (the loopback address for a job on this machine). Beside it is a
//! fresh owner-only secret every rank proves its membership with. Each
//! attempt gets its own files, so a retry runs under a fresh coordinator and
//! secret while naming the same job identity.

use std::fs;
use std::io::{self, Read};
use std::net::{IpAddr, Ipv4Addr, TcpListener};
use std::path::Path;

use splinter_core::checkpoint::JOB_PLAN_FILE;
use splinter_core::job::{JobSpec, JOB_SECRET_FILE};

use super::JobError;

/// The loopback address a job on this machine registers its ranks on.
pub(crate) const LOOPBACK: IpAddr = IpAddr::V4(Ipv4Addr::LOCALHOST);

/// Write the plan `spec` asks for into `dir`, its coordinator on `host` (the
/// address of the machine rank 0 runs on), and the job's secret beside it;
/// write nothing when it asks for none.
pub(crate) fn write(spec: &JobSpec, dir: &Path, host: IpAddr) -> Result<(), JobError> {
    let Some(policy) = &spec.plan else {
        return Ok(());
    };
    let secret = dir.join(JOB_SECRET_FILE);
    crate::step::write_secret(&secret)
        .map_err(|error| JobError::io(format!("writing {}", secret.display()), error))?;
    let plan = policy
        .plan(
            spec.digest().hex(),
            spec.workers.world_size() as usize,
            coordinator(host)?,
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

/// An address on `host` whose port was free when it was chosen.
///
/// On this machine the port is one the system reports free; the listener is
/// closed before the address is returned, because the ranks bind it
/// themselves: the port is a reservation, not a socket this process keeps. On
/// another machine nothing here can ask, so the port is drawn at random from
/// the range above the usual ephemeral ports' start, where a clash is rare
/// and is reported by the rank that cannot bind it.
fn coordinator(host: IpAddr) -> Result<String, JobError> {
    let port = match TcpListener::bind((host, 0)) {
        Ok(listener) => listener
            .local_addr()
            .map_err(|error| JobError::io("reading the coordinator port", error))?
            .port(),
        Err(_) => {
            let mut bytes = [0u8; 2];
            fs::File::open("/dev/urandom")
                .and_then(|mut random| random.read_exact(&mut bytes))
                .map_err(|error| JobError::io("drawing a coordinator port", error))?;
            20_000 + u16::from_le_bytes(bytes) % 30_000
        }
    };
    Ok(std::net::SocketAddr::new(host, port).to_string())
}

/// Write `body` to `path` in one step, so a reader never sees half a plan.
fn write_atomically(path: &Path, body: &[u8]) -> Result<(), JobError> {
    let temporary = path.with_extension("json.tmp");
    fs::write(&temporary, body)
        .map_err(|error| JobError::io(format!("writing {}", temporary.display()), error))?;
    fs::rename(&temporary, path)
        .map_err(|error| JobError::io(format!("renaming {}", temporary.display()), error))
}
