// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems that start the model
// servers they need, on this machine or on every node of a cluster
// allocation, and wait for them to be ready, for its clients. If your team
// needs expertise in operating model servers as part of a training
// pipeline, you can procure our services by sending an email to
// info@swedishembedded.com.

//! A `brain serve` over a whole model store, the service a run requests
//! every local model from: started on this machine's loopback and waited
//! for until it is ready. The servers of a fleet, one on each node of the
//! allocation, are started by `fleet_launch` with the same arguments.
//!
//! Starting is two halves, [`spawn_service`] and [`Pending::wait`].

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::Receiver;
use std::time::Duration;

use splinter_sandbox::step::{die_with_caller, StepProcess};
use sven_sdk::CancelToken;

use crate::server::{
    forward_lines, free_port, read_key, recent_output, wait_ready, Failure, DEFAULT_SERVE_SLOTS,
};

/// What it takes to start a server that serves a whole model store, as the
/// service every `local:` model of a run is requested from.
pub struct ServiceLaunch<'a> {
    /// The `brain` binary.
    pub binary: &'a Path,
    /// A scratch directory for the keys, the ready marker and the runtime
    /// directory; on a node, one every node and this process see.
    pub work: &'a Path,
    /// Brain's model store: every base in it is served under the id the
    /// store gives it.
    pub models: &'a Path,
    /// The directory adapters are offered in (`--serve-adapters`): one
    /// written there is served as a model of its own, with no restart.
    /// `None` serves none, for a store with no base an adapter can sit on
    /// (brain refuses `--serve-adapters` then).
    pub adapters: Option<&'a Path>,
    /// The base those adapters sit on (`--checkpoint`); brain's choice when
    /// `None`.
    pub adapter_base: Option<&'a Path>,
    /// The directory datasets and trained adapters move through
    /// (`--artifact-store`), which also turns the training workload on.
    pub artifacts: &'a Path,
    /// The compute it may schedule, in brain's `--device` grammar; brain's
    /// own choice when `None`.
    pub device: Option<&'a str>,
    /// How the hardware is driven (`--backend`); brain's choice when `None`.
    pub backend: Option<&'a str>,
    /// Sequences it decodes together (`--max-batch`);
    /// [`DEFAULT_SERVE_SLOTS`] when `None`.
    pub max_batch: Option<u32>,
    /// How long the server may take to bind.
    pub startup: Duration,
    /// A file every line the server writes is appended to; only the last
    /// lines are kept otherwise.
    pub log: Option<&'a Path>,
}

/// The server's process: a child of this one, or the `srun` of its step.
pub enum ServiceProcess {
    /// Started on this machine; killed when dropped.
    Local(Child),
    /// Started as a step; its step ended when dropped.
    Step(StepProcess),
}

impl ServiceProcess {
    pub(crate) fn child(&mut self) -> &mut Child {
        match self {
            Self::Local(child) => child,
            Self::Step(step) => &mut step.child,
        }
    }
}

impl Drop for ServiceProcess {
    fn drop(&mut self) {
        if let Self::Local(child) = self {
            // Stopping a server that already exited is not an error worth
            // reporting over the caller's own result.
            let _ = child.kill();
            let _ = child.wait();
        }
        // A step stops itself when its `StepProcess` is dropped.
    }
}

/// A running service: the server, and how to reach it.
pub struct Service {
    /// The process, stopped when the service is dropped.
    pub process: ServiceProcess,
    /// What it writes, line by line.
    pub lines: Receiver<String>,
    /// The last lines it wrote.
    pub kept: Vec<String>,
    /// Where it listens, `scheme://host:port`.
    pub origin: String,
    /// The OpenAI surface's key.
    pub key: String,
    /// The sequences it was started to decode together.
    pub slots: u32,
}

impl Service {
    /// The OpenAI base URL clients request the service at.
    #[must_use]
    pub fn url(&self) -> String {
        format!("{}/v1", self.origin)
    }
}

/// A service started on this machine and not yet ready.
pub struct Pending {
    process: ServiceProcess,
    lines: Receiver<String>,
    ready: PathBuf,
    /// The key file the server writes.
    key: PathBuf,
    /// Where it listens, `http://127.0.0.1:port`: the port this process chose.
    origin: String,
    slots: u32,
    startup: Duration,
}

pub(crate) enum KeySource {
    /// `--api-keys-out`: the server's own key, as JSON.
    Written(PathBuf),
    /// `--api-key-file`: the key the caller gave every server.
    Given(PathBuf),
}

/// The `brain` arguments of `launch`, listening on `address`, publishing
/// its key and readiness as `key` and `ready` say.
pub(crate) fn service_args(
    launch: &ServiceLaunch<'_>,
    address: &str,
    key: &KeySource,
    ready: &Path,
) -> Vec<OsString> {
    let mut args: Vec<OsString> = Vec::new();
    let mut push = |words: &[&str]| args.extend(words.iter().map(OsString::from));
    if let Some(device) = launch.device {
        push(&["--device", device]);
    }
    if let Some(backend) = launch.backend {
        push(&["--backend", backend]);
    }
    if let Some(gib) = crate::device::vram_budget_gib() {
        push(&["--vram-budget", &gib.to_string()]);
    }
    let slots = launch.max_batch.unwrap_or(DEFAULT_SERVE_SLOTS).max(1);
    push(&[
        "serve",
        "--openai",
        address,
        "--max-batch",
        &slots.to_string(),
    ]);
    args.push("--models-dir".into());
    args.push(launch.models.into());
    if let Some(adapters) = launch.adapters {
        args.push("--serve-adapters".into());
        args.push(adapters.into());
        args.extend(["--adapter-targets", "all"].map(OsString::from));
        if let Some(base) = launch.adapter_base {
            args.push("--checkpoint".into());
            args.push(base.into());
        }
    }
    args.extend(["--training", "--artifact-store"].map(OsString::from));
    args.push(launch.artifacts.into());
    args.push("--runtime-dir".into());
    args.push(launch.work.into());
    match key {
        KeySource::Written(path) => {
            args.push("--api-keys-out".into());
            args.push(path.into());
        }
        KeySource::Given(path) => {
            args.push("--api-key-file".into());
            args.push(path.into());
        }
    }
    args.push("--ready-file".into());
    args.push(ready.into());
    args
}

/// Starts the service `launch` describes; [`Pending::wait`] waits until it
/// is ready.
pub fn spawn_service(launch: &ServiceLaunch<'_>) -> Result<Pending, Failure> {
    let ready = launch.work.join("ready");
    // A marker left by an earlier server in this directory describes a
    // process that is gone.
    match std::fs::remove_file(&ready) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => {
            return Err(Failure::Unmeasured(format!(
                "cannot remove {}: {e}",
                ready.display()
            )))
        }
    }
    let slots = launch.max_batch.unwrap_or(DEFAULT_SERVE_SLOTS).max(1);
    let port = free_port().map_err(|e| Failure::Unmeasured(format!("no free port: {e}")))?;
    let address = format!("127.0.0.1:{port}");
    let key = launch.work.join("keys.json");
    let mut command = Command::new(launch.binary);
    command.args(service_args(
        launch,
        &address,
        &KeySource::Written(key.clone()),
        &ready,
    ));
    // SIGINT, so a training job saves its state as on a cancel.
    die_with_caller(&mut command, libc::SIGINT);
    let origin = format!("http://{address}");
    let program = command.get_program().to_string_lossy().into_owned();
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| Failure::Unmeasured(format!("{program} could not start: {e}")))?;
    let log = match launch.log {
        Some(path) => Some(
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .map_err(|e| Failure::Unmeasured(format!("{}: {e}", path.display())))?,
        ),
        None => None,
    };
    let lines = forward_lines(&mut child, log);
    let process = ServiceProcess::Local(child);
    Ok(Pending {
        process,
        lines,
        ready,
        key,
        origin,
        slots,
        startup: launch.startup,
    })
}

impl Pending {
    /// Waits until the server is ready, and reads how to reach it.
    pub fn wait(mut self, cancel: &CancelToken) -> Result<Service, Failure> {
        let mut kept: Vec<String> = Vec::new();
        wait_ready(
            self.process.child(),
            &self.lines,
            &mut kept,
            &self.ready,
            self.startup,
            cancel,
            false,
        )?;
        let with_output = |why: String, kept: &mut Vec<String>| {
            Failure::Unmeasured(format!(
                "{why}; it wrote: {}",
                recent_output(&self.lines, kept)
            ))
        };
        let key = read_key(&self.key).map_err(|why| with_output(why, &mut kept))?;
        Ok(Service {
            process: self.process,
            lines: self.lines,
            kept,
            origin: self.origin,
            key,
            slots: self.slots,
        })
    }
}
