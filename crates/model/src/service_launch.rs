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
//! every local model from: started on this machine's loopback, or on a node
//! of the allocation as a job step bound to that node's address, and waited
//! for until it is ready.
//!
//! Starting is two halves, [`spawn_service`] and [`Pending::wait`], so several
//! nodes' servers are started from the caller's own thread (a server dies
//! with the thread that started it) and load at once.

use std::ffi::OsString;
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::Receiver;
use std::time::{Duration, Instant};

use splinter_sandbox::step::{die_with_caller, NodeSteps, Placement, StepProcess};
use sven_sdk::CancelToken;

use crate::server::{
    forward_lines, free_port, read_key, recent_output, wait_ready, Failure, DEFAULT_SERVE_SLOTS,
};

/// How long a server on a node may take to publish its connection record
/// once it reported ready: the shared filesystem may show the file late.
const RECORD_WAIT: Duration = Duration::from_secs(60);

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
    /// Where it runs.
    pub host: ServiceHost<'a>,
}

/// Where a service runs.
pub enum ServiceHost<'a> {
    /// This machine, on the loopback, with a key of its own.
    Local,
    /// A node of the allocation, as a job step.
    Node(NodeService<'a>),
}

/// A service started on a node of the allocation.
pub struct NodeService<'a> {
    /// How a step is started there.
    pub steps: &'a NodeSteps,
    /// The node.
    pub node: &'a str,
    /// The node's address the server binds; the port is the one it is
    /// given by the system, read back from its connection record.
    pub address: IpAddr,
    /// The GPUs the step takes on the node.
    pub gpus: u32,
    /// The key every server of the fleet answers to (`--api-key-file`).
    pub key_file: &'a Path,
    /// The certificate chain and private key the server serves TLS with
    /// (`--tls-cert`, `--tls-key`); `None` serves plaintext off the loopback
    /// (`--insecure-plaintext`), which the caller asked for by name.
    pub tls: Option<(&'a Path, &'a Path)>,
    /// The node-local directory brain stages the model store's entries in
    /// (`--stage-models`, from `--models-dir`); `None` reads them from the
    /// shared store.
    pub stage: Option<&'a Path>,
    /// The secret a training job over several servers proves membership with
    /// (`--train-job-secret`), and, for the server that holds rank 0, the
    /// address it binds as the job's coordinator (`--train-job-coordinator`).
    pub train_job: Option<(&'a Path, Option<&'a str>)>,
}

/// The server's process: a child of this one, or the `srun` of its step.
pub enum ServiceProcess {
    /// Started on this machine; killed when dropped.
    Local(Child),
    /// Started as a step; its step ended when dropped.
    Step(StepProcess),
}

impl ServiceProcess {
    fn child(&mut self) -> &mut Child {
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

/// A service started and not yet ready.
pub struct Pending {
    process: ServiceProcess,
    lines: Receiver<String>,
    work: PathBuf,
    ready: PathBuf,
    /// The key file the server writes, or the one it was given.
    key: KeySource,
    /// Where it listens, `scheme://host:port`, when known before it is
    /// ready (the loopback port this process chose).
    origin: Option<String>,
    slots: u32,
    startup: Duration,
}

enum KeySource {
    /// `--api-keys-out`: the server's own key, as JSON.
    Written(PathBuf),
    /// `--api-key-file`: the key the caller gave every server.
    Given(PathBuf),
}

/// The `brain` arguments of `launch`, listening on `address`, publishing
/// its key and readiness as `key` and `ready` say.
fn service_args(
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
    let (mut command, key, origin) = match &launch.host {
        ServiceHost::Local => {
            let port =
                free_port().map_err(|e| Failure::Unmeasured(format!("no free port: {e}")))?;
            let address = format!("127.0.0.1:{port}");
            let key = KeySource::Written(launch.work.join("keys.json"));
            let mut command = Command::new(launch.binary);
            command.args(service_args(launch, &address, &key, &ready));
            // SIGINT, so a training job saves its state as on a cancel.
            die_with_caller(&mut command, libc::SIGINT);
            (command, key, Some(format!("http://{address}")))
        }
        ServiceHost::Node(node) => {
            let key = KeySource::Given(node.key_file.to_path_buf());
            // Port 0: the node's system picks a free one, which the server's
            // connection record names once it is bound.
            let mut args = service_args(launch, &format!("{}:0", node.address), &key, &ready);
            if let Some(dir) = node.stage {
                // Global flags, before the subcommand.
                let staging = crate::distributed::staging_args(dir, launch.models);
                args.splice(0..0, staging.into_iter().map(OsString::from));
            }
            if let Some((secret, coordinator)) = node.train_job {
                args.push("--train-job-secret".into());
                args.push(secret.into());
                if let Some(address) = coordinator {
                    args.push("--train-job-coordinator".into());
                    args.push(address.into());
                }
            }
            match node.tls {
                Some((chain, private_key)) => {
                    args.push("--tls-cert".into());
                    args.push(chain.into());
                    args.push("--tls-key".into());
                    args.push(private_key.into());
                }
                // Off the loopback brain serves plaintext only when told.
                None => args.push("--insecure-plaintext".into()),
            }
            let program: Vec<String> = std::iter::once(launch.binary.as_os_str())
                .chain(args.iter().map(OsString::as_os_str))
                .map(|word| word.to_string_lossy().into_owned())
                .collect();
            let script = launch.work.join("serve.sh");
            std::fs::write(&script, node.steps.script(launch.work, &program)).map_err(|e| {
                Failure::Unmeasured(format!("cannot write {}: {e}", script.display()))
            })?;
            let mut command = node.steps.command(
                &Placement {
                    nodes: vec![node.node.to_string()],
                    tasks_per_node: 1,
                    gpus_per_node: node.gpus,
                },
                &script,
            );
            // SIGTERM: `srun` forwards it to the step; a SIGINT only makes it
            // print the step's status.
            die_with_caller(&mut command, libc::SIGTERM);
            (command, key, None)
        }
    };
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
    let process = match &launch.host {
        ServiceHost::Local => ServiceProcess::Local(child),
        ServiceHost::Node(node) => ServiceProcess::Step(StepProcess {
            child,
            dir: launch.work.to_path_buf(),
            scancel: node.steps.scancel.clone(),
        }),
    };
    Ok(Pending {
        process,
        lines,
        work: launch.work.to_path_buf(),
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
        let key = match &self.key {
            KeySource::Written(path) => read_key(path),
            KeySource::Given(path) => {
                crate::endpoint::BrainEndpoints::read_key(path).map_err(|e| e.to_string())
            }
        }
        .map_err(|why| with_output(why, &mut kept))?;
        let origin = match self.origin.take() {
            Some(origin) => origin,
            None => bound_origin(&self.work, cancel).map_err(|why| with_output(why, &mut kept))?,
        };
        Ok(Service {
            process: self.process,
            lines: self.lines,
            kept,
            origin,
            key,
            slots: self.slots,
        })
    }
}

/// The `scheme://host:port` the OpenAI surface of the server whose runtime
/// directory is `work` bound, from its connection record (`serve-<pid>.json`).
fn bound_origin(work: &Path, cancel: &CancelToken) -> Result<String, String> {
    let deadline = Instant::now() + RECORD_WAIT;
    loop {
        if let Some(origin) = read_record(work)? {
            return Ok(origin);
        }
        if cancel.is_cancelled() {
            return Err("cancelled while reading the server's connection record".into());
        }
        if Instant::now() >= deadline {
            return Err(format!(
                "the server reported ready but published no connection record in {} within {}s",
                work.display(),
                RECORD_WAIT.as_secs()
            ));
        }
        std::thread::sleep(Duration::from_millis(200));
    }
}

/// The OpenAI surface's `scheme://host:port` from a connection record in
/// `work`, if one is there yet.
fn read_record(work: &Path) -> Result<Option<String>, String> {
    let entries =
        std::fs::read_dir(work).map_err(|e| format!("cannot read {}: {e}", work.display()))?;
    for entry in entries.filter_map(Result::ok) {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !(name.starts_with("serve-") && name.ends_with(".json")) {
            continue;
        }
        // A record being written reads as no record yet.
        let Ok(text) = std::fs::read_to_string(entry.path()) else {
            continue;
        };
        let Ok(record) = serde_json::from_str::<serde_json::Value>(&text) else {
            continue;
        };
        let url = record["surfaces"].as_array().and_then(|surfaces| {
            surfaces
                .iter()
                .find(|s| s["provider"] == "openai")
                .and_then(|s| s["url"].as_str())
        });
        let Some(url) = url else {
            return Err(format!("{name} names no openai surface: {text}"));
        };
        let origin = url.trim_end_matches('/');
        let usable = ["http://", "https://"].iter().any(|scheme| {
            origin
                .strip_prefix(scheme)
                .is_some_and(|rest| !rest.is_empty() && !rest.ends_with(":0"))
        });
        if !usable {
            return Err(format!("{name} names an unusable openai url {url:?}"));
        }
        return Ok(Some(origin.to_string()));
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_bound_address_is_read_from_the_connection_record() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(read_record(dir.path()).unwrap(), None);
        std::fs::write(
            dir.path().join("serve-77.json"),
            r#"{"pid":77,"surfaces":[{"provider":"openai","url":"http://10.0.0.5:41234"}]}"#,
        )
        .unwrap();
        assert_eq!(
            read_record(dir.path()).unwrap().as_deref(),
            Some("http://10.0.0.5:41234")
        );
        std::fs::write(
            dir.path().join("serve-77.json"),
            r#"{"pid":77,"surfaces":[{"provider":"openai","url":"https://10.0.0.5:41234/"}]}"#,
        )
        .unwrap();
        assert_eq!(
            read_record(dir.path()).unwrap().as_deref(),
            Some("https://10.0.0.5:41234")
        );
        std::fs::write(
            dir.path().join("serve-77.json"),
            r#"{"pid":77,"surfaces":[{"provider":"anthropic","url":"http://10.0.0.5:1"}]}"#,
        )
        .unwrap();
        assert!(read_record(dir.path()).is_err(), "no openai surface");
    }
}
