// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems that start a model server
// on every node of a cluster allocation as one job step, so that the
// training ranks they host reach each other over the cluster's fabric, for
// its clients. If your team needs expertise in multi-node model serving and
// training on HPC allocations, you can procure our services by sending an
// email to info@swedishembedded.com.

//! The servers of a fleet: one `brain serve` on each node of the
//! allocation, every one a task of ONE job step ([`spawn_fleet`]), bound to
//! its node's address on a port its system picks, and waited for until each
//! is ready ([`PendingFleet::wait`]).
//!
//! One step, not one per node: on an HPE Slingshot fabric the network (a
//! VNI) is granted per step, so a server started in a step of its own on
//! each node cannot reach the others' over the fabric, and a training job
//! over several servers falls back to sockets or fails. Each task finds its
//! node's arguments by the scheduler's node name
//! ([`NodeSteps::per_node_script`]) and records its exit, so a server that
//! ends before it is ready is named while the others still run.

use std::ffi::OsString;
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

use splinter_sandbox::step::{
    die_with_caller, NodeSteps, NodeTask, Placement, StepProcess, TASK_EXIT_FILE,
};
use sven_sdk::CancelToken;

use crate::server::{forward_lines, recent_output, remember, Failure, DEFAULT_SERVE_SLOTS};
use crate::service_launch::{service_args, KeySource, ServiceLaunch, ServiceProcess};

/// The script of the fleet's step, in the fleet's directory.
const STEP_SCRIPT: &str = "serve-step.sh";

/// One node's server of a fleet.
pub struct NodeServer<'a> {
    /// What the server serves; its `work` is the node's own directory (its
    /// runtime directory, ready marker and connection record), one every
    /// node and this process see.
    pub launch: ServiceLaunch<'a>,
    /// The node, by the scheduler's name for it.
    pub node: &'a str,
    /// The node's address the server binds; the port is the one it is
    /// given by the system, read back from its connection record.
    pub address: IpAddr,
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

/// Every node's server, as one step.
pub struct FleetLaunch<'a> {
    /// How a step is started on the allocation's nodes.
    pub steps: &'a NodeSteps,
    /// The fleet's directory, which every node sees: the step's script and
    /// its recorded id.
    pub work: &'a Path,
    /// The GPUs the step takes on each node.
    pub gpus_per_node: u32,
    /// A file every line the step writes is appended to.
    pub log: Option<&'a Path>,
    /// How long the servers may take to be ready, all of them.
    pub startup: Duration,
    /// The servers, in the order the fleet lists them; the first holds rank
    /// 0 of a training job over them.
    pub servers: Vec<NodeServer<'a>>,
}

/// The `brain` command of one node's server.
fn node_command(server: &NodeServer<'_>, ready: &Path) -> Vec<String> {
    let key = KeySource::Given(server.key_file.to_path_buf());
    // Port 0: the node's system picks a free one, which the server's
    // connection record names once it is bound.
    let mut args = service_args(
        &server.launch,
        &format!("{}:0", server.address),
        &key,
        ready,
    );
    if let Some(dir) = server.stage {
        // Global flags, before the subcommand.
        let staging = crate::distributed::staging_args(dir, server.launch.models);
        args.splice(0..0, staging.into_iter().map(OsString::from));
    }
    if let Some((secret, coordinator)) = server.train_job {
        args.push("--train-job-secret".into());
        args.push(secret.into());
        if let Some(address) = coordinator {
            args.push("--train-job-coordinator".into());
            args.push(address.into());
        }
    }
    match server.tls {
        Some((chain, private_key)) => {
            args.push("--tls-cert".into());
            args.push(chain.into());
            args.push("--tls-key".into());
            args.push(private_key.into());
        }
        // Off the loopback brain serves plaintext only when told.
        None => args.push("--insecure-plaintext".into()),
    }
    std::iter::once(server.launch.binary.as_os_str())
        .chain(args.iter().map(OsString::as_os_str))
        .map(|word| word.to_string_lossy().into_owned())
        .collect()
}

/// A node's server started and not yet ready.
struct PendingNode {
    node: String,
    work: PathBuf,
    ready: PathBuf,
    key_file: PathBuf,
    slots: u32,
}

/// The fleet's step, started, its servers not yet all ready.
pub struct PendingFleet {
    process: ServiceProcess,
    lines: Receiver<String>,
    nodes: Vec<PendingNode>,
    startup: Duration,
}

/// One node's server, ready.
pub struct NodeEndpoint {
    /// Where it listens, `scheme://host:port`.
    pub origin: String,
    /// The OpenAI surface's key.
    pub key: String,
    /// The sequences it was started to decode together.
    pub slots: u32,
}

impl NodeEndpoint {
    /// The OpenAI base URL clients request the server at.
    #[must_use]
    pub fn url(&self) -> String {
        format!("{}/v1", self.origin)
    }
}

/// The fleet, ready: its step and every node's server.
pub struct Fleet {
    /// The step, ended when dropped.
    pub process: ServiceProcess,
    /// What the step writes, line by line (every node's server).
    pub lines: Receiver<String>,
    /// The last lines it wrote.
    pub kept: Vec<String>,
    /// The servers, in the order they were launched.
    pub servers: Vec<NodeEndpoint>,
}

/// Starts every node's server of `launch` as one step; [`PendingFleet::wait`]
/// waits until each is ready.
///
/// # Errors
/// A marker or script cannot be written, or `srun` cannot be started.
pub fn spawn_fleet(launch: &FleetLaunch<'_>) -> Result<PendingFleet, Failure> {
    let unmeasured = |what: String| Failure::Unmeasured(what);
    let mut tasks = Vec::with_capacity(launch.servers.len());
    let mut nodes = Vec::with_capacity(launch.servers.len());
    for server in &launch.servers {
        let work = server.launch.work;
        let ready = work.join("ready");
        // Markers left by an earlier server in this directory describe a
        // process that is gone.
        for stale in [ready.clone(), work.join(TASK_EXIT_FILE)] {
            match std::fs::remove_file(&stale) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => {
                    return Err(unmeasured(format!(
                        "cannot remove {}: {e}",
                        stale.display()
                    )))
                }
            }
        }
        tasks.push(NodeTask {
            node: server.node.to_string(),
            dir: work.to_path_buf(),
            command: node_command(server, &ready),
        });
        nodes.push(PendingNode {
            node: server.node.to_string(),
            work: work.to_path_buf(),
            ready,
            key_file: server.key_file.to_path_buf(),
            slots: server
                .launch
                .max_batch
                .unwrap_or(DEFAULT_SERVE_SLOTS)
                .max(1),
        });
    }
    let script = launch.work.join(STEP_SCRIPT);
    std::fs::write(&script, launch.steps.per_node_script(launch.work, &tasks))
        .map_err(|e| unmeasured(format!("cannot write {}: {e}", script.display())))?;
    let mut command = launch.steps.command(
        &Placement {
            nodes: nodes.iter().map(|n| n.node.clone()).collect(),
            tasks_per_node: 1,
            gpus_per_node: launch.gpus_per_node,
        },
        &script,
    );
    // SIGTERM: `srun` forwards it to the step's tasks; a SIGINT only makes it
    // print the step's status.
    die_with_caller(&mut command, libc::SIGTERM);
    let program = command.get_program().to_string_lossy().into_owned();
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = crate::server::spawn_when_written(&mut command)
        .map_err(|e| unmeasured(format!("{program} could not start: {e}")))?;
    let log = match launch.log {
        Some(path) => Some(
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .map_err(|e| unmeasured(format!("{}: {e}", path.display())))?,
        ),
        None => None,
    };
    let lines = forward_lines(&mut child, log);
    Ok(PendingFleet {
        process: ServiceProcess::Step(StepProcess {
            child,
            dir: launch.work.to_path_buf(),
            scancel: launch.steps.scancel.clone(),
        }),
        lines,
        nodes,
        startup: launch.startup,
    })
}

impl PendingFleet {
    /// Waits until every node's server is ready, and reads how to reach
    /// each. A server that ends first is named, with what the step wrote.
    ///
    /// # Errors
    /// The caller cancelled, a server or the step ended, or a server was not
    /// ready in time.
    pub fn wait(mut self, cancel: &CancelToken) -> Result<Fleet, Failure> {
        let mut kept: Vec<String> = Vec::new();
        let deadline = Instant::now() + self.startup;
        loop {
            let waiting: Vec<&PendingNode> =
                self.nodes.iter().filter(|n| !n.ready.exists()).collect();
            if waiting.is_empty() {
                break;
            }
            if cancel.is_cancelled() {
                return Err(Failure::Cancelled);
            }
            for node in &waiting {
                if let Ok(status) = std::fs::read_to_string(node.work.join(TASK_EXIT_FILE)) {
                    return Err(Failure::Unmeasured(format!(
                        "the brain server on {} exited (status {}) before it served: {}",
                        node.node,
                        status.trim(),
                        recent_output(&self.lines, &mut kept)
                    )));
                }
            }
            if let Ok(Some(status)) = self.process.child().try_wait() {
                return Err(Failure::Unmeasured(format!(
                    "the step serving {} ended ({status}) before the server on {} served: {}",
                    names(&self.nodes),
                    names(waiting.iter().copied()),
                    recent_output(&self.lines, &mut kept)
                )));
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Err(Failure::Unmeasured(format!(
                    "the brain server on {} did not report ready within {}s: {}",
                    names(waiting.iter().copied()),
                    self.startup.as_secs(),
                    recent_output(&self.lines, &mut kept)
                )));
            }
            match self
                .lines
                .recv_timeout(left.min(Duration::from_millis(100)))
            {
                Ok(line) => remember(&mut kept, line),
                Err(RecvTimeoutError::Timeout) => {}
                // Both streams closed: the exit is reported by `try_wait`
                // above once the process is reaped.
                Err(RecvTimeoutError::Disconnected) => {
                    std::thread::sleep(Duration::from_millis(20));
                }
            }
        }
        let mut servers = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let named = |why: String, kept: &mut Vec<String>| {
                Failure::Unmeasured(format!(
                    "the brain server on {}: {why}; the step wrote: {}",
                    node.node,
                    recent_output(&self.lines, kept)
                ))
            };
            let key = crate::endpoint::BrainEndpoints::read_key(&node.key_file)
                .map_err(|e| named(e.to_string(), &mut kept))?;
            let origin = bound_origin(&node.work, cancel).map_err(|why| named(why, &mut kept))?;
            servers.push(NodeEndpoint {
                origin,
                key,
                slots: node.slots,
            });
        }
        Ok(Fleet {
            process: self.process,
            lines: self.lines,
            kept,
            servers,
        })
    }
}

/// The nodes of `pending`, as one list.
fn names<'a>(pending: impl IntoIterator<Item = &'a PendingNode>) -> String {
    pending
        .into_iter()
        .map(|n| n.node.as_str())
        .collect::<Vec<_>>()
        .join(",")
}

/// How long a server on a node may take to publish its connection record
/// once it reported ready: the shared filesystem may show the file late.
const RECORD_WAIT: Duration = Duration::from_secs(60);

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
