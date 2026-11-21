// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems that run on a model
// service they own the lifetime of, so that inference, training and
// evaluation share one set of cards, for its clients. If your team needs
// expertise in operating a model server as part of a training pipeline, you
// can procure our services by sending an email to info@swedishembedded.com.

//! The brain server a run is requested from.
//!
//! Splinter reaches brain only through its API. A run is either given a
//! server (`--brain-endpoint`) or starts one it owns for the length of the
//! run: a `brain serve` over the model store, with adapters served from a
//! directory and the training workload on, so inference, evaluation and
//! training jobs share its cards. [`BrainConnection`] is that choice, and
//! [`BrainConnection::endpoints`] is what every request is made against.

use std::net::{IpAddr, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use splinter_sandbox::step::{write_secret, NodeSteps};
use sven_sdk::CancelToken;

use crate::endpoint::BrainEndpoints;
use crate::error::PolicyError;
use crate::served_train::ArtifactStore;
use crate::server::{remember, Failure};
use crate::service_launch::{
    spawn_service, NodeService, Service, ServiceHost, ServiceLaunch, ServiceProcess,
};

/// How long a server Splinter starts may take to bind and report ready: it
/// scans the model store before it does.
const STARTUP: Duration = Duration::from_secs(600);

/// What it takes to start the server a run owns.
#[derive(Clone, Debug)]
pub struct ServiceSettings {
    /// The `brain` binary.
    pub binary: PathBuf,
    /// Brain's model store.
    pub models: PathBuf,
    /// Where the service keeps its state for the run: its scratch directory,
    /// the adapters it serves and the artifact store training moves data
    /// through. Created if absent.
    pub state: PathBuf,
    /// The compute it may schedule, in brain's `--device` grammar.
    pub device: Option<String>,
    /// How the hardware is driven (`--backend`).
    pub backend: Option<String>,
    /// Sequences it decodes together, which is also how many requests are
    /// asked of it at once.
    pub max_batch: Option<u32>,
    /// Whether adapters written to its adapter directory are served; a
    /// server whose store holds no base an adapter can sit on refuses to
    /// start with it.
    pub serve_adapters: bool,
    /// The base the adapters it serves sit on (`--checkpoint`): brain binds
    /// every adapter of the directory to one served base, and in a store of
    /// several it would pick another than the one the run trains (found
    /// against a real server, 2026-11-18). `None` leaves the choice to brain.
    pub adapter_base: Option<PathBuf>,
}

/// The file in a fleet's directory that lists its endpoints, one URL per
/// line; the key is the file `key` beside it.
pub const ENDPOINTS_FILE: &str = "endpoints";

/// Where a fleet of servers runs: one `brain serve` per node of the
/// allocation, each a job step serving every card of its node.
#[derive(Clone, Debug)]
pub struct FleetSettings {
    /// What each server is started with. Its state directory must be on a
    /// filesystem every node sees: the ready markers, the connection records,
    /// the adapters and the artifact store are shared through it. A device
    /// left unnamed is every GPU of the node (`gpu0,...`), as in-process
    /// replicas.
    pub service: ServiceSettings,
    /// How a step is started on a node.
    pub steps: NodeSteps,
    /// The nodes, by name; each must resolve to the address it is reached
    /// at.
    pub nodes: Vec<String>,
    /// The GPUs each node gives its server.
    pub gpus_per_node: u32,
}

/// The brain servers this process started: stopped, and their scratch
/// directory removed, when dropped.
pub struct ManagedBrain {
    processes: Vec<ServiceProcess>,
    endpoints: BrainEndpoints,
    work: PathBuf,
    /// The last lines the servers wrote, kept for an operator to read when a
    /// request to one fails.
    output: Arc<Mutex<Vec<String>>>,
}

/// The state directories every server of a run shares: adapters offered,
/// artifacts moved, and the scratch directory `work`.
fn prepare(state: &Path, work: &Path) -> Result<(PathBuf, PathBuf), PolicyError> {
    let adapters = state.join("adapters");
    let artifacts = state.join("artifacts");
    for dir in [&adapters, &artifacts, &work.to_path_buf()] {
        std::fs::create_dir_all(dir).map_err(|source| PolicyError::Io {
            path: dir.clone(),
            source,
        })?;
    }
    Ok((adapters, artifacts))
}

fn refused(what: &str, reason: String) -> PolicyError {
    PolicyError::Endpoint {
        url: what.to_string(),
        reason,
    }
}

fn failure_reason(failure: Failure, what: &str) -> String {
    match failure {
        Failure::Cancelled => format!("starting {what} was cancelled"),
        Failure::Unmeasured(why) => why,
    }
}

impl ManagedBrain {
    /// Starts a server on this machine as `settings` say and waits until it
    /// is ready.
    pub fn start(settings: &ServiceSettings, cancel: &CancelToken) -> Result<Self, PolicyError> {
        let work = settings.state.join(format!("run-{}", std::process::id()));
        let (adapters, artifacts) = prepare(&settings.state, &work)?;
        let launch = settings.launch(&work, &adapters, &artifacts, None, ServiceHost::Local);
        let started = spawn_service(&launch).and_then(|pending| pending.wait(cancel));
        let service = match started {
            Ok(service) => service,
            Err(failure) => {
                let _ = std::fs::remove_dir_all(&work);
                return Err(refused(
                    &settings.binary.display().to_string(),
                    failure_reason(failure, "the brain server"),
                ));
            }
        };
        let endpoints = BrainEndpoints::new(vec![service.url()], service.key.clone())?
            .with_attempt_concurrency(service.slots as usize)
            .with_adapter_directory(adapters)
            .with_artifact_store(ArtifactStore::open(&artifacts)?);
        Ok(Self::running(
            vec![(String::new(), service)],
            endpoints,
            work,
        ))
    }

    /// Starts one server per node of `settings.nodes`, each a job step bound
    /// to its node's address and serving every card of it, all answering to
    /// one key, and waits until every one is ready. The endpoints are asked
    /// as many attempts at once as the servers decode together.
    ///
    /// # Errors
    /// A node does not resolve or its server does not start, named; the
    /// servers already started are stopped.
    pub fn start_fleet(
        settings: &FleetSettings,
        cancel: &CancelToken,
    ) -> Result<Self, PolicyError> {
        let service = &settings.service;
        if settings.nodes.is_empty() {
            return Err(refused("fleet", "a fleet needs at least one node".into()));
        }
        let work = service.state.join(format!("fleet-{}", std::process::id()));
        let (adapters, artifacts) = prepare(&service.state, &work)?;
        let started = Self::start_steps(settings, &work, &adapters, &artifacts, cancel);
        let services = match started {
            Ok(services) => services,
            Err(error) => {
                let _ = std::fs::remove_dir_all(&work);
                return Err(error);
            }
        };
        let slots: usize = services.iter().map(|(_, s)| s.slots as usize).sum();
        let urls = services.iter().map(|(_, s)| s.url()).collect();
        let key = services[0].1.key.clone();
        // One URL per line, for a client that is pointed at the fleet by file.
        let listing = work.join(ENDPOINTS_FILE);
        let text: String = services
            .iter()
            .map(|(_, s)| format!("{}\n", s.url()))
            .collect();
        if let Err(source) = std::fs::write(&listing, text) {
            let _ = std::fs::remove_dir_all(&work);
            return Err(PolicyError::Io {
                path: listing,
                source,
            });
        }
        let endpoints = BrainEndpoints::new(urls, key)?
            .with_attempt_concurrency(slots)
            .with_adapter_directory(adapters)
            .with_artifact_store(ArtifactStore::open(&artifacts)?);
        Ok(Self::running(services, endpoints, work))
    }

    /// Every node's server started, then waited for; the first that fails
    /// names its node and stops the rest.
    fn start_steps(
        settings: &FleetSettings,
        work: &Path,
        adapters: &Path,
        artifacts: &Path,
        cancel: &CancelToken,
    ) -> Result<Vec<(String, Service)>, PolicyError> {
        // Owner-only: the key and the scripts of every server.
        set_owner_only(work)?;
        let key_file = work.join("key");
        write_secret(&key_file).map_err(|source| PolicyError::Io {
            path: key_file.clone(),
            source,
        })?;
        let every_card = (settings.gpus_per_node > 0).then(|| {
            (0..settings.gpus_per_node)
                .map(|i| format!("gpu{i}"))
                .collect::<Vec<_>>()
                .join(",")
        });
        let device = settings.service.device.clone().or(every_card);
        let mut pending = Vec::with_capacity(settings.nodes.len());
        // Started from this thread, one after the other, so each loads while
        // the next starts; a server dies with the thread that started it.
        for node in &settings.nodes {
            let address = node_address(node)?;
            let node_work = work.join(node);
            std::fs::create_dir_all(&node_work).map_err(|source| PolicyError::Io {
                path: node_work.clone(),
                source,
            })?;
            // brain publishes its connection record only in a directory no
            // other user can enter.
            set_owner_only(&node_work)?;
            let log = work.join(format!("{node}.log"));
            let mut service = settings.service.clone();
            service.device.clone_from(&device);
            let host = ServiceHost::Node(NodeService {
                steps: &settings.steps,
                node,
                address,
                gpus: settings.gpus_per_node,
                key_file: &key_file,
            });
            let launch = service.launch(&node_work, adapters, artifacts, Some(&log), host);
            let started = spawn_service(&launch).map_err(|failure| {
                refused(
                    node,
                    failure_reason(failure, &format!("the brain server on {node}")),
                )
            })?;
            pending.push((node.clone(), started));
        }
        let mut services = Vec::with_capacity(pending.len());
        for (node, started) in pending {
            let service = started.wait(cancel).map_err(|failure| {
                refused(
                    &node,
                    format!(
                        "the brain server on {node} did not start: {}",
                        failure_reason(failure, &format!("the brain server on {node}"))
                    ),
                )
            })?;
            services.push((node, service));
        }
        Ok(services)
    }

    fn running(services: Vec<(String, Service)>, endpoints: BrainEndpoints, work: PathBuf) -> Self {
        // Everything the servers write is read as it comes, so a full pipe
        // never stops one; only the last lines are kept.
        let output = Arc::new(Mutex::new(Vec::new()));
        let mut processes = Vec::with_capacity(services.len());
        for (node, service) in services {
            let Service {
                process,
                lines,
                kept,
                ..
            } = service;
            let tag = |line: String| {
                if node.is_empty() {
                    line
                } else {
                    format!("{node}: {line}")
                }
            };
            {
                let mut held = output
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                for line in kept {
                    remember(&mut held, tag(line));
                }
            }
            let drained = Arc::clone(&output);
            let node = node.clone();
            std::thread::spawn(move || {
                for line in lines {
                    let line = if node.is_empty() {
                        line
                    } else {
                        format!("{node}: {line}")
                    };
                    remember(
                        &mut drained
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner),
                        line,
                    );
                }
            });
            processes.push(process);
        }
        Self {
            processes,
            endpoints,
            work,
            output,
        }
    }

    /// The last lines the servers wrote, as one line of text.
    #[must_use]
    pub fn recent_output(&self) -> String {
        self.output
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .join(" | ")
    }

    /// The endpoints that reach these servers.
    #[must_use]
    pub fn endpoints(&self) -> &BrainEndpoints {
        &self.endpoints
    }

    /// The scratch directory of these servers: for a fleet, it holds the
    /// [`ENDPOINTS_FILE`] and the key file `key` for clients outside this
    /// process, while the fleet runs.
    #[must_use]
    pub fn work_dir(&self) -> &Path {
        &self.work
    }
}

impl Drop for ManagedBrain {
    fn drop(&mut self) {
        // Stopped before their scratch directory goes: the keys live there.
        self.processes.clear();
        let _ = std::fs::remove_dir_all(&self.work);
    }
}

impl ServiceSettings {
    fn launch<'a>(
        &'a self,
        work: &'a Path,
        adapters: &'a Path,
        artifacts: &'a Path,
        log: Option<&'a Path>,
        host: ServiceHost<'a>,
    ) -> ServiceLaunch<'a> {
        ServiceLaunch {
            binary: &self.binary,
            work,
            models: &self.models,
            adapters: self.serve_adapters.then_some(adapters),
            adapter_base: self.adapter_base.as_deref(),
            artifacts,
            device: self.device.as_deref(),
            backend: self.backend.as_deref(),
            max_batch: self.max_batch,
            startup: STARTUP,
            log,
            host,
        }
    }
}

/// The address `node` is reached at: its first IPv4 address, else its first.
fn node_address(node: &str) -> Result<IpAddr, PolicyError> {
    let addresses: Vec<IpAddr> = (node, 0)
        .to_socket_addrs()
        .map_err(|e| refused(node, format!("the node {node} does not resolve: {e}")))?
        .map(|address| address.ip())
        .collect();
    addresses
        .iter()
        .find(|ip| ip.is_ipv4())
        .or_else(|| addresses.first())
        .copied()
        .ok_or_else(|| refused(node, format!("the node {node} resolves to no address")))
}

fn set_owner_only(dir: &Path) -> Result<(), PolicyError> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700)).map_err(|source| {
        PolicyError::Io {
            path: dir.to_path_buf(),
            source,
        }
    })
}

/// How a run reaches brain: a server it was given, or one it owns.
pub enum BrainConnection {
    /// A server somebody else runs.
    Given(BrainEndpoints),
    /// A server this process started.
    Owned(ManagedBrain),
}

impl BrainConnection {
    /// The endpoints requests are made against.
    #[must_use]
    pub fn endpoints(&self) -> &BrainEndpoints {
        match self {
            Self::Given(endpoints) => endpoints,
            Self::Owned(owned) => owned.endpoints(),
        }
    }
}
