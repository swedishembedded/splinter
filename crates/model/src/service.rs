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
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use splinter_core::digest::Digest;
use splinter_sandbox::step::{write_secret, NodeSteps};
use sven_sdk::CancelToken;

use crate::endpoint::{BrainEndpoints, ServersJob, TrainSpan};
use crate::error::PolicyError;
use crate::fleet_launch::{spawn_fleet, Fleet, FleetLaunch, NodeEndpoint, NodeServer};
use crate::served_train::ArtifactStore;
use crate::server::{remember, Failure};
use crate::service_launch::{spawn_service, Service, ServiceLaunch, ServiceProcess};

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

/// What a training job over the fleet's servers is set up with when the
/// servers start: the secret every server proves membership with, and the
/// address the first server binds as the job's coordinator.
struct TrainJob {
    secret: PathBuf,
    secret_sha256: String,
    coordinator: String,
}

impl TrainJob {
    /// A fresh secret in `work` and a coordinator port on `rank0`, the first
    /// server's address: drawn at random from the range above the usual
    /// ephemeral ports' start (the node's system cannot be asked from here; a
    /// clash fails the job with the reason).
    fn prepare(work: &Path, rank0: IpAddr) -> Result<Self, PolicyError> {
        use std::io::Read;
        let secret = work.join(TRAIN_JOB_SECRET_FILE);
        let io = |path: &Path| {
            let path = path.to_path_buf();
            move |source| PolicyError::Io { path, source }
        };
        write_secret(&secret).map_err(io(&secret))?;
        let secret_sha256 = secret_sha256(&secret)?;
        let mut port = [0u8; 2];
        std::fs::File::open("/dev/urandom")
            .and_then(|mut random| random.read_exact(&mut port))
            .map_err(io(Path::new("/dev/urandom")))?;
        let port = 20_000 + u16::from_le_bytes(port) % 30_000;
        Ok(Self {
            secret,
            secret_sha256,
            coordinator: std::net::SocketAddr::new(rank0, port).to_string(),
        })
    }

    /// How a data-parallel training run is spread over the fleet: one job
    /// over every card of every server when there are several, every card of
    /// the one server when it has several, one card otherwise; and one card
    /// whenever the servers' `brain/train` does not take `cards` and `job`.
    fn span(&self, endpoints: &BrainEndpoints, servers: usize, cards: u32) -> TrainSpan {
        if servers <= 1 && cards <= 1 {
            return TrainSpan::OneCard;
        }
        match trains_across_cards(endpoints) {
            Ok(true) => {}
            Ok(false) => {
                eprintln!(
                    "brain: these servers' brain/train takes no cards and job: a training run \
                     uses one card"
                );
                return TrainSpan::OneCard;
            }
            Err(why) => {
                eprintln!(
                    "brain: the servers' capabilities could not be read ({why}): a training run \
                     uses one card"
                );
                return TrainSpan::OneCard;
            }
        }
        if servers <= 1 {
            return TrainSpan::Cards(cards);
        }
        TrainSpan::Servers(ServersJob {
            cards: vec![cards.max(1); servers],
            coordinator: self.coordinator.clone(),
            secret_sha256: self.secret_sha256.clone(),
        })
    }
}

/// The SHA-256, in hex, of the 32 bytes a job secret file holds as 64 hex
/// digits (what `--train-job-secret` and `--job-secret` read): the digest a
/// job over several servers names, never the secret.
///
/// # Errors
/// The file cannot be read or is not 64 hex digits.
pub fn secret_sha256(file: &Path) -> Result<String, PolicyError> {
    let text = std::fs::read_to_string(file).map_err(|source| PolicyError::Io {
        path: file.to_path_buf(),
        source,
    })?;
    let hex = text.trim();
    let bytes: Option<Vec<u8>> = (hex.len() == 64)
        .then(|| {
            (0..64)
                .step_by(2)
                .map(|at| u8::from_str_radix(&hex[at..at + 2], 16).ok())
                .collect()
        })
        .flatten();
    let bytes = bytes.ok_or_else(|| {
        refused(
            &file.display().to_string(),
            "a job secret is 32 bytes written as 64 hex digits".into(),
        )
    })?;
    let digest = Digest::sha256_of_reader(bytes.as_slice()).map_err(|source| PolicyError::Io {
        path: file.to_path_buf(),
        source,
    })?;
    Ok(digest.hex().to_string())
}

/// Whether the first endpoint's `brain/train` `train` action takes `cards`
/// and `job` (`GET /v1/capabilities`): what a run over several cards and
/// servers needs.
fn trains_across_cards(endpoints: &BrainEndpoints) -> Result<bool, String> {
    let listing = endpoints
        .get(0, "capabilities")
        .map_err(|e| e.to_string())?;
    Ok(takes_cards_and_job(&listing))
}

/// Whether a capability listing (`{"data": [{"model", "action", "params":
/// [{"name"}]}]}`) has `brain/train`'s `train` take `cards` and `job`.
fn takes_cards_and_job(listing: &serde_json::Value) -> bool {
    crate::served_train::action_params(listing, "train").is_some_and(|params| {
        ["cards", "job"]
            .iter()
            .all(|p| params.iter().any(|n| n == p))
    })
}

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
    /// Serve plaintext HTTP between the nodes instead of TLS: only where the
    /// cluster's network is trusted, since the key crosses it readable. By
    /// default the fleet makes a certificate authority of its own, serves
    /// TLS under it and this process's model clients trust it.
    pub insecure_plaintext: bool,
    /// A node-local directory each server stages the model store's entries
    /// in (brain's `--stage-models`): the first process on a node copies an
    /// entry there, verified, and the node's others read the copy instead of
    /// the shared filesystem. `None` reads the shared store.
    pub stage_models: Option<PathBuf>,
}

/// The directory, under a service's state, adapters are served from.
pub const ADAPTERS_DIR: &str = "adapters";

/// The directory, under a service's state, of the artifact store training
/// moves data through.
pub const ARTIFACTS_DIR: &str = "artifacts";

/// The file in a fleet's directory holding the job secret its servers were
/// started with (`--train-job-secret`), for a client outside this process
/// that trains one job over them.
pub const TRAIN_JOB_SECRET_FILE: &str = "train-job.secret";

/// The file in a server's or a fleet's directory holding the key its
/// endpoints answer to, readable by its owner only.
pub const KEY_FILE: &str = "key";

/// The file in a fleet's directory holding the certificate authority its
/// servers' certificates chain to, for clients outside this process.
pub const AUTHORITY_FILE: &str = "ca.pem";

/// The brain servers this process started: stopped, and their scratch
/// directory removed, when dropped.
pub struct ManagedBrain {
    /// `None` once stopped, before the scratch directory goes.
    process: Option<ServiceProcess>,
    endpoints: BrainEndpoints,
    work: PathBuf,
    /// The last lines the servers wrote, kept for an operator to read when a
    /// request to one fails.
    output: Arc<Mutex<Vec<String>>>,
}

/// The state directories every server of a run shares: adapters offered,
/// artifacts moved, and the scratch directory `work`.
fn prepare(state: &Path, work: &Path) -> Result<(PathBuf, PathBuf), PolicyError> {
    let adapters = state.join(ADAPTERS_DIR);
    let artifacts = state.join(ARTIFACTS_DIR);
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
        let launch = settings.launch(&work, &adapters, &artifacts, None);
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
        // The same files a fleet publishes, for a client outside this process.
        let published = write_private(&work.join(ENDPOINTS_FILE), &format!("{}\n", service.url()))
            .and_then(|()| write_private(&work.join(KEY_FILE), &format!("{}\n", service.key)));
        if let Err(error) = published {
            let _ = std::fs::remove_dir_all(&work);
            return Err(error);
        }
        let endpoints = BrainEndpoints::new(vec![service.url()], service.key.clone())?
            .with_attempt_concurrency(service.slots as usize)
            .with_adapter_directory(adapters)
            .with_artifact_store(ArtifactStore::open(&artifacts)?);
        let Service {
            process,
            lines,
            kept,
            ..
        } = service;
        Ok(Self::running(process, lines, kept, endpoints, work))
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
        let (fleet, train_job) = match started {
            Ok(started) => started,
            Err(error) => {
                let _ = std::fs::remove_dir_all(&work);
                return Err(error);
            }
        };
        let slots: usize = fleet.servers.iter().map(|s| s.slots as usize).sum();
        let urls: Vec<String> = fleet.servers.iter().map(NodeEndpoint::url).collect();
        let key = fleet.servers[0].key.clone();
        // One URL per line, for a client that is pointed at the fleet by file.
        let listing = work.join(ENDPOINTS_FILE);
        let text: String = urls.iter().map(|url| format!("{url}\n")).collect();
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
        let span = train_job.span(&endpoints, settings.nodes.len(), settings.gpus_per_node);
        let endpoints = endpoints.with_train_span(span);
        let Fleet {
            process,
            lines,
            kept,
            ..
        } = fleet;
        Ok(Self::running(process, lines, kept, endpoints, work))
    }

    /// Every node's server started as a task of one step, then waited for;
    /// a server that fails is named and the step, with every server, is
    /// stopped.
    fn start_steps(
        settings: &FleetSettings,
        work: &Path,
        adapters: &Path,
        artifacts: &Path,
        cancel: &CancelToken,
    ) -> Result<(Fleet, TrainJob), PolicyError> {
        // Owner-only: the key and the scripts of every server.
        set_owner_only(work)?;
        let key_file = work.join(KEY_FILE);
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
        let addresses = settings
            .nodes
            .iter()
            .map(|node| node_address(node).map(|address| (node.clone(), address)))
            .collect::<Result<Vec<_>, _>>()?;
        let identities = if settings.insecure_plaintext {
            None
        } else {
            Some(Self::trust_fleet(work, &addresses)?)
        };
        let train_job = TrainJob::prepare(work, addresses[0].1)?;
        let mut service = settings.service.clone();
        service.device.clone_from(&device);
        let mut node_dirs = Vec::with_capacity(addresses.len());
        for (index, (node, _)) in addresses.iter().enumerate() {
            let node_work = work.join(node);
            std::fs::create_dir_all(&node_work).map_err(|source| PolicyError::Io {
                path: node_work.clone(),
                source,
            })?;
            // brain publishes its connection record only in a directory no
            // other user can enter.
            set_owner_only(&node_work)?;
            let tls = match &identities {
                Some(identities) => {
                    let chain = node_work.join("tls.crt");
                    let private_key = node_work.join("tls.key");
                    write_private(&chain, &identities[index].chain_pem)?;
                    write_private(&private_key, &identities[index].key_pem)?;
                    Some((chain, private_key))
                }
                None => None,
            };
            node_dirs.push((node_work, tls));
        }
        let servers = addresses
            .iter()
            .zip(&node_dirs)
            .enumerate()
            .map(|(index, ((node, address), (node_work, tls)))| NodeServer {
                launch: service.launch(node_work, adapters, artifacts, None),
                node,
                address: *address,
                key_file: &key_file,
                tls: tls.as_ref().map(|(c, k)| (c.as_path(), k.as_path())),
                stage: settings.stage_models.as_deref(),
                train_job: Some((
                    train_job.secret.as_path(),
                    (index == 0).then_some(train_job.coordinator.as_str()),
                )),
            })
            .collect();
        let log = work.join("fleet.log");
        let launch = FleetLaunch {
            steps: &settings.steps,
            work,
            gpus_per_node: settings.gpus_per_node,
            log: Some(&log),
            startup: STARTUP,
            servers,
        };
        let nodes = settings.nodes.join(",");
        let fleet = spawn_fleet(&launch)
            .and_then(|pending| pending.wait(cancel))
            .map_err(|failure| {
                refused(
                    &nodes,
                    failure_reason(failure, &format!("the brain servers on {nodes}")),
                )
            })?;
        Ok((fleet, train_job))
    }

    /// Issues the fleet's certificates for `nodes`, publishes the authority
    /// as [`AUTHORITY_FILE`] in `work` and makes this process's model clients
    /// trust it.
    fn trust_fleet(
        work: &Path,
        nodes: &[(String, IpAddr)],
    ) -> Result<Vec<crate::fleet_tls::ServerIdentity>, PolicyError> {
        let issued = crate::fleet_tls::issue(nodes)?;
        write_private(&work.join(AUTHORITY_FILE), &issued.authority_pem)?;
        crate::endpoint::trust_authorities(issued.authority_pem.as_bytes())
            .map_err(|reason| refused("fleet", reason))?;
        Ok(issued.servers)
    }

    /// The servers' process (the local server, or the fleet's step), what
    /// it writes and the last lines it wrote so far.
    fn running(
        process: ServiceProcess,
        lines: Receiver<String>,
        kept: Vec<String>,
        endpoints: BrainEndpoints,
        work: PathBuf,
    ) -> Self {
        // Everything the servers write is read as it comes, so a full pipe
        // never stops one; only the last lines are kept.
        let output = Arc::new(Mutex::new(Vec::new()));
        {
            let mut held = output
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            for line in kept {
                remember(&mut held, line);
            }
        }
        let drained = Arc::clone(&output);
        std::thread::spawn(move || {
            for line in lines {
                remember(
                    &mut drained
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner),
                    line,
                );
            }
        });
        Self {
            process: Some(process),
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
        self.process = None;
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

/// Writes `text` to a new file at `path` that only its owner can read.
fn write_private(path: &Path, text: &str) -> Result<(), PolicyError> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let io = |source| PolicyError::Io {
        path: path.to_path_buf(),
        source,
    };
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)
        .map_err(io)?;
    file.write_all(text.as_bytes()).map_err(io)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_job_secrets_digest_is_of_its_32_bytes_and_the_coordinator_on_rank_0s_node() {
        let dir = tempfile::tempdir().unwrap();
        let job = TrainJob::prepare(dir.path(), "10.0.0.7".parse().unwrap()).unwrap();
        let hex = std::fs::read_to_string(&job.secret).unwrap();
        let bytes: Vec<u8> = (0..64)
            .step_by(2)
            .map(|at| u8::from_str_radix(&hex.trim()[at..at + 2], 16).unwrap())
            .collect();
        assert_eq!(bytes.len(), 32);
        let expected = Digest::sha256_of_reader(bytes.as_slice()).unwrap();
        assert_eq!(job.secret_sha256, expected.hex());
        assert!(
            job.coordinator.starts_with("10.0.0.7:"),
            "{}",
            job.coordinator
        );
    }

    #[test]
    fn a_run_spans_cards_and_servers_only_where_brain_takes_cards_and_job() {
        let listing = |params: &[&str]| {
            serde_json::json!({"data": [
                {"model": "brain/train", "action": "check_dataset", "params": [{"name": "cards"}, {"name": "job"}]},
                {"model": "brain/train", "action": "train",
                 "params": params.iter().map(|p| serde_json::json!({"name": p})).collect::<Vec<_>>()},
            ]})
        };
        assert!(takes_cards_and_job(&listing(&["seed", "cards", "job"])));
        assert!(
            !takes_cards_and_job(&listing(&["seed", "cards"])),
            "an older brain"
        );
        assert!(!takes_cards_and_job(&serde_json::json!({"data": []})));

        let dir = tempfile::tempdir().unwrap();
        let job = TrainJob::prepare(dir.path(), "10.0.0.7".parse().unwrap()).unwrap();
        let unreachable = BrainEndpoints::new(vec!["http://127.0.0.1:1/v1".into()], "k").unwrap();
        assert_eq!(job.span(&unreachable, 1, 1), TrainSpan::OneCard);
        assert_eq!(
            job.span(&unreachable, 2, 4),
            TrainSpan::OneCard,
            "capabilities that cannot be read train on one card, and say so"
        );
    }
}
