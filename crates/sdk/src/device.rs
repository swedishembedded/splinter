// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements controlled placement of model work on named
// GPUs and CPU cores for its clients. If your team needs expertise in
// scheduling learning workloads on shared hardware, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The flags every program that asks a model takes, declared once: where
//! brain is, and on which compute it runs.
//!
//! A program flattens [`DeviceFlags`] into its command line, applies it to
//! its [`Config`] and, where it asks a model directly, takes the brain it is
//! to ask from [`DeviceFlags::connection`]. Splinter reaches brain only
//! through its API: `--brain-endpoint` names a server that is running, and
//! without it the program starts one of its own from `--brain-bin`. `--device`
//! is brain's grammar, handed to that server; no environment variable chooses
//! hardware.

use std::path::PathBuf;

use crate::cluster::{ClusterMode, ClusterSettings, StageModels};
use crate::model::endpoint::{ServersJob, TrainSpan};
use crate::model::{BrainConnection, BrainEndpoints, PolicyError};
use crate::Config;

/// `--device`, `--backend`, `--vram-budget` and the flags that name the brain
/// server, as a program declares them.
#[derive(clap::Args, Clone, Debug, Default)]
pub struct DeviceFlags {
    /// Which compute the brain server may schedule: `gpu`, `gpu1`,
    /// `gpu0,gpu1`, `cpu0-7`, `vulkan`, ... in brain's grammar. Without it,
    /// every device the machine has.
    #[arg(long, global = true)]
    pub device: Option<String>,
    /// How the hardware is driven: wgpu, vulkan, cuda or cpu. Without it
    /// brain picks, which on a CUDA host is not CUDA.
    #[arg(long, global = true)]
    pub backend: Option<String>,
    /// The VRAM, in GiB, this process was granted: every card the server
    /// budgets from is capped at it. Without it the whole card is budgeted.
    #[arg(long, global = true)]
    pub vram_budget: Option<f64>,
    /// A running `brain serve --openai` to request every local model from
    /// (its base URL, ending in `/v1`); repeat for more. Without one, a
    /// server is started from `--brain-bin`.
    #[arg(long, global = true)]
    pub brain_endpoint: Vec<String>,
    /// The file holding the endpoints' API key.
    #[arg(long, global = true)]
    pub brain_key_file: Option<PathBuf>,
    /// A PEM file of certificate authorities the endpoints' https://
    /// certificates may chain to (the `ca.pem` a fleet that serves TLS under
    /// its own authority publishes); added to the trusted roots, never in
    /// place of verification.
    #[arg(long, global = true, value_name = "FILE")]
    pub brain_ca_file: Option<PathBuf>,
    /// With --brain-endpoint: the cards a data-parallel training run takes
    /// on each endpoint, in their order (`2,1`); one number for one endpoint
    /// trains on that many of its cards, several make one job over the
    /// endpoints, which needs --brain-train-job-secret and
    /// --brain-train-coordinator. Without it a run takes one card.
    #[arg(long, global = true, value_name = "N[,N...]", value_delimiter = ',')]
    pub brain_train_cards: Vec<u32>,
    /// The job secret the endpoints were started with
    /// (`brain serve --train-job-secret FILE`, 64 hex digits).
    #[arg(long, global = true, value_name = "FILE")]
    pub brain_train_job_secret: Option<PathBuf>,
    /// The address the first endpoint's server binds as the job's
    /// coordinator (`brain serve --train-job-coordinator HOST:PORT`).
    #[arg(long, global = true, value_name = "HOST:PORT")]
    pub brain_train_coordinator: Option<String>,
    /// The `brain` binary a server is started from (default: the first on
    /// `PATH`).
    #[arg(long, global = true)]
    pub brain_bin: Option<PathBuf>,
    /// Whether a Slurm allocation this process runs inside is used: `auto`
    /// (default) serves models on every usable node of a multi-node
    /// allocation and spreads a distributed job's ranks over them; `off`
    /// keeps everything on this machine.
    #[arg(long, global = true, value_name = "MODE", value_parser = cluster_mode)]
    pub cluster: Option<ClusterMode>,
    /// The nodes of the allocation this process may use, as a Slurm host
    /// list (`n[442,445]`); default: every node of the allocation. Naming
    /// nodes uses them even when only one is named.
    #[arg(long, global = true, value_name = "HOSTLIST")]
    pub cluster_nodes: Option<String>,
    /// `NODE=CARDS`: the node's brain server serves, and a training job's
    /// ranks there train on, only these of its cards, by index
    /// (`n80=0,1`), for a node that brings fewer cards than the others; the
    /// other nodes serve every card. Repeat for more nodes.
    #[arg(long = "cluster-node-cards", global = true, value_name = "NODE=CARDS",
        value_parser = node_cards)]
    pub cluster_node_cards: Vec<(String, Vec<u32>)>,
    /// The words a program started on a node runs under, split at spaces: a
    /// step starts on the node's host, so a container this process runs in
    /// is entered again there (`apptainer exec --nv IMAGE`, or a wrapper
    /// script that also sets the node's environment).
    #[arg(long, global = true, value_name = "WORDS")]
    pub cluster_exec_prefix: Option<String>,
    /// The words a distributed job's ranks are started under instead of
    /// --cluster-exec-prefix, split at spaces; `{local_rank}` in them is a
    /// rank's place on its node, the card a wrapper pins it to.
    #[arg(long, global = true, value_name = "WORDS")]
    pub cluster_rank_exec_prefix: Option<String>,
    /// The `srun` program steps on the allocation's nodes are started with
    /// (default: the first on `PATH`).
    #[arg(long, global = true, value_name = "PATH")]
    pub slurm_srun: Option<PathBuf>,
    /// An argument every step passes to `srun` (`--mem=64G`); repeat for
    /// more.
    #[arg(
        long = "slurm-step-arg",
        global = true,
        value_name = "ARG",
        allow_hyphen_values = true
    )]
    pub slurm_step_arg: Vec<String>,
    /// An argument a step on a single node passes to `srun` besides
    /// `--slurm-step-arg` (on an HPE Slingshot fabric,
    /// `--network=single_node_vni`, without which a single-node step has no
    /// fabric access); repeat for more.
    #[arg(
        long = "slurm-single-node-step-arg",
        global = true,
        value_name = "ARG",
        allow_hyphen_values = true
    )]
    pub slurm_single_node_step_arg: Vec<String>,
    /// Serve plaintext HTTP between the allocation's nodes instead of TLS:
    /// only on a cluster network you trust, since the servers' key then
    /// crosses it readable. By default the servers serve TLS under a
    /// certificate authority made for them, whose key is discarded at once.
    #[arg(long, global = true)]
    pub cluster_insecure_plaintext: bool,
    /// Where brain stages the model store's entries on each node's own disk
    /// for the fleet's servers and a distributed job's brain ranks (the
    /// first process of a node copies an entry, verified; the others read the
    /// copy instead of the shared filesystem): `auto` (default) under the
    /// node-local scratch the site announces to every step (SLURM_TMPDIR,
    /// SNIC_TMP or LOCAL_SCRATCH), checked writable on every node, else not
    /// at all; an absolute directory there; `off` never. The directory must
    /// be bound into the container a step enters.
    #[arg(long, global = true, value_name = "DIR|auto|off", value_parser = stage_models)]
    pub cluster_stage_models: Option<StageModels>,
}

/// Removes every `name` from `args`; whether there was one.
fn take_flag(args: &mut Vec<String>, name: &str) -> bool {
    let before = args.len();
    args.retain(|arg| arg != name);
    args.len() != before
}

fn stage_models(text: &str) -> Result<StageModels, String> {
    StageModels::parse(text).map_err(|e| e.to_string())
}

/// `NODE=CARDS`: a node and the indices of its cards (`n80=0,1`).
fn node_cards(text: &str) -> Result<(String, Vec<u32>), String> {
    let usage = || format!("{text:?} is not NODE=CARDS (n80=0,1)");
    let (node, cards) = text.split_once('=').ok_or_else(usage)?;
    if node.is_empty() || cards.is_empty() {
        return Err(usage());
    }
    let cards = cards
        .split(',')
        .map(|card| card.trim().parse::<u32>().map_err(|_| usage()))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((node.to_string(), cards))
}

fn cluster_mode(text: &str) -> Result<ClusterMode, String> {
    ClusterMode::parse(text).map_err(|e| e.to_string())
}

impl DeviceFlags {
    /// Removes `--device VALUE`, `--backend VALUE`, `--vram-budget GIB`,
    /// `--brain-endpoint URL` (any number), `--brain-key-file FILE` and
    /// `--brain-bin FILE` from `args` and returns them, for a program that
    /// reads its own arguments.
    ///
    /// # Errors
    /// A flag has no value after it, or `--vram-budget` is not a positive number.
    pub fn take(args: &mut Vec<String>) -> Result<Self, String> {
        let cluster_insecure_plaintext = take_flag(args, "--cluster-insecure-plaintext");
        let mut take_all = |name: &str| -> Result<Vec<String>, String> {
            let mut values = Vec::new();
            while let Some(at) = args.iter().position(|a| a == name) {
                if at + 1 >= args.len() {
                    return Err(format!("{name} needs a value"));
                }
                values.push(args.remove(at + 1));
                args.remove(at);
            }
            Ok(values)
        };
        let device = take_all("--device")?.pop();
        let backend = take_all("--backend")?.pop();
        let vram_budget = take_all("--vram-budget")?
            .pop()
            .map(|text| match text.parse::<f64>() {
                Ok(gib) if gib > 0.0 && gib.is_finite() => Ok(gib),
                _ => Err(format!(
                    "--vram-budget needs a positive number of GiB, got {text}"
                )),
            })
            .transpose()?;
        let brain_endpoint = take_all("--brain-endpoint")?;
        let brain_key_file = take_all("--brain-key-file")?.pop().map(PathBuf::from);
        let brain_ca_file = take_all("--brain-ca-file")?.pop().map(PathBuf::from);
        let brain_bin = take_all("--brain-bin")?.pop().map(PathBuf::from);
        let brain_train_cards = take_all("--brain-train-cards")?
            .iter()
            .flat_map(|list| list.split(','))
            .map(|n| {
                n.trim()
                    .parse::<u32>()
                    .map_err(|_| format!("--brain-train-cards needs card counts, got {n}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let brain_train_job_secret = take_all("--brain-train-job-secret")?
            .pop()
            .map(PathBuf::from);
        let brain_train_coordinator = take_all("--brain-train-coordinator")?.pop();
        let cluster = take_all("--cluster")?
            .pop()
            .map(|text| cluster_mode(&text))
            .transpose()?;
        Ok(Self {
            device,
            backend,
            vram_budget,
            brain_endpoint,
            brain_key_file,
            brain_ca_file,
            brain_train_cards,
            brain_train_job_secret,
            brain_train_coordinator,
            brain_bin,
            cluster,
            cluster_nodes: take_all("--cluster-nodes")?.pop(),
            cluster_node_cards: take_all("--cluster-node-cards")?
                .iter()
                .map(|text| node_cards(text))
                .collect::<Result<_, _>>()?,
            cluster_exec_prefix: take_all("--cluster-exec-prefix")?.pop(),
            cluster_rank_exec_prefix: take_all("--cluster-rank-exec-prefix")?.pop(),
            slurm_srun: take_all("--slurm-srun")?.pop().map(PathBuf::from),
            slurm_step_arg: take_all("--slurm-step-arg")?,
            slurm_single_node_step_arg: take_all("--slurm-single-node-step-arg")?,
            cluster_insecure_plaintext,
            cluster_stage_models: take_all("--cluster-stage-models")?
                .pop()
                .map(|text| stage_models(&text))
                .transpose()?,
        })
    }

    /// The allocation settings these flags make of `detected` (what
    /// [`crate::config::detect_cluster`] found).
    #[must_use]
    pub fn cluster_settings(&self, detected: ClusterSettings) -> ClusterSettings {
        let mut cluster = detected;
        if let Some(mode) = self.cluster {
            cluster.mode = mode;
        }
        cluster.nodes.clone_from(&self.cluster_nodes);
        cluster.node_cards = self.cluster_node_cards.iter().cloned().collect();
        if let Some(prefix) = &self.cluster_exec_prefix {
            cluster.exec_prefix = prefix.split_whitespace().map(String::from).collect();
        }
        if let Some(prefix) = &self.cluster_rank_exec_prefix {
            cluster.rank_exec_prefix = Some(prefix.split_whitespace().map(String::from).collect());
        }
        if let Some(srun) = &self.slurm_srun {
            cluster.srun = Some(srun.clone());
        }
        cluster.step_args.clone_from(&self.slurm_step_arg);
        cluster
            .single_node_step_args
            .clone_from(&self.slurm_single_node_step_arg);
        cluster.insecure_plaintext = self.cluster_insecure_plaintext;
        if let Some(staging) = &self.cluster_stage_models {
            cluster.stage_models = staging.clone();
        }
        cluster
    }

    /// Puts the flags into `config`: the compute and the binary of the server
    /// a run starts, the endpoints of one it is given, and the batch
    /// allocation this process runs inside ([`crate::config::detect_cluster`])
    /// as the cluster flags restrict it.
    ///
    /// # Errors
    /// An endpoint is not a base URL, it has no key file, or the key file
    /// cannot be read.
    pub fn apply_to(&self, config: &mut Config) -> Result<(), PolicyError> {
        config.cluster = self.cluster_settings(crate::config::detect_cluster());
        config.brain_device.clone_from(&self.device);
        config.brain_backend.clone_from(&self.backend);
        if let Some(gib) = self.vram_budget {
            crate::model::device::set_vram_budget(gib);
        }
        if let Some(binary) = &self.brain_bin {
            config.brain_binary = Some(binary.clone());
        }
        if self.brain_endpoint.is_empty() {
            let stray = [
                (self.brain_ca_file.is_some(), "--brain-ca-file"),
                (!self.brain_train_cards.is_empty(), "--brain-train-cards"),
                (
                    self.brain_train_job_secret.is_some(),
                    "--brain-train-job-secret",
                ),
                (
                    self.brain_train_coordinator.is_some(),
                    "--brain-train-coordinator",
                ),
            ];
            return match stray.iter().find(|(given, _)| *given) {
                Some((_, flag)) => Err(PolicyError::Endpoint {
                    url: String::new(),
                    reason: format!("{flag} configures --brain-endpoint, which is not given"),
                }),
                None => Ok(()),
            };
        }
        if let Some(ca_file) = &self.brain_ca_file {
            let pem = std::fs::read(ca_file).map_err(|source| PolicyError::Io {
                path: ca_file.clone(),
                source,
            })?;
            crate::model::endpoint::trust_authorities(&pem).map_err(|reason| {
                PolicyError::Endpoint {
                    url: self.brain_endpoint[0].clone(),
                    reason: format!("--brain-ca-file {}: {reason}", ca_file.display()),
                }
            })?;
        }
        let Some(key_file) = &self.brain_key_file else {
            return Err(PolicyError::Endpoint {
                url: self.brain_endpoint[0].clone(),
                reason: "--brain-endpoint needs --brain-key-file".into(),
            });
        };
        let key = BrainEndpoints::read_key(key_file)?;
        let endpoints = BrainEndpoints::new(self.brain_endpoint.clone(), key)?;
        config.brain = Some(endpoints.with_train_span(self.train_span()?));
        Ok(())
    }

    /// How a training run spreads over the given endpoints' cards.
    fn train_span(&self) -> Result<TrainSpan, PolicyError> {
        let refused = |reason: String| PolicyError::Endpoint {
            url: self.brain_endpoint[0].clone(),
            reason,
        };
        match self.brain_train_cards.as_slice() {
            [] | [1] => Ok(TrainSpan::OneCard),
            [cards] if self.brain_endpoint.len() == 1 => Ok(TrainSpan::Cards(*cards)),
            cards if cards.len() == self.brain_endpoint.len() => {
                let (Some(secret), Some(coordinator)) =
                    (&self.brain_train_job_secret, &self.brain_train_coordinator)
                else {
                    return Err(refused(
                        "a training job over several endpoints needs --brain-train-job-secret \
                         and --brain-train-coordinator, as their servers were started"
                            .into(),
                    ));
                };
                Ok(TrainSpan::Servers(ServersJob {
                    cards: cards.to_vec(),
                    coordinator: coordinator.clone(),
                    secret_sha256: crate::model::service::secret_sha256(secret)?,
                }))
            }
            cards => Err(refused(format!(
                "--brain-train-cards names {} counts for {} endpoints",
                cards.len(),
                self.brain_endpoint.len()
            ))),
        }
    }

    /// States the flags as the process's brain settings, for a program that
    /// builds its [`Config`] in several places: every [`Config::from_env`]
    /// after carries them.
    ///
    /// # Errors
    /// The flags are refused ([`Self::apply_to`]), or were already stated.
    pub fn apply(&self) -> Result<(), PolicyError> {
        let mut config = Config::from_env();
        self.apply_to(&mut config)?;
        crate::set_brain_defaults(crate::BrainDefaults {
            device: config.brain_device,
            backend: config.brain_backend,
            max_batch: config.brain_max_batch,
            endpoints: config.brain,
            binary: self.brain_bin.clone(),
            cluster: config.cluster,
        })
        .map_err(|e| PolicyError::Endpoint {
            url: String::new(),
            reason: e.into(),
        })
    }

    /// [`Self::connection`] for a program that keeps no [`Config`] of its own:
    /// the machine's defaults (`brain` on `PATH`, brain's model store) with
    /// these flags applied.
    ///
    /// # Errors
    /// The flags are refused ([`Self::apply_to`]) or no server can be reached
    /// or started ([`Self::connection`]).
    pub fn brain(&self) -> Result<BrainConnection, PolicyError> {
        let mut config = Config::from_env();
        self.apply_to(&mut config)?;
        self.connection(&config)
    }

    /// The brain a program that asks a model directly requests it from: the
    /// endpoints `config` names, else servers started for the program's
    /// lifetime (one on every usable node of an allocation that spans nodes) ([`Runtime::brain`](crate::Runtime::brain) does the same for
    /// a program that runs on a runtime).
    ///
    /// # Errors
    /// No endpoint is named and no server can be started.
    pub fn connection(&self, config: &Config) -> Result<BrainConnection, PolicyError> {
        crate::brain_service::start_brain(config, &crate::agent::CancelToken::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(text: &str) -> Vec<String> {
        text.split_whitespace().map(String::from).collect()
    }

    #[test]
    fn the_flags_are_taken_out_wherever_they_stand() {
        let mut args = words("train --device gpu1 --out dir --backend cuda");
        let flags = DeviceFlags::take(&mut args).unwrap();
        assert_eq!(flags.device.as_deref(), Some("gpu1"));
        assert_eq!(flags.backend.as_deref(), Some("cuda"));
        assert_eq!(args, words("train --out dir"));
    }

    #[test]
    fn a_nodes_cards_are_named_by_index_and_reach_the_cluster_settings() {
        let mut args = words("serve --cluster-node-cards n80=0,1 --cluster-node-cards n105=3");
        let flags = DeviceFlags::take(&mut args).unwrap();
        assert_eq!(args, words("serve"));
        let cluster = flags.cluster_settings(ClusterSettings::default());
        assert_eq!(cluster.node_cards["n80"], [0, 1]);
        assert_eq!(cluster.node_cards["n105"], [3]);
        for bad in ["n80", "=0", "n80=", "n80=gpu0", "n80=0,,1"] {
            let mut args = words(&format!("serve --cluster-node-cards {bad}"));
            assert!(DeviceFlags::take(&mut args).is_err(), "{bad}");
        }
    }

    #[test]
    fn the_vram_budget_is_taken_as_gib_and_refused_when_not_a_positive_number() {
        let mut args = words("train --vram-budget 66 --out dir");
        let flags = DeviceFlags::take(&mut args).unwrap();
        assert_eq!(flags.vram_budget, Some(66.0));
        assert_eq!(args, words("train --out dir"));
        for bad in ["0", "-3", "lots", "NaN"] {
            let mut args = words(&format!("train --vram-budget {bad}"));
            assert!(DeviceFlags::take(&mut args).is_err(), "{bad}");
        }
    }

    #[test]
    fn a_program_without_the_flags_is_left_alone() {
        let mut args = words("train --out dir");
        let flags = DeviceFlags::take(&mut args).unwrap();
        assert!(flags.device.is_none() && flags.backend.is_none() && flags.vram_budget.is_none());
        assert!(flags.brain_endpoint.is_empty() && flags.brain_bin.is_none());
        assert_eq!(args, words("train --out dir"));
    }

    #[test]
    fn the_brain_flags_are_taken_and_endpoints_may_repeat() {
        let mut args = words(
            "run --brain-endpoint http://a/v1 --brain-key-file k --brain-endpoint http://b/v1 \
             --brain-bin /bin/brain --x",
        );
        let flags = DeviceFlags::take(&mut args).unwrap();
        assert_eq!(flags.brain_endpoint, ["http://a/v1", "http://b/v1"]);
        assert_eq!(flags.brain_key_file, Some(PathBuf::from("k")));
        assert_eq!(flags.brain_bin, Some(PathBuf::from("/bin/brain")));
        assert_eq!(args, words("run --x"));
    }

    #[test]
    fn endpoints_without_a_key_file_are_refused_and_flags_land_in_the_config() {
        let mut config = Config::from_env();
        let flags = DeviceFlags {
            device: Some("gpu1".into()),
            brain_endpoint: vec!["http://127.0.0.1:1/v1".into()],
            ..DeviceFlags::default()
        };
        let refused = flags.apply_to(&mut config).unwrap_err().to_string();
        assert!(refused.contains("--brain-key-file"), "{refused}");
        assert_eq!(config.brain_device.as_deref(), Some("gpu1"));
        assert!(config.brain.is_none());
    }

    #[test]
    fn a_program_with_no_server_and_no_binary_is_told_what_to_name() {
        let mut config = Config::from_env();
        config.brain = None;
        config.brain_binary = None;
        let refused = DeviceFlags::default()
            .connection(&config)
            .err()
            .expect("refused")
            .to_string();
        assert!(
            refused.contains("--brain-endpoint") && refused.contains("--brain-bin"),
            "{refused}"
        );
    }

    #[test]
    fn the_cluster_flags_restrict_and_shape_the_detected_allocation() {
        let mut args = words(
            "run --cluster off --cluster-nodes n[442,445] --slurm-srun /s/srun \
             --slurm-step-arg --mem=64G --slurm-single-node-step-arg --network=single_node_vni \
             --cluster-exec-prefix x --cluster-insecure-plaintext --x",
        );
        let flags = DeviceFlags::take(&mut args).unwrap();
        assert_eq!(args, words("run --x"));
        assert_eq!(flags.cluster, Some(ClusterMode::Off));
        assert!(flags.cluster_insecure_plaintext);
        let flags = DeviceFlags {
            cluster_exec_prefix: Some("apptainer exec --nv /img.sif".into()),
            ..flags
        };
        let detected = ClusterSettings {
            srun: Some("/path/srun".into()),
            ..ClusterSettings::default()
        };
        let cluster = flags.cluster_settings(detected);
        assert_eq!(cluster.mode, ClusterMode::Off);
        assert_eq!(cluster.nodes.as_deref(), Some("n[442,445]"));
        assert_eq!(
            cluster.srun,
            Some(PathBuf::from("/s/srun")),
            "the flag wins"
        );
        assert_eq!(cluster.step_args, ["--mem=64G"]);
        assert_eq!(cluster.single_node_step_args, ["--network=single_node_vni"]);
        assert!(
            cluster.insecure_plaintext,
            "plaintext only when asked by name"
        );
        assert_eq!(
            cluster.exec_prefix,
            ["apptainer", "exec", "--nv", "/img.sif"]
        );
        let mut args = words("run --cluster-stage-models off");
        let off = DeviceFlags::take(&mut args).unwrap();
        assert_eq!(
            off.cluster_settings(ClusterSettings::default())
                .stage_models,
            StageModels::Off
        );
        let mut args = words("run --cluster-stage-models relative");
        assert!(DeviceFlags::take(&mut args).is_err());
        let mut args = words("run --cluster sometimes");
        assert!(DeviceFlags::take(&mut args).is_err(), "an unknown mode");
    }

    #[test]
    fn a_ca_file_needs_the_endpoints_it_vouches_for_and_a_certificate() {
        let mut config = Config::from_env();
        let stray = DeviceFlags {
            brain_ca_file: Some(PathBuf::from("/nonexistent/ca.pem")),
            ..DeviceFlags::default()
        };
        let refused = stray.apply_to(&mut config).unwrap_err().to_string();
        assert!(refused.contains("--brain-ca-file"), "{refused}");
        let dir = tempfile::tempdir().unwrap();
        let junk = dir.path().join("ca.pem");
        std::fs::write(&junk, "not a certificate").unwrap();
        let bad = DeviceFlags {
            brain_endpoint: vec!["https://10.0.0.1:1/v1".into()],
            brain_ca_file: Some(junk),
            ..DeviceFlags::default()
        };
        let refused = bad.apply_to(&mut config).unwrap_err().to_string();
        assert!(refused.contains("--brain-ca-file"), "{refused}");
    }

    #[test]
    fn given_endpoints_train_on_the_cards_and_as_the_job_they_were_started_for() {
        let dir = tempfile::tempdir().unwrap();
        let key = dir.path().join("key");
        std::fs::write(&key, "k\n").unwrap();
        let secret = dir.path().join("secret");
        std::fs::write(&secret, format!("{}\n", "0f".repeat(32))).unwrap();
        let mut args = words(&format!(
            "train --brain-endpoint http://a/v1 --brain-endpoint http://b/v1 --brain-key-file {} \
             --brain-train-cards 2,1 --brain-train-job-secret {} --brain-train-coordinator 10.0.0.1:29500",
            key.display(),
            secret.display()
        ));
        let flags = DeviceFlags::take(&mut args).unwrap();
        assert_eq!(flags.brain_train_cards, [2, 1]);
        let mut config = Config::from_env();
        flags.apply_to(&mut config).unwrap();
        let TrainSpan::Servers(job) = config.brain.unwrap().train_span().clone() else {
            panic!("one job over the two endpoints");
        };
        assert_eq!(job.cards, [2, 1]);
        assert_eq!(job.coordinator, "10.0.0.1:29500");
        assert_eq!(job.secret_sha256.len(), 64);

        let one = DeviceFlags {
            brain_endpoint: vec!["http://a/v1".into()],
            brain_key_file: Some(key.clone()),
            brain_train_cards: vec![3],
            ..DeviceFlags::default()
        };
        let mut config = Config::from_env();
        one.apply_to(&mut config).unwrap();
        assert_eq!(config.brain.unwrap().train_span(), &TrainSpan::Cards(3));

        let unkeyed = DeviceFlags {
            brain_train_job_secret: None,
            ..flags
        };
        let refused = unkeyed
            .apply_to(&mut Config::from_env())
            .unwrap_err()
            .to_string();
        assert!(refused.contains("--brain-train-job-secret"), "{refused}");
        let stray = DeviceFlags {
            brain_train_cards: vec![2],
            ..DeviceFlags::default()
        };
        let refused = stray
            .apply_to(&mut Config::from_env())
            .unwrap_err()
            .to_string();
        assert!(refused.contains("--brain-train-cards"), "{refused}");
    }

    #[test]
    fn a_flag_without_a_value_is_refused() {
        let mut args = words("train --device");
        assert!(DeviceFlags::take(&mut args).is_err());
    }
}
