// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems that serve their models
// on every node a batch scheduler granted them, with no hand-wired
// endpoints, for its clients. If your team needs expertise in running
// model serving across a cluster allocation, you can procure our services
// by sending an email to info@swedishembedded.com.

//! The brain a process requests its local models from, decided once from
//! the configuration: the endpoints it was given; else, inside an allocation
//! that spans nodes, a server on every usable node (a fleet); else a server
//! on this machine, as outside any allocation.

use splinter_agent::CancelToken;
use splinter_model::{BrainConnection, FleetSettings, ManagedBrain, PolicyError, ServiceSettings};
use splinter_sandbox::step::NodeSteps;

use crate::config::Config;

/// The brain `config` says to request local models from, started when it
/// names none and stopped when the connection is dropped.
///
/// # Errors
/// No endpoint is named and no `brain` binary is known; the allocation
/// cannot be used (a bad `--cluster-nodes`, no `srun`); or a server does not
/// start, named by its node.
pub fn start_brain(config: &Config, cancel: &CancelToken) -> Result<BrainConnection, PolicyError> {
    if let Some(endpoints) = &config.brain {
        return Ok(BrainConnection::Given(endpoints.clone()));
    }
    let refused = |reason: String| PolicyError::Endpoint {
        url: String::new(),
        reason,
    };
    let binary = config.brain_binary.clone().ok_or_else(|| {
        refused(
            "a local model is requested from a brain server: name one with --brain-endpoint, \
             or make the `brain` binary known (--brain-bin, or on PATH) so one can be started"
                .into(),
        )
    })?;
    let service = ServiceSettings {
        binary,
        models: config.model_store.clone(),
        state: config.state_root.brain_service(),
        device: config.brain_device.clone(),
        backend: config.brain_backend.clone(),
        max_batch: config.brain_max_batch,
        serve_adapters: true,
        adapter_base: Some(config.policy_base.clone()).filter(|base| base.exists()),
    };
    let usable = config
        .cluster
        .usable()
        .map_err(|e| refused(format!("the cluster allocation: {e}")))?;
    let Some(cluster) = usable.filter(|cluster| cluster.spans_nodes()) else {
        return ManagedBrain::start(&service, cancel).map(BrainConnection::Owned);
    };
    let fleet = FleetSettings {
        service,
        steps: node_steps(config)?,
        nodes: cluster.nodes,
        gpus_per_node: cluster.allocation.gpus_per_node,
        insecure_plaintext: config.cluster.insecure_plaintext,
    };
    let managed = ManagedBrain::start_fleet(&fleet, cancel)?;
    let work = managed.work_dir();
    let transport = if fleet.insecure_plaintext {
        "plaintext".to_string()
    } else {
        format!(
            "TLS under {}",
            work.join(splinter_model::service::AUTHORITY_FILE).display()
        )
    };
    eprintln!(
        "brain: serving on {} ({} GPUs each, {transport}); endpoints listed in {}",
        fleet.nodes.join(","),
        fleet.gpus_per_node,
        work.join(splinter_model::service::ENDPOINTS_FILE).display()
    );
    Ok(BrainConnection::Owned(managed))
}

/// How a step is started on the allocation's nodes, as the configuration
/// says.
///
/// # Errors
/// No `srun` is known.
pub fn node_steps(config: &Config) -> Result<NodeSteps, PolicyError> {
    let cluster = &config.cluster;
    let srun = cluster.srun.clone().ok_or_else(|| PolicyError::Endpoint {
        url: String::new(),
        reason: "the allocation spans nodes, but no srun is known to start work on them: put it \
                 on PATH or name it with --slurm-srun, or run with --cluster off"
            .into(),
    })?;
    Ok(NodeSteps {
        srun,
        scancel: cluster.scancel.clone(),
        args: cluster.step_args.clone(),
        single_node_args: cluster.single_node_step_args.clone(),
        exec_prefix: cluster.exec_prefix.clone(),
    })
}
