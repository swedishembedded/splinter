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
use std::path::PathBuf;

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
    let steps = node_steps(config)?;
    let stage_models = stage_dir(config, &steps, &cluster.nodes)?;
    let fleet = FleetSettings {
        service,
        steps,
        nodes: cluster.nodes,
        gpus_per_node: cluster.allocation.gpus_per_node,
        node_cards: config.cluster.node_cards.clone(),
        insecure_plaintext: config.cluster.insecure_plaintext,
        stage_models,
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
    let cards: Vec<String> = fleet
        .nodes
        .iter()
        .zip(fleet.server_cards())
        .map(|(node, cards)| format!("{node} {cards} card(s)"))
        .collect();
    eprintln!(
        "brain: serving on {} ({transport}); endpoints listed in {}",
        cards.join(", "),
        work.join(splinter_model::service::ENDPOINTS_FILE).display()
    );
    Ok(BrainConnection::Owned(managed))
}

/// Where brain stages the model store on `nodes`, checked there: the
/// directory the configuration names, or the one under the scratch the site
/// announces (`--cluster-stage-models auto`), once a step on every node,
/// under the exec prefix, made or found it writable. A named directory a
/// node cannot write fails the run; an announced one is dropped, and the
/// reason is said, so the servers and ranks read the shared store as they
/// would without one.
///
/// # Errors
/// A named directory one of `nodes` cannot write.
pub fn stage_dir(
    config: &Config,
    steps: &NodeSteps,
    nodes: &[String],
) -> Result<Option<PathBuf>, PolicyError> {
    let staging = config.cluster.staging();
    let Some(dir) = staging.dir else {
        return Ok(None);
    };
    let work = config.state_root.brain_service();
    std::fs::create_dir_all(&work).map_err(|source| PolicyError::Io {
        path: work.clone(),
        source,
    })?;
    match steps.check_writable(nodes, &dir, &work) {
        Ok(()) => {
            eprintln!(
                "brain: staging models on each node under {} ({})",
                dir.display(),
                staging.source
            );
            Ok(Some(dir))
        }
        Err(why) if staging.named => Err(PolicyError::Endpoint {
            url: String::new(),
            reason: format!("--cluster-stage-models {}: {why}", dir.display()),
        }),
        Err(why) => {
            eprintln!(
                "brain: not staging models under {} ({}): {why}; reading the shared store",
                dir.display(),
                staging.source
            );
            Ok(None)
        }
    }
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

#[cfg(test)]
mod tests {
    use std::os::unix::fs::PermissionsExt;

    use super::*;
    use crate::cluster::StageModels;

    /// A stand-in `srun` that runs the step's script once, here.
    fn steps(dir: &std::path::Path) -> NodeSteps {
        let srun = dir.join("srun");
        std::fs::write(
            &srun,
            "#!/bin/sh\nwhile [ \"$1\" != sh ]; do shift; done\nshift\nSLURM_PROCID=0 exec sh \"$@\"\n",
        )
        .unwrap();
        std::fs::set_permissions(&srun, std::fs::Permissions::from_mode(0o755)).unwrap();
        NodeSteps {
            srun,
            ..NodeSteps::default()
        }
    }

    fn config(dir: &std::path::Path, stage: StageModels, scratch: &std::path::Path) -> Config {
        let mut config = Config::from_env();
        config.state_root = splinter_store::StateRoot::new(dir.join("state"));
        config.cluster.stage_models = stage;
        config.cluster.announced_scratch = Some(("SNIC_TMP".into(), scratch.to_path_buf()));
        config
    }

    #[test]
    fn an_announced_scratch_is_used_once_a_node_can_write_it_and_dropped_when_it_cannot() {
        let dir = tempfile::tempdir().unwrap();
        let scratch = dir.path().join("scratch");
        std::fs::create_dir(&scratch).unwrap();
        let nodes = ["here".to_string()];
        let steps = steps(dir.path());

        let staged = stage_dir(
            &config(dir.path(), StageModels::Auto, &scratch),
            &steps,
            &nodes,
        );
        assert_eq!(staged.unwrap(), Some(scratch.join("brain-models")));

        // A scratch no node can write, even as root: a file where the
        // directory was announced (permission bits do not stop root).
        std::fs::remove_dir_all(&scratch).unwrap();
        std::fs::write(&scratch, "").unwrap();
        let dropped = stage_dir(
            &config(dir.path(), StageModels::Auto, &scratch),
            &steps,
            &nodes,
        );
        let named = stage_dir(
            &config(
                dir.path(),
                StageModels::Dir(scratch.join("named")),
                &scratch,
            ),
            &steps,
            &nodes,
        );
        assert_eq!(
            dropped.unwrap(),
            None,
            "an announced scratch a node cannot write is not used"
        );
        let refused = named.unwrap_err().to_string();
        assert!(refused.contains("cannot write"), "{refused}");
        assert_eq!(
            stage_dir(
                &config(dir.path(), StageModels::Off, &scratch),
                &steps,
                &nodes
            )
            .unwrap(),
            None
        );
    }
}
