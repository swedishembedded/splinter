// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements distributed training jobs that run their
// ranks across the nodes of a batch allocation as one scheduler step, for
// its clients. If your team needs expertise in multi-node training launch
// on Slurm, you can procure our services by sending an email to
// info@swedishembedded.com.

//! A [`JobLauncher`] that runs every rank of a job as a task of ONE step of
//! the Slurm allocation this process runs inside.
//!
//! One step, because a fabric that isolates steps (a Slingshot VNI per step)
//! lets only the tasks of one step talk to each other over it. The job's
//! nodes are the first `nodes` of the allocation's usable nodes, each running
//! `ranks_per_node` tasks with `ranks_per_node x gpus_per_rank` GPUs; the
//! scheduler's task variables give each task its rank (`{rank}`), its place
//! on its node (`{local_rank}`, the card a one-card rank takes) and the world
//! (`{world}`). A distributed job's plan registers the ranks with a
//! coordinator on rank 0's node, the first node, and every rank reads the
//! same owner-only secret on the shared filesystem.
//!
//! The `srun` runs on this machine under the configured supervisor, so the
//! step's exit (the first failing task's) is recorded in the attempt
//! directory like a local worker's, and a restarted controller reads it.
//! The launch record names that one process; stopping it ends the step, and
//! a step that outlives `SIGTERM` is cancelled by its recorded id.

use std::fs;
use std::net::{IpAddr, ToSocketAddrs};
use std::path::Path;
use std::process::Command;

use splinter_core::job::JobSpec;

use super::{
    claim_attempt, write_launch, JobError, JobLauncher, Launch, LocalLauncher, Observation,
};
use crate::step::{read_step_id, NodeSteps, Placement};

/// The script the step runs, in the attempt directory.
const SCRIPT_FILE: &str = "job.step.sh";

/// How many ranks the step carries, beside its launch record.
const RANKS_FILE: &str = "step.ranks";

/// Runs a job's ranks as one step on the allocation's nodes.
#[derive(Debug)]
pub struct NodeStepLauncher {
    steps: NodeSteps,
    nodes: Vec<String>,
    local: LocalLauncher,
}

impl NodeStepLauncher {
    /// A launcher placing jobs on `nodes` (the allocation's usable nodes, in
    /// order) with `steps`, its `srun` run under `supervisor` when one is
    /// given.
    #[must_use]
    pub fn new(steps: NodeSteps, nodes: Vec<String>, supervisor: Option<&Path>) -> Self {
        Self {
            steps,
            nodes,
            local: supervisor.map_or_else(LocalLauncher::default, LocalLauncher::supervised),
        }
    }

    /// The address rank 0's node is reached at.
    fn rank0_address(node: &str) -> Result<IpAddr, JobError> {
        let addresses: Vec<IpAddr> = (node, 0)
            .to_socket_addrs()
            .map_err(|e| JobError::io(format!("resolving the node {node}"), e))?
            .map(|address| address.ip())
            .collect();
        addresses
            .iter()
            .find(|ip| ip.is_ipv4())
            .or_else(|| addresses.first())
            .copied()
            .ok_or_else(|| {
                JobError::io(
                    format!("resolving the node {node}"),
                    std::io::Error::new(std::io::ErrorKind::NotFound, "no address"),
                )
            })
    }
}

impl JobLauncher for NodeStepLauncher {
    fn name(&self) -> &'static str {
        "slurm-step"
    }

    fn launch(&self, spec: &JobSpec, attempt: u32, dir: &Path) -> Result<Launch, JobError> {
        let wanted = spec.workers.nodes as usize;
        if wanted > self.nodes.len() {
            return Err(JobError::Unsupported {
                action: format!(
                    "place a job on {wanted} nodes: the allocation offers {} ({})",
                    self.nodes.len(),
                    self.nodes.join(",")
                ),
            });
        }
        let nodes = self.nodes[..wanted].to_vec();
        claim_attempt(spec, attempt, dir, Self::rank0_address(&nodes[0])?)?;
        let script = dir.join(SCRIPT_FILE);
        fs::write(&script, self.steps.script(dir, &spec.command))
            .map_err(|e| JobError::io(format!("writing {}", script.display()), e))?;
        let world = spec.workers.world_size();
        fs::write(dir.join(RANKS_FILE), format!("{world}\n"))
            .map_err(|e| JobError::io(format!("writing {}", dir.display()), e))?;
        let placement = Placement {
            nodes,
            tasks_per_node: spec.workers.ranks_per_node,
            gpus_per_node: spec
                .workers
                .ranks_per_node
                .saturating_mul(spec.workers.gpus_per_rank),
        };
        let srun = self.steps.command(&placement, &script);
        // The step runs under the supervisor as one worker: its argv, with
        // the inherited GPU count the step must not see dropped by `env`.
        let mut command = vec!["env".to_string(), "-u".into(), "SLURM_GPUS".into()];
        command.push(srun.get_program().to_string_lossy().into_owned());
        command.extend(srun.get_args().map(|a| a.to_string_lossy().into_owned()));
        let mut process = self
            .local
            .worker_process(&command, dir, 0, spec.wall_clock_seconds)?;
        let child = process.spawn().map_err(|source| JobError::Spawn {
            rank: 0,
            world,
            source,
        })?;
        let launch = Launch {
            digest: spec.digest(),
            attempt,
            world_size: 1,
            pids: vec![child.id()],
            dir: dir.to_path_buf(),
        };
        let mut children = vec![child];
        if let Err(error) = write_launch(&launch) {
            for child in &mut children {
                child.kill().ok();
                child.wait().ok();
            }
            return Err(error);
        }
        self.local.adopt(dir, children);
        Ok(launch)
    }

    fn observe(&self, launch: &Launch) -> Observation {
        match self.local.observe(launch) {
            // The one `srun` alive is every rank of its step alive.
            Observation::Running { ranks_alive } if !ranks_alive.is_empty() => {
                let ranks = fs::read_to_string(launch.dir.join(RANKS_FILE))
                    .ok()
                    .and_then(|text| text.trim().parse::<u32>().ok())
                    .unwrap_or(1);
                Observation::Running {
                    ranks_alive: (0..ranks).collect(),
                }
            }
            other => other,
        }
    }

    fn stop(&self, launch: &Launch, rank: Option<u32>) -> Result<(), JobError> {
        if let Some(rank) = rank {
            return Err(JobError::Unsupported {
                action: format!("stop rank {rank} alone: the ranks are one scheduler step"),
            });
        }
        let stopped = launch.stop(None);
        // A step that outlived its `srun` is cancelled by its id.
        if let (Some(scancel), Some(step)) = (&self.steps.scancel, read_step_id(&launch.dir)) {
            let _ = Command::new(scancel).arg(step).status();
        }
        stopped
    }
}
