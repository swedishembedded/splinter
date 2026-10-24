// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements durable execution contracts for learning
// campaigns, for its clients. If your team needs expertise in scheduler-
// independent job specifications or multi-node run lifecycle, you can procure
// our services by sending an email to info@swedishembedded.com.

//! A job: the durable description of work a launcher carries out, and the
//! lifecycle states it is observed in.
//!
//! A campaign decides what should be learned and what evidence counts; a job is
//! the part that runs somewhere else and may outlive the process that submitted
//! it. The specification is a value - the same bytes on a laptop, under a batch
//! system, or in a test that simulates several nodes - so its digest is the
//! job's identity and resubmitting it after a controller crash names the same
//! job. Neither a scheduler nor a model is named here.
//!
//! The command is a program and its arguments, never a shell line: a launcher
//! substitutes `{rank}` and `{world}` for the process it starts and runs the
//! result directly.

use serde::{Deserialize, Serialize};

use crate::checkpoint::{JobPlan, JOB_PLAN_FILE};
use crate::digest::{digest_of, Digest};

/// The placeholders a job's command may carry, filled per worker.
pub const RANK_PLACEHOLDER: &str = "{rank}";
/// The world-size placeholder a job's command may carry.
pub const WORLD_PLACEHOLDER: &str = "{world}";

/// The option a distributed job's command carries with the plan file's path.
pub const JOB_PLAN_FLAG: &str = "--job-plan";
/// The option a distributed job's command carries with the worker's rank.
pub const JOB_RANK_FLAG: &str = "--job-rank";

/// How many workers a job runs and what each may use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerPlan {
    /// How many machines the job is placed on.
    pub nodes: u32,
    /// How many workers run on each machine.
    pub ranks_per_node: u32,
    /// How many accelerators each worker may use (`0` for none).
    pub gpus_per_rank: u32,
}

impl WorkerPlan {
    /// One worker on one machine, with the accelerators it asks for.
    #[must_use]
    pub fn single(gpus_per_rank: u32) -> WorkerPlan {
        WorkerPlan {
            nodes: 1,
            ranks_per_node: 1,
            gpus_per_rank,
        }
    }

    /// How many ranks the job has.
    #[must_use]
    pub fn world_size(&self) -> u32 {
        self.nodes.saturating_mul(self.ranks_per_node)
    }
}

/// Every rank as one rank per worker, which is what a distributed trainer
/// expects.
impl Default for WorkerPlan {
    fn default() -> Self {
        WorkerPlan::single(0)
    }
}

/// How a distributed job's ranks are split, and how long a rank may wait.
///
/// The world size comes from the [`WorkerPlan`]; the data-parallel degree is
/// the world size divided by the tensor and pipeline degrees, which must divide
/// it exactly. A job that carries a policy has its attempt directory hold the
/// plan every rank reads, and its command gains the two options that name it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanPolicy {
    /// How the ranks are split across tensor parallelism.
    pub tp: u32,
    /// How the ranks are split across pipeline parallelism.
    pub pp: u32,
    /// How long a rank waits for the others to register, in milliseconds.
    pub handshake_timeout_ms: u64,
    /// How long one collective operation may take, in milliseconds.
    pub op_timeout_ms: u64,
}

impl Default for PlanPolicy {
    fn default() -> Self {
        PlanPolicy {
            tp: 1,
            pp: 1,
            handshake_timeout_ms: 120_000,
            op_timeout_ms: 600_000,
        }
    }
}

impl PlanPolicy {
    /// The data-parallel degree `world_size` ranks split into under this
    /// policy, or why they cannot be split.
    pub fn data_parallel(&self, world_size: usize) -> Result<u32, PlanPolicyError> {
        if self.tp == 0 || self.pp == 0 {
            return Err(PlanPolicyError::ZeroParallelism);
        }
        let group = u64::from(self.tp) * u64::from(self.pp);
        let world = u64::try_from(world_size).map_err(|_| PlanPolicyError::NotDivisible {
            world_size,
            tp: self.tp,
            pp: self.pp,
        })?;
        if world == 0 || world % group != 0 {
            return Err(PlanPolicyError::NotDivisible {
                world_size,
                tp: self.tp,
                pp: self.pp,
            });
        }
        u32::try_from(world / group).map_err(|_| PlanPolicyError::NotDivisible {
            world_size,
            tp: self.tp,
            pp: self.pp,
        })
    }

    /// The plan for a job named `job` with `world_size` ranks, registering with
    /// `coordinator`.
    pub fn plan(
        &self,
        job: impl Into<String>,
        world_size: usize,
        coordinator: impl Into<String>,
    ) -> Result<JobPlan, PlanPolicyError> {
        Ok(JobPlan {
            job: job.into(),
            world_size,
            tp: self.tp,
            pp: self.pp,
            dp: self.data_parallel(world_size)?,
            coordinator: coordinator.into(),
            handshake_timeout_ms: self.handshake_timeout_ms,
            op_timeout_ms: self.op_timeout_ms,
        })
    }
}

/// Why a plan does not fit the workers it would run on.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PlanPolicyError {
    /// Tensor or pipeline parallelism of zero names no split at all.
    #[error("a plan names zero tensor or pipeline parallelism")]
    ZeroParallelism,
    /// The world size is not the product of the parallelism degrees and the
    /// data-parallel degree.
    #[error("{world_size} ranks cannot be split across tensor parallelism {tp} and pipeline parallelism {pp}")]
    NotDivisible {
        /// How many ranks the job has.
        world_size: usize,
        /// The tensor-parallel degree.
        tp: u32,
        /// The pipeline-parallel degree.
        pp: u32,
    },
}

/// What a launcher may do with a job that fails.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetryPolicy {
    /// Attempts in all, including the first; never zero.
    pub attempts: u32,
    /// Whether a preempted attempt counts against `attempts`. A scheduler that
    /// takes a job back did not fail it.
    pub retry_on_preemption: bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        RetryPolicy {
            attempts: 1,
            retry_on_preemption: true,
        }
    }
}

/// A job as it is submitted: what runs, where it may run and what it reads and
/// writes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JobSpec {
    /// The program and its arguments; `{rank}` and `{world}` are substituted
    /// per worker. Never a shell line and never empty.
    pub command: Vec<String>,
    /// How many workers run and what each uses.
    pub workers: WorkerPlan,
    /// The artifacts the job reads, by content address.
    pub inputs: Vec<Digest>,
    /// The artifact roles the job may produce.
    pub outputs: Vec<String>,
    /// The job's wall clock, in seconds; `0` for no bound the launcher enforces.
    pub wall_clock_seconds: u64,
    /// What may be retried.
    pub retry: RetryPolicy,
    /// How a distributed job's ranks are split and how long they may wait, when
    /// the launcher writes a plan for them; `None` for a job that needs none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan: Option<PlanPolicy>,
}

impl JobSpec {
    /// A single-worker job running `command`.
    #[must_use]
    pub fn single(command: Vec<String>) -> JobSpec {
        JobSpec {
            command,
            workers: WorkerPlan::default(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            wall_clock_seconds: 0,
            retry: RetryPolicy::default(),
            plan: None,
        }
    }

    /// This job launched under a generated plan: every worker gains
    /// `--job-plan <file>` and `--job-rank <rank>`, and the launcher writes the
    /// plan the file names into each attempt directory before any worker starts.
    #[must_use]
    pub fn with_plan(mut self, plan: PlanPolicy) -> JobSpec {
        self.command.extend([
            JOB_PLAN_FLAG.to_owned(),
            JOB_PLAN_FILE.to_owned(),
            JOB_RANK_FLAG.to_owned(),
            RANK_PLACEHOLDER.to_owned(),
        ]);
        self.plan = Some(plan);
        self
    }

    /// The job's identity: the BLAKE3 address of its canonical form. The same
    /// specification submitted twice is the same job.
    ///
    #[must_use]
    pub fn digest(&self) -> Digest {
        digest_of(self)
    }

    /// What every rank relies on: a program to run and at least one worker.
    pub fn validate(&self) -> Result<(), JobSpecError> {
        if self.command.is_empty() || self.command[0].trim().is_empty() {
            return Err(JobSpecError::EmptyCommand);
        }
        if self.workers.nodes == 0 || self.workers.ranks_per_node == 0 {
            return Err(JobSpecError::NoWorkers);
        }
        if self.retry.attempts == 0 {
            return Err(JobSpecError::NoAttempts);
        }
        if let Some(plan) = &self.plan {
            plan.data_parallel(self.workers.world_size() as usize)?;
        }
        Ok(())
    }

    /// The command this worker runs: the placeholders filled with `rank` and
    /// the job's world size.
    #[must_use]
    pub fn command_for(&self, rank: u32) -> Vec<String> {
        let world = self.workers.world_size();
        self.command
            .iter()
            .map(|part| {
                part.replace(RANK_PLACEHOLDER, &rank.to_string())
                    .replace(WORLD_PLACEHOLDER, &world.to_string())
            })
            .collect()
    }
}

/// Why a job specification is refused.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum JobSpecError {
    /// There is nothing to run.
    #[error("a job names no program")]
    EmptyCommand,
    /// There is nowhere to run it.
    #[error("a job has no workers")]
    NoWorkers,
    /// A job that may not even be attempted once.
    #[error("a job allows no attempt")]
    NoAttempts,
    /// The job's plan does not fit its workers.
    #[error(transparent)]
    Plan(#[from] PlanPolicyError),
}

/// Where a job stands. Every observed state is attributable to one attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    /// Accepted by the controller, not yet placed.
    Pending,
    /// A scheduler has placed it and its workers may start.
    Admitted,
    /// Its workers are running.
    Running,
    /// A worker is committing a checkpoint.
    Checkpointing,
    /// Finished with everything it set out to do.
    Completed,
    /// Stopped by an error.
    Failed,
    /// A failed (or preempted) attempt will be tried again.
    Retrying,
    /// Stopped by a cancel request.
    Cancelled,
    /// Its workers are gone and no scheduler holds it: a controller restart or
    /// a lost allocation, to be reconciled.
    Lost,
    /// The scheduler took its resources back.
    Preempted,
}

impl JobState {
    /// Whether nothing more will happen to the job in this state.
    #[must_use]
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            JobState::Completed | JobState::Failed | JobState::Cancelled
        )
    }

    /// Whether a job in this state might still run, so a controller that
    /// restarted must reconcile it.
    #[must_use]
    pub fn needs_reconciliation(self) -> bool {
        matches!(
            self,
            JobState::Pending
                | JobState::Admitted
                | JobState::Running
                | JobState::Checkpointing
                | JobState::Retrying
                | JobState::Lost
                | JobState::Preempted
        )
    }

    /// The state as it is serialized.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            JobState::Pending => "pending",
            JobState::Admitted => "admitted",
            JobState::Running => "running",
            JobState::Checkpointing => "checkpointing",
            JobState::Completed => "completed",
            JobState::Failed => "failed",
            JobState::Retrying => "retrying",
            JobState::Cancelled => "cancelled",
            JobState::Lost => "lost",
            JobState::Preempted => "preempted",
        }
    }
}

/// What a launcher reports about a submitted job now.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JobObservation {
    /// The state the launcher observed.
    pub state: JobState,
    /// The attempt this observation is about, counting from one.
    pub attempt: u32,
    /// The process's exit code, when the launcher saw it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
    /// What happened, when the launcher can say.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> JobSpec {
        JobSpec {
            command: vec!["brain".into(), "train".into(), "--rank={rank}".into()],
            ..JobSpec::single(vec![])
        }
    }

    #[test]
    fn a_specification_without_a_program_or_workers_is_refused() {
        let mut job = spec();
        job.command.clear();
        assert_eq!(job.validate(), Err(JobSpecError::EmptyCommand));
        let mut job = spec();
        job.workers.nodes = 0;
        assert_eq!(job.validate(), Err(JobSpecError::NoWorkers));
    }

    #[test]
    fn the_digest_is_the_identity_of_the_specification() {
        let a = spec();
        let b = spec();
        assert_eq!(a.digest(), b.digest());
        let mut changed = spec();
        changed.command.push("--steps".into());
        assert_ne!(a.digest(), changed.digest());
    }

    #[test]
    fn placeholders_are_filled_per_rank() {
        let mut job = spec();
        job.workers = WorkerPlan {
            nodes: 2,
            ranks_per_node: 2,
            gpus_per_rank: 1,
        };
        assert_eq!(job.workers.world_size(), 4);
        assert_eq!(job.command_for(2), vec!["brain", "train", "--rank=2"]);
        let mut job = spec();
        job.command.push("--world={world}".into());
        assert_eq!(job.command_for(1).last().unwrap(), "--world=1");
    }

    #[test]
    fn terminal_and_reconcilable_states_do_not_overlap() {
        for state in [
            JobState::Pending,
            JobState::Admitted,
            JobState::Running,
            JobState::Checkpointing,
            JobState::Retrying,
            JobState::Lost,
            JobState::Preempted,
        ] {
            assert!(state.needs_reconciliation(), "{state:?}");
            assert!(!state.is_terminal(), "{state:?}");
        }
        for state in [JobState::Completed, JobState::Failed, JobState::Cancelled] {
            assert!(state.is_terminal(), "{state:?}");
            assert!(!state.needs_reconciliation(), "{state:?}");
        }
    }

    fn distributed() -> JobSpec {
        let mut job = JobSpec::single(vec!["brain".into(), "qwen3".into(), "finetune".into()])
            .with_plan(PlanPolicy::default());
        job.workers = WorkerPlan {
            nodes: 1,
            ranks_per_node: 2,
            gpus_per_rank: 1,
        };
        job
    }

    #[test]
    fn a_distributed_job_appends_the_plan_and_its_rank_to_every_worker() {
        let job = distributed();
        assert_eq!(job.plan, Some(PlanPolicy::default()));
        let command = job.command_for(1);
        assert_eq!(
            command[command.len() - 4..],
            [JOB_PLAN_FLAG, JOB_PLAN_FILE, JOB_RANK_FLAG, "1"]
        );
        let first = job.command_for(0);
        assert_eq!(first.last().unwrap(), "0");
        assert_eq!(
            job.digest().hex().len(),
            64,
            "the identity a plan names is the digest in hex"
        );
    }

    #[test]
    fn the_plan_names_the_jobs_identity_and_splits_its_ranks() {
        let job = distributed();
        let plan = job
            .plan
            .unwrap()
            .plan(
                job.digest().hex(),
                job.workers.world_size() as usize,
                "127.0.0.1:7000",
            )
            .unwrap();
        assert_eq!(plan.job, job.digest().hex());
        assert_eq!(plan.world_size, 2);
        assert_eq!((plan.tp, plan.pp, plan.dp), (1, 1, 2));
        assert_eq!(
            (plan.handshake_timeout_ms, plan.op_timeout_ms),
            (120_000, 600_000)
        );
    }

    #[test]
    fn a_plan_that_does_not_divide_the_ranks_is_refused() {
        let policy = PlanPolicy {
            tp: 2,
            pp: 2,
            ..PlanPolicy::default()
        };
        let mut job = JobSpec::single(vec!["brain".into()]).with_plan(policy);
        job.workers = WorkerPlan {
            nodes: 1,
            ranks_per_node: 2,
            gpus_per_rank: 0,
        };
        assert_eq!(
            job.validate(),
            Err(JobSpecError::Plan(PlanPolicyError::NotDivisible {
                world_size: 2,
                tp: 2,
                pp: 2,
            }))
        );
        assert_eq!(
            PlanPolicy {
                tp: 0,
                ..PlanPolicy::default()
            }
            .data_parallel(2),
            Err(PlanPolicyError::ZeroParallelism)
        );
        assert_eq!(policy.plan("job", 4, "127.0.0.1:7000").unwrap().dp, 1);
    }

    #[test]
    fn a_job_without_a_plan_serialises_without_the_field() {
        let job = JobSpec::single(vec!["/bin/true".into()]);
        let value = serde_json::to_value(&job).unwrap();
        assert!(value.get("plan").is_none(), "{value}");
        let back: JobSpec = serde_json::from_value(value).unwrap();
        assert_eq!(back, job);
        assert_eq!(back.digest(), job.digest());
    }
}
