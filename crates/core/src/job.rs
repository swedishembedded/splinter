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

use crate::digest::{digest_of, Digest};

/// The placeholders a job's command may carry, filled per worker.
pub const RANK_PLACEHOLDER: &str = "{rank}";
/// The world-size placeholder a job's command may carry.
pub const WORLD_PLACEHOLDER: &str = "{world}";

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
        }
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
}
