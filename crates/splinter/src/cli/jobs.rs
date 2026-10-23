// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements a command line for work that outlives the
// process that submitted it, for its clients. If your team needs expertise in
// detached job submission or operator interfaces for batch work, you can
// procure our services by sending an email to info@swedishembedded.com.

//! `jobs ...`: submit a program as a detached job, list the jobs, reconcile
//! them after a restart, show one and cancel one.

use clap::{Args, Subcommand};
use splinter_sdk::vocabulary::job::{JobSpec, RetryPolicy, WorkerPlan};

/// `jobs ...`; none lists the subcommands.
#[derive(Debug, Args)]
pub struct JobsArgs {
    /// What to do with the jobs.
    #[command(subcommand)]
    pub command: Option<JobsCommand>,
}

/// `jobs ...`.
#[derive(Debug, Subcommand)]
pub enum JobsCommand {
    /// Submit a program as one job and return as soon as its workers are
    /// started and detached.
    Submit(Box<SubmitArgs>),
    /// List the submitted jobs and the state each was last seen in.
    List,
    /// Ask each unfinished job's launcher what became of it and record the
    /// answer: completed, failed, or lost when nothing can account for it.
    Reconcile,
    /// Start the next attempt of every job whose last attempt failed and whose
    /// retry policy allows another.
    Retry,
    /// Show one job, its state and the specification it runs.
    Show {
        /// The job's digest, or a unique prefix of its hex.
        #[arg(value_name = "JOB")]
        id: String,
    },
    /// Stop a job's workers and record it cancelled.
    Cancel {
        /// The job's digest, or a unique prefix of its hex.
        #[arg(value_name = "JOB")]
        id: String,
    },
}

/// `jobs submit`.
#[derive(Debug, Args)]
pub struct SubmitArgs {
    /// The program and its arguments, run directly and never through a shell.
    /// `{rank}` and `{world}` are filled in for each worker.
    #[arg(
        required = true,
        num_args = 1..,
        trailing_var_arg = true,
        allow_hyphen_values = true,
        value_name = "PROGRAM ARG..."
    )]
    pub command: Vec<String>,
    /// How many machines the job is placed on.
    #[arg(long, default_value_t = 1, value_name = "N",
        value_parser = clap::value_parser!(u32).range(1..))]
    pub nodes: u32,
    /// How many workers run on each machine.
    #[arg(long, default_value_t = 1, value_name = "N",
        value_parser = clap::value_parser!(u32).range(1..))]
    pub ranks_per_node: u32,
    /// How many accelerators each worker may use (`0` for none).
    #[arg(long, default_value_t = 0, value_name = "N")]
    pub gpus_per_rank: u32,
    /// The job's wall clock in seconds; a worker that outlives it is stopped
    /// and the job fails with exit code 124. `0` for no bound.
    #[arg(long, default_value_t = 0, value_name = "SECONDS")]
    pub wall_clock: u64,
    /// Attempts in all, including the first.
    #[arg(long, default_value_t = 1, value_name = "N",
        value_parser = clap::value_parser!(u32).range(1..))]
    pub attempts: u32,
}

impl SubmitArgs {
    /// The specification the flags and the command describe.
    #[must_use]
    pub fn spec(&self) -> JobSpec {
        JobSpec {
            command: self.command.clone(),
            workers: WorkerPlan {
                nodes: self.nodes,
                ranks_per_node: self.ranks_per_node,
                gpus_per_rank: self.gpus_per_rank,
            },
            inputs: Vec::new(),
            outputs: Vec::new(),
            wall_clock_seconds: self.wall_clock,
            retry: RetryPolicy {
                attempts: self.attempts,
                ..RetryPolicy::default()
            },
        }
    }
}

/// The help of `jobs` alone, for the verb without a subcommand.
#[must_use]
pub fn help() -> String {
    use clap::CommandFactory;
    let mut root = crate::cli::Cli::command();
    match root.find_subcommand_mut("jobs") {
        Some(jobs) => jobs.render_long_help().to_string(),
        None => String::new(),
    }
}
