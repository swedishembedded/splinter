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
use splinter_sdk::vocabulary::job::{JobSpec, PlanPolicy, RetryPolicy, WorkerPlan};

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
    /// Run one `brain/train` action (`speech_sft`, `full_train`, `train`,
    /// ...) on the brain servers as one job: over every card of every
    /// server when the action takes `cards` and `job`, else on one card.
    /// Inside an allocation the servers are started on its nodes (one step,
    /// one server per node) and stopped after; `--brain-endpoint` uses
    /// running ones. Rank 0's step frames are printed as JSON lines on
    /// stdout as they come; Ctrl-C cancels the job on every server.
    RunServed(Box<RunServedArgs>),
    /// Stop a job's workers and record it cancelled.
    Cancel {
        /// The job's digest, or a unique prefix of its hex.
        #[arg(value_name = "JOB")]
        id: String,
    },
}

/// `jobs run-served`.
#[derive(Debug, Args)]
pub struct RunServedArgs {
    /// The `brain/train` action, as the servers' `GET /v1/capabilities`
    /// lists it.
    #[arg(long, value_name = "ACTION")]
    pub action: String,
    /// A file holding the action's params as one JSON object, as brain
    /// takes them; `cards` and `job` are set by the spread, and a `seed`
    /// left out is drawn and reported.
    #[arg(long, value_name = "FILE")]
    pub params: std::path::PathBuf,
    /// `PARAM=FILE`: publish FILE to the servers' artifact store and set
    /// PARAM to its digest (`dataset_digest=train.bttshrd`); repeat for more.
    #[arg(long = "artifact", value_name = "PARAM=FILE", value_parser = artifact)]
    pub artifacts: Vec<(String, std::path::PathBuf)>,
    /// Fetch the files the job publishes (a checkpoint, an adapter, a
    /// training record) into DIR by digest, each verified, with the result
    /// as `result.json`.
    #[arg(long, value_name = "DIR")]
    pub out: Option<std::path::PathBuf>,
}

/// `PARAM=FILE`.
fn artifact(text: &str) -> Result<(String, std::path::PathBuf), String> {
    match text.split_once('=') {
        Some((param, file)) if !param.is_empty() && !file.is_empty() => {
            Ok((param.to_string(), file.into()))
        }
        _ => Err(format!("{text:?} is not PARAM=FILE")),
    }
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
    /// Prepare a distributed training job: write its plan and a fresh
    /// owner-only secret into every attempt directory before launching and
    /// append `--job-plan <file>`, `--job-rank <rank>` and `--job-secret
    /// <file>` to each worker. The plan registers the ranks on the address of
    /// the machine rank 0 runs on: this machine's loopback for a local job,
    /// the first node's address for a job on the allocation's nodes.
    /// `{local_rank}` in the command is a rank's place on its node.
    #[arg(long)]
    pub distributed: bool,
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
        let spec = JobSpec {
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
            plan: None,
        };
        if self.distributed {
            return spec.with_plan(PlanPolicy::default());
        }
        spec
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
