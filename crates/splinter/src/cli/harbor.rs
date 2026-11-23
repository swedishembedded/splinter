// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements interchange of agent benchmark tasks between
// task formats for its clients. If your team needs expertise in importing
// containerised benchmark tasks into reproducible, verifiable task sets, you
// can procure our services by sending an email to info@swedishembedded.com.

//! `tasks harbor ...`.

use std::path::PathBuf;
use std::time::Duration;

use clap::{Subcommand, ValueEnum};
use splinter_sdk::seed::Seed;
use splinter_sdk::vocabulary::model_ref::ModelRef;

use super::train::positive_count;
use super::{budget, model_ref};

/// What to do about a Harbor task's oracle solution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum OracleArg {
    /// Do not run it.
    Skip,
    /// Run it; import without a reference when it does not run.
    Try,
    /// Run it; refuse the task unless it runs and passes.
    Require,
}

/// `tasks harbor ...`.
#[derive(Debug, Subcommand)]
pub enum HarborCommand {
    /// List the Harbor tasks under a directory and whether each can be
    /// imported, at the revision of the checkout it is in. Writes nothing.
    Discover {
        /// The directory (a checkout of a task repository).
        #[arg(value_name = "DIR")]
        root: PathBuf,
    },
    /// Import Harbor tasks (each a task directory) as workspace tasks in one
    /// task set. A task the host cannot honour (a Dockerfile with RUN, a
    /// prebuilt image, a Windows or multi-step task) is refused with the
    /// reason; the rest are held to the bar of every workspace task.
    Import {
        /// The task directories.
        #[arg(required = true, value_name = "DIR")]
        directories: Vec<PathBuf>,
        /// What to do about a task's solution/solve.sh: `skip` does not run
        /// it, `try` runs it and imports without a reference when it does
        /// not run on the host, `require` refuses the task unless it runs and
        /// its change passes the task's check.
        #[arg(long, value_enum, default_value = "try")]
        oracle: OracleArg,
    },
    /// Write the workspace tasks of a task set as Harbor tasks, one directory
    /// each under --out, with everything needed to import them back exactly.
    Export {
        /// The task set, by id or unique prefix.
        #[arg(value_name = "TASKSET-ID")]
        task_set: String,
        /// The directory to write into (it must not hold the tasks yet).
        #[arg(long, required = true, value_name = "DIR")]
        out: PathBuf,
        /// The base image of each Dockerfile.
        #[arg(long, default_value = "ubuntu:24.04", value_name = "IMAGE")]
        image: String,
    },
    /// Run a directory of practice task folders (or one folder) as
    /// episodes: the policy attempts each task --attempts times, each in a
    /// fresh copy of its starting files through sven's agent, asked for the
    /// tokens and log-probabilities it samples; afterwards the task's own
    /// tests grade what it left, outside its reach. Every attempt is stored
    /// as an experience with a record of its outcome (passed, failed,
    /// exposed, verifier error or timeout, infrastructure), reward, usage and
    /// provenance. Exam material (`[metadata].bench_sealed`, a held-out
    /// tier, the eval set of the collection's split.toml) is refused before
    /// anything runs. A rerun skips attempts already recorded and redoes the
    /// ones the infrastructure stopped.
    Episodes {
        /// The directory of task folders, or one task folder.
        #[arg(value_name = "DIR")]
        root: PathBuf,
        /// The policy.
        #[arg(long, value_parser = model_ref, default_value_t = ModelRef::policy_default(), value_name = "REF")]
        policy: ModelRef,
        /// Attempts per task.
        #[arg(long, default_value = "1", value_parser = positive_count, value_name = "N")]
        attempts: usize,
        /// The seed every attempt's draws derive from (drawn and printed
        /// when not given; a rerun with the same seed finds its attempts).
        #[arg(long, value_name = "SEED")]
        seed: Option<Seed>,
        /// Attempts run at once.
        #[arg(long, default_value = "1", value_parser = positive_count, value_name = "N")]
        jobs: usize,
        /// How long one attempt's agent may run (`90s`, `10m`, ...); the
        /// solve stage's deadline when not given.
        #[arg(long, value_parser = budget, value_name = "DUR")]
        deadline: Option<Duration>,
        /// Tool calls one attempt may request.
        #[arg(long, value_name = "N")]
        max_tool_calls: Option<u32>,
        /// Output tokens one attempt may generate.
        #[arg(long, value_name = "N")]
        max_output_tokens: Option<u64>,
        /// Also write every attempt's record to this file, one JSON object
        /// per line.
        #[arg(long, value_name = "FILE")]
        out: Option<PathBuf>,
    },
}
