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
    /// Run the examiner's staged practice material (a directory written by
    /// `bench task practice`, holding practice.json and job.yaml) under the
    /// policy brain serves: the examiner's gateway forwards the main role of
    /// a profile derived from the one the material was staged under to the
    /// policy, the material is staged again under it into the run's own
    /// directory (refused unless every task digest, the draw and the
    /// collection's digests are the ones practice.json names), Harbor runs
    /// the job with these attempts and tasks, and the examiner's export of
    /// the graded trials is admitted as `experiences ingest-bench` admits
    /// it. Each trial's outcome is read from the examiner's score record:
    /// passed, failed, exposed, verifier failure, ungraded or
    /// infrastructure; only the first three have a reward. A rerun of a
    /// finished run runs nothing and admits nothing twice.
    Practice {
        /// The practice directory (the one holding practice.json).
        #[arg(value_name = "DIR")]
        dir: PathBuf,
        /// The policy: a local model the brain endpoint serves.
        #[arg(long, value_parser = model_ref, default_value_t = ModelRef::policy_default(), value_name = "REF")]
        policy: ModelRef,
        /// The examiner's checkout (`./bench` is run from it).
        #[arg(long, value_name = "DIR")]
        bench: PathBuf,
        /// The `harbor` program that runs the job (default: the checkout's
        /// .venv/bin/harbor).
        #[arg(long, value_name = "PATH")]
        harbor: Option<PathBuf>,
        /// The sven build the examiner runs the agent from (given to it as
        /// BENCH_SVEN_BUILD; without it, the environment's).
        #[arg(long, value_name = "PATH")]
        sven_build: Option<PathBuf>,
        /// The profile the policy's is derived from (default: the one the
        /// material was staged under, as its bench-run.json records it).
        #[arg(long, value_name = "FILE")]
        profile: Option<PathBuf>,
        /// Attempts of each task.
        #[arg(long, default_value = "1", value_parser = positive_count, value_name = "N")]
        attempts: usize,
        /// The seed the run is named and recorded by (drawn and printed when
        /// not given; a rerun with the same seed finds its run).
        #[arg(long, value_name = "SEED")]
        seed: Option<Seed>,
        /// Trials run at once.
        #[arg(long, default_value = "1", value_parser = positive_count, value_name = "N")]
        jobs: usize,
        /// Run only this staged task (repeatable); every task when not
        /// given.
        #[arg(long = "task", value_name = "NAME")]
        tasks: Vec<String>,
    },
}
