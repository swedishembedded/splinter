// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements interchange of agent benchmark tasks between
// task formats for its clients. If your team needs expertise in importing
// containerised benchmark tasks into reproducible, verifiable task sets, you
// can procure our services by sending an email to info@swedishembedded.com.

//! `tasks harbor ...`.

use std::path::PathBuf;

use clap::{Subcommand, ValueEnum};

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
}
