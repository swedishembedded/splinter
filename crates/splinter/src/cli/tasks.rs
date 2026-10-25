// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements a command line that turns any source of
// knowledge into executable tasks, for its clients. If your team needs
// expertise in task generation and command-line design, you can procure our
// services by sending an email to info@swedishembedded.com.

//! `tasks ...`: generate task sets from sources or a repository's history,
//! vary, seed canaries into, pick, list and show them.

use std::path::PathBuf;

use clap::Subcommand;
use splinter_sdk::canaries::CANARY_COUNT;
use splinter_sdk::seed::Seed;
use splinter_sdk::variants::DEFAULT_VARIANTS_PER_TASK;
use splinter_sdk::vocabulary::model_ref::ModelRef;

use super::harbor::HarborCommand;
use super::{fraction, model_ref, positive_count};

/// `tasks ...`.
#[derive(Debug, Subcommand)]
pub enum TasksCommand {
    /// Generate tasks from stored sources into a task set.
    Generate {
        /// The sources, by id or unique prefix.
        #[arg(required = true, value_name = "SOURCE-ID")]
        sources: Vec<String>,
        /// Task kinds, comma-separated.
        #[arg(long, required = true, value_delimiter = ',', value_name = "K,..")]
        kinds: Vec<String>,
        /// The model that writes the tasks.
        #[arg(long, value_parser = model_ref, default_value_t = ModelRef::policy_default(), value_name = "REF")]
        generator: ModelRef,
        /// Who wrote the sources, when they are one person's: the generator is
        /// told, and a question may name the author as its subject.
        #[arg(long, value_name = "NAME")]
        author: Option<String>,
        /// The seed the `denoise` tasks are built with (drawn and printed
        /// when not given and the kinds include denoise).
        #[arg(long, value_name = "N")]
        denoise_seed: Option<Seed>,
    },
    /// Read tasks from a repository's history: a commit that changes source
    /// and tests together is a task when its parent tree fails the commit's
    /// own tests and the commit's tree passes them. Formatting-only,
    /// generated, vendored, merge and test-less commits are excluded and
    /// counted.
    History {
        /// The git repository.
        #[arg(value_name = "REPOSITORY")]
        repository: PathBuf,
        /// The revision range, as `git rev-list` reads it (`A..B`); a bare
        /// revision reads all of its history.
        #[arg(long, default_value = "HEAD", value_name = "RANGE")]
        range: String,
        /// The command that runs the project's whole test suite in the root
        /// of its tree (everything after this flag); detected by rule when
        /// not given.
        #[arg(long, num_args = 1.., allow_hyphen_values = true, value_name = "PROGRAM ARG...")]
        test_command: Vec<String>,
        /// The most commits considered, newest first.
        #[arg(long, value_name = "N", value_parser = clap::value_parser!(usize))]
        max_commits: Option<usize>,
        /// Store the admitted tasks as a task set (no rollout is made).
        #[arg(long)]
        store: bool,
    },
    /// Write differently worded questions about each task of a task set,
    /// into a task set of their own: the same facts, to be measured and
    /// never trained on.
    Variants {
        /// The task set, by id or unique prefix.
        #[arg(value_name = "TASKSET-ID")]
        task_set: String,
        /// The model that writes the variants (the one that wrote the
        /// tasks).
        #[arg(long, value_parser = model_ref, default_value_t = ModelRef::policy_default(), value_name = "REF")]
        generator: ModelRef,
        /// The most variants one task is given.
        #[arg(long, default_value_t = DEFAULT_VARIANTS_PER_TASK, value_name = "N")]
        per_task: usize,
    },
    /// Make canaries of the workspace tasks of a task set: each task with its
    /// hidden assertions also required to fail, so that no honest change
    /// passes it. They go into a task set of their own, which `release
    /// --canaries` hands the gate: a candidate that passes one is refused.
    Canaries {
        /// The task set, by id or unique prefix.
        #[arg(value_name = "TASKSET-ID")]
        task_set: String,
        /// The most canaries made.
        #[arg(long, value_name = "N", default_value_t = CANARY_COUNT,
            value_parser = positive_count)]
        count: usize,
    },
    /// Divide the union of task sets into a picked set and the rest, by a
    /// stable hash of the seed and the task's family (the practice exam's
    /// ranking): a development holdout, a sample of one task per family, or
    /// one set out of several. Naming the sets in another order, or a task in
    /// two of them, changes nothing. A task that declares no family is a
    /// family of its own.
    Pick {
        /// The task sets, by id or unique prefix.
        #[arg(required = true, value_name = "TASKSET-ID")]
        task_sets: Vec<String>,
        /// Pick this fraction (0 to 1) of the families, every task of each.
        #[arg(long, value_name = "F", conflicts_with = "per_family",
            required_unless_present = "per_family", value_parser = fraction)]
        family_share: Option<f64>,
        /// Pick at most this many tasks of every family.
        #[arg(long, value_name = "N", value_parser = positive_count)]
        per_family: Option<usize>,
        /// The seed that decides what is picked.
        #[arg(long, value_name = "N")]
        seed: Seed,
    },
    /// Harbor tasks: discover, import and export.
    #[command(subcommand)]
    Harbor(HarborCommand),
    /// List the stored task sets.
    List,
    /// Show a task set's tasks, or one task.
    Show {
        /// A task set's or a task's id, or a unique prefix of it.
        id: String,
    },
}
