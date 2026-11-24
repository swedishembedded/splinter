// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements a command line that turns any source of
// knowledge into executable tasks, for its clients. If your team needs
// expertise in task generation and command-line design, you can procure our
// services by sending an email to info@swedishembedded.com.

//! `tasks ...`: generate task sets from sources or a repository's history,
//! vary, combine, pick, list and show them.

use std::path::PathBuf;

use clap::Subcommand;
use splinter_sdk::knowledge::tasks::workspace::{
    DEFAULT_MAX_TASKS, DEFAULT_PER_FUNCTION, MIN_COMBINED_FAULTS,
};
use splinter_sdk::seed::Seed;
use splinter_sdk::variants::DEFAULT_VARIANTS_PER_TASK;
use splinter_sdk::vocabulary::model_ref::ModelRef;

use super::harbor::HarborCommand;
use super::{fraction, model_ref, positive_count};

/// `tasks ...`.
#[derive(Debug, Subcommand)]
pub enum TasksCommand {
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
    /// Make harder tasks of the admitted single-fault tasks of a task set:
    /// each holds several independent faults in different files of one
    /// snapshot, is graded by every member's hidden tests, and is admitted
    /// only when its reference change makes it pass (twice) and no single
    /// member's does. A task with a fault in a file the exam reserved is an
    /// exam task. The choice is a function of the seed and the tasks alone.
    /// They go into task sets of their own: all of them, the exam's, and
    /// the rest.
    Combine {
        /// The task set of single-fault tasks, by id or unique prefix.
        #[arg(value_name = "TASKSET-ID")]
        task_set: String,
        /// Faults in each task, in as many different files.
        #[arg(long, value_name = "K", default_value_t = MIN_COMBINED_FAULTS,
            value_parser = fault_count)]
        faults: usize,
        /// The most combined tasks any one file takes part in.
        #[arg(long, value_name = "N", value_parser = positive_count)]
        per_family: usize,
        /// The most combined tasks made in all.
        #[arg(long, value_name = "N", default_value_t = DEFAULT_MAX_TASKS,
            value_parser = positive_count)]
        max_tasks: usize,
        /// The seed that decides which tasks are combined (drawn and printed
        /// when not given).
        #[arg(long, value_name = "N")]
        seed: Option<Seed>,
    },
    /// Make comprehension drills from a repository's clean snapshot: small
    /// closed-book tasks that show a program calling one function of the
    /// repository on inputs its own tests use, and ask what it prints. The
    /// reference is what the interpreter prints against the snapshot (never a
    /// task's broken tree, never a hidden test); a candidate is admitted only
    /// when it runs, prints something informative and prints the same twice.
    /// A drill is tagged with its file and function, and one of a file the
    /// practice exam reserved is an exam task. They go into task sets of their
    /// own: all of them, the exam's, and the rest.
    Drills {
        /// The git repository: a work tree or a bare repository.
        #[arg(value_name = "REPO")]
        repository: PathBuf,
        /// The revision whose snapshot is drilled: anything git resolves to a
        /// commit. The tracked files of that commit are the snapshot;
        /// uncommitted work is not in it.
        #[arg(long, default_value = "HEAD", value_name = "REV")]
        rev: String,
        /// Snapshot the committed files of a revision that is checked out with
        /// uncommitted changes, instead of refusing it.
        #[arg(long)]
        allow_dirty: bool,
        /// The most drills of any one function.
        #[arg(long, value_name = "N", default_value_t = DEFAULT_PER_FUNCTION,
            value_parser = positive_count)]
        per_function: usize,
        /// The seed that chooses among the inputs the tests give (drawn and
        /// printed when not given).
        #[arg(long, value_name = "N")]
        seed: Option<Seed>,
        /// A source file (`.py` or `.lua`), relative to the repository, to
        /// drill; repeat for more. Every source file that is not a test
        /// without any.
        #[arg(long = "file", value_name = "PATH")]
        files: Vec<String>,
        /// Drill only the functions that contain a line the reference
        /// changes of this task set's workspace tasks change (the tasks a
        /// policy fails or solves only sometimes), so that it can be trained
        /// on the drills before it attempts them again. The revision must be
        /// the one the tasks were made from.
        #[arg(long, value_name = "TASKSET-ID")]
        for_tasks: Option<String>,
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
    /// Drop the tasks whose request does not state the contract their
    /// hidden check enforces (a returned value or a raised error that no
    /// failing test's title speaks of, so the request cannot lead to the
    /// fix). The kept tasks, unchanged, are written as a task set of their
    /// own; the dropped are counted by what they leave unstated. The same
    /// rule refuses such tasks when they are generated.
    CheckRequests {
        /// The task set, by id or unique prefix.
        #[arg(value_name = "TASKSET-ID")]
        task_set: String,
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

/// A number of faults for a combined task: at least [`MIN_COMBINED_FAULTS`].
fn fault_count(text: &str) -> Result<usize, String> {
    let count: usize = text
        .parse()
        .map_err(|_| format!("{text:?} is not a number"))?;
    if count < MIN_COMBINED_FAULTS {
        return Err(format!(
            "a combined task holds at least {MIN_COMBINED_FAULTS} faults, not {count}"
        ));
    }
    Ok(count)
}
