// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements executable evaluation of code changes
// against a project's own test suite, for its clients. If your team needs
// expertise in test-driven benchmarks for code-editing agents, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Tasks of several independent faults, made of admitted single-fault tasks
//! of one snapshot.
//!
//! [`WorkspaceGenerator::combine`] takes admitted tasks whose faults lie in
//! different files and makes harder tasks of them:
//!
//! * the tree is the clean snapshot with every fault applied and none of the
//!   hidden tests;
//! * the check is the union of the members' fail-to-pass and pass-to-pass
//!   commands over the union of their hidden files;
//! * the reference change is the union of the members';
//! * the request states every symptom of every member, says that several
//!   independent faults cause them, and names no file;
//! * the family is the sorted member families joined
//!   ([`splinter_core::family`]), so a split by family sees which files the
//!   task draws on, and the members are recorded on the result.
//!
//! A combination is admitted only when it is shown to be one (see `prove`);
//! everything else is counted under a [`CombineReason`]. A single task that
//! cannot be combined at all is counted under `unusable` once, however many
//! combinations it would have been in. Nothing is drawn at random: the same
//! seed and tasks give the same combined tasks, whatever order the tasks
//! are named in.
//!
//! The pass-to-pass commands are the members', and each passed on its own
//! snapshot; on the combined tree the other members' faults may break them,
//! which the check does not require them not to: they must pass after the
//! change.

mod build;
mod prove;
mod request;
mod select;
mod single;

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use splinter_core::clock::Clock;
use splinter_core::digest::Digest;
use splinter_core::experience::Task;
use splinter_core::rng::Seed;

use self::select::Chooser;
use self::single::Single;
use super::{WorkspaceError, WorkspaceGenerator, WorkspaceTask};

/// The generator's name in provenance.
pub const COMBINED_GENERATOR: &str = "splinter-knowledge/combined-workspace-tasks@1";

/// The fewest faults a combined task holds: one is a single task.
pub const MIN_COMBINED_FAULTS: usize = 2;

/// How many combinations are tried for each task the plan asks for, however
/// many of them are refused.
pub const ATTEMPTS_PER_TASK: usize = 4;

/// Why a single task cannot be combined, or a combination was not admitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CombineReason {
    /// The task is not a change to a source tree.
    NotAWorkspaceTask,
    /// The task carries no usable workspace check.
    NoCheck,
    /// The task carries no reference change, or one that changes no file.
    NoReference,
    /// The task declares no family, or one whose name cannot be told from a
    /// composite one.
    NoFamily,
    /// The task's environment names no tree.
    NoSnapshot,
    /// The task's request is not worded so that it can be combined with
    /// others (only the original wording is).
    UnsupportedWording,
    /// The faults' reference changes change the same file.
    SharedFiles,
    /// The members' clean snapshots differ: they are tasks of different
    /// trees.
    DifferentSnapshots,
    /// A member's tree cannot be put back to the clean snapshot by its own
    /// reference change.
    NotRestorable,
    /// Two members cut different cases out of one test file, so no tree is
    /// without both.
    HiddenOverlap,
    /// The combined request would quote a line a fault changed.
    LeaksChange,
    /// The combined request quotes or names the hidden tests.
    NotSelfContained,
    /// A fail-to-pass command passes on the combined tree: one fault hides
    /// another.
    NotFailing,
    /// The reference change makes the check pass on one run and not on the
    /// next.
    Flaky,
    /// One member's reference change alone makes the check pass.
    SingleSuffices,
}

/// What bounds a combination.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CombinePlan {
    /// Faults in one task: at least [`MIN_COMBINED_FAULTS`].
    pub faults: usize,
    /// The most admitted tasks any one file takes part in.
    pub per_family: usize,
    /// The most tasks admitted in all.
    pub max_tasks: usize,
    /// What the choice of tasks to combine is a function of.
    pub seed: Seed,
}

impl CombinePlan {
    fn validate(&self) -> Result<(), WorkspaceError> {
        let bad = |name, reason: &str| {
            Err(WorkspaceError::Parameter {
                name,
                reason: reason.into(),
            })
        };
        if self.faults < MIN_COMBINED_FAULTS {
            return bad("faults", "a task of several faults holds at least two");
        }
        if self.per_family == 0 || self.max_tasks == 0 {
            return bad(
                "per_family",
                "at least one task per file, and one task in all",
            );
        }
        Ok(())
    }

    /// The most tasks the plan can admit among `families` files.
    fn target(&self, families: usize) -> usize {
        self.max_tasks.min(families * self.per_family / self.faults)
    }
}

/// A combined task and the tasks it was made of.
#[derive(Clone, Debug)]
pub struct Combined {
    /// The task, its family the sorted member families joined.
    pub task: WorkspaceTask,
    /// The addresses of the single tasks it was made of, in family order.
    pub members: Vec<Digest>,
}

/// What a combination admitted and what it refused.
#[derive(Clone, Debug, Default)]
pub struct Combination {
    /// The tasks admitted, in the order they were made.
    pub admitted: Vec<Combined>,
    /// The combinations tried and not admitted, counted by reason.
    pub refused: BTreeMap<CombineReason, usize>,
    /// The single tasks that cannot be combined, counted by reason.
    pub unusable: BTreeMap<CombineReason, usize>,
    /// The combinations tried.
    pub attempted: usize,
}

/// Why a combination stopped: it is refused for a reason, or the work
/// itself failed.
pub(super) enum Stop {
    Refuse(CombineReason),
    Fail(WorkspaceError),
}

impl<E: Into<WorkspaceError>> From<E> for Stop {
    fn from(error: E) -> Self {
        Self::Fail(error.into())
    }
}

impl WorkspaceGenerator {
    /// Combined tasks of `tasks`, admitted single-fault tasks of one snapshot
    /// (see the module documentation), as many as `plan` allows: every file
    /// takes part in at most `plan.per_family` of them, and files and
    /// single tasks are drawn so that each file, and each pair of files, is
    /// met about as often as any other. `clock` stamps the throwaway
    /// experiences the reference changes are verified through.
    ///
    /// # Errors
    ///
    /// [`WorkspaceError::Parameter`] when the plan is out of range or the
    /// usable tasks lie in fewer files than the plan has faults.
    pub fn combine(
        &self,
        tasks: &[Task],
        plan: &CombinePlan,
        clock: &dyn Clock,
    ) -> Result<Combination, WorkspaceError> {
        plan.validate()?;
        let mut ordered: Vec<&Task> = tasks.iter().collect();
        ordered.sort_by(|a, b| a.task.id.cmp(&b.task.id));
        ordered.dedup_by(|a, b| a.task.id == b.task.id);
        let mut combination = Combination::default();
        let mut singles = Vec::new();
        for task in ordered {
            match Single::read(task) {
                Ok(single) => singles.push(single),
                Err(reason) => *combination.unusable.entry(reason).or_default() += 1,
            }
        }
        let files: BTreeSet<&str> = singles.iter().map(|s| s.family.as_str()).collect();
        if files.len() < plan.faults {
            return Err(WorkspaceError::Parameter {
                name: "faults",
                reason: format!(
                    "the tasks lie in {} families; {} faults in different files need at least \
                     that many families",
                    files.len(),
                    plan.faults
                ),
            });
        }

        let mut chooser = Chooser::new(&singles, plan.faults, plan.per_family, plan.seed);
        let target = plan.target(chooser.families());
        let attempts = target * ATTEMPTS_PER_TASK;
        while combination.admitted.len() < target && combination.attempted < attempts {
            let Some(tuple) = chooser.next() else { break };
            combination.attempted += 1;
            let members: Vec<&Single> = tuple.iter().map(|at| &singles[*at]).collect();
            match self.attempt(&members, clock) {
                Ok(combined) => {
                    chooser.admitted(&tuple);
                    combination.admitted.push(combined);
                }
                Err(Stop::Refuse(reason)) => *combination.refused.entry(reason).or_default() += 1,
                Err(Stop::Fail(error)) => return Err(error),
            }
        }
        Ok(combination)
    }
}
