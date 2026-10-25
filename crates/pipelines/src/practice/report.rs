// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that learn a codebase from its own
// tests, for its clients. If your team needs expertise in training
// code-editing agents on executable tasks drawn from a repository, you can
// procure our services by sending an email to info@swedishembedded.com.

//! What a `practice` reports, stage by stage.

use std::collections::BTreeMap;

use serde::Serialize;
use splinter_core::digest::Digest;
use splinter_knowledge::tasks::workspace::{HistoryReport, Reason};
use splinter_store::tasks::TaskSetId;

use super::exam::ExamReserved;
use crate::curriculum::corrections::Corrections;
use crate::curriculum::frontier::Frontier;
use crate::curriculum::schedule::{ScheduleReport, Yield};
use crate::curriculum::teacher::Taught;
use crate::datasets::Built;
use crate::learn::PolicyUsed;
use crate::release::Released;
use crate::solving::Solved;
use crate::train::Candidate;
use crate::verify::Verified;

/// What the snapshot stage captured.
#[derive(Clone, Debug, Serialize)]
pub struct Snapshotted {
    /// The commit that was archived.
    pub commit: String,
    /// The digest of the snapshot's archive: what every task of the run is
    /// pinned to.
    pub snapshot: Digest,
    /// The id of the git tree the archive makes.
    pub tree_id: String,
    /// Whether the revision was checked out with uncommitted changes, which
    /// the snapshot does not hold.
    pub dirty: bool,
}

/// What the profile stage found: how the tests are run, and that the
/// baseline can be trusted.
#[derive(Clone, Debug, Serialize)]
pub struct Profiled {
    /// The program and arguments that run the whole suite.
    pub test_command: Vec<String>,
    /// How many files tasks may change.
    pub files_under_test: usize,
    /// How many tests the baseline ran and passed, in both runs.
    pub tests: usize,
    /// How long the slower of the two baseline runs took, in seconds.
    pub baseline_secs: Option<f64>,
}

/// What the tasks stage admitted.
#[derive(Clone, Debug, Serialize)]
pub struct TasksMade {
    /// The tasks admitted, now in the store.
    pub admitted: usize,
    /// Of the faults considered, those not made into a task, by reason.
    pub refused: BTreeMap<Reason, usize>,
    /// The tasks earlier runs hold, which this run did not make again.
    pub held_before: usize,
    /// The set the admitted tasks are in; `None` when there are none.
    pub task_set: Option<TaskSetId>,
    /// The funnel of a history-derived generation: which commits were
    /// considered and why each was refused. Absent when the run seeded faults
    /// instead.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub history: Option<HistoryReport>,
}

/// What a `practice` run reports, stage by stage.
#[derive(Clone, Debug, Default, Serialize)]
pub struct PracticeReport {
    /// The policy the whole run used.
    pub policy: PolicyUsed,
    /// The snapshot stage.
    pub snapshot: Option<Snapshotted>,
    /// The profile stage.
    pub profile: Option<Profiled>,
    /// The tasks stage.
    pub tasks: Option<TasksMade>,
    /// The reserve stage: the exam's families, decided before any attempt.
    pub reserve: Option<ExamReserved>,
    /// The schedule stage, when a curriculum strategy other than the
    /// frontier order was asked for.
    pub schedule: Option<ScheduleReport>,
    /// The solve stage: k attempts at each task not reserved.
    pub solve: Option<Solved>,
    /// The verify stage, with the experiences the exposure audit failed.
    pub verify: Option<Verified>,
    /// The correction stage, when a teacher was named for it: the attempts
    /// continued from where they stopped, each verified by the task's own
    /// hidden checks.
    pub correction: Option<Corrections>,
    /// The teach stage, when a teacher was named: its demonstrations of the
    /// tasks never solved, graded.
    pub teach: Option<Taught>,
    /// The frontier stage: pass@k, and the tasks kept.
    pub frontier: Option<Frontier>,
    /// What the solve and teach stages cost in device time against the tasks
    /// the frontier kept.
    #[serde(rename = "yield")]
    pub yield_: Option<Yield>,
    /// The dataset stage: the agent's actions in the passing attempts.
    pub dataset: Option<Built>,
    /// The candidate trained.
    pub candidate: Option<Candidate>,
    /// The release gate on it, and the release when it passed.
    pub release: Option<Released>,
    /// Why the pipeline stopped before training or releasing, if it did.
    pub stopped: Option<String>,
}

impl PracticeReport {
    /// Whether the run got as far as it was asked: a candidate trained, and
    /// released unless the release was not asked for.
    #[must_use]
    pub fn finished(&self, release_asked: bool) -> bool {
        self.candidate.is_some()
            && (!release_asked || self.release.as_ref().is_some_and(|r| r.release.is_some()))
    }
}
