// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements executable evaluation of code changes
// against a project's own test suite, for its clients. If your team needs
// expertise in test-driven benchmarks for code-editing agents, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Workspace tasks: a change to a source tree, graded by the project's own
//! tests, built by code from a repository. No model is asked for anything
//! here.
//!
//! * [`RepositoryProfile`] - how the tests are run, detected by rule or
//!   given, and [`RepositoryProfile::qualify`], which runs the suite twice
//!   and refuses a red or flaky baseline.
//! * [`WorkspaceGenerator::from_mutations`] - tasks seeded by faults: one
//!   small textual change to a file under test
//!   ([`splinter_eval::verifiers::mutation::mutants`]) is admitted as a task
//!   only when some tests go from passing to failing and fail on every
//!   rerun, the rest still pass, and the change reversed makes the project's
//!   own check pass under the workspace verifier. The tests that catch it
//!   are taken out of the tree the solver is given and kept as a hidden
//!   overlay; the request names their symptoms and never the change.
//! * [`WorkspaceGenerator::import_fixture`] - a task someone authored
//!   (a rendered seeded-bug repository, a family's workspace template),
//!   checked the same way before it is admitted.
//!
//! A mutation-seeded task is also measured for the strength of its hidden
//! tests: of the other faults seeded within a few lines of its own that
//! some test catches, the share the hidden tests catch is recorded on the
//! task's check ([`HiddenStrength`]) and a task below
//! [`MIN_HIDDEN_KILL_RATE`] is refused as [`Reason::WeakHiddenTests`].
//!
//! Every candidate that is not admitted is counted under a [`Reason`].
//! Admission is idempotent: a task is identified by a key over the baseline
//! snapshot, the change that seeded it and its family, and a key already
//! [held](Held) is not admitted again; nor is a request that repeats, by the
//! near-duplicate rule, one already made for the family. The family of a mutation-seeded task is the
//! path of the file that was changed.

mod fixtures;
mod harness;
mod held;
mod instruction;
mod mutate;
mod pack;
mod profile;
mod staging;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use splinter_core::digest::Digest;
use splinter_core::experience::{ExperienceError, Task};
use splinter_eval::verifiers::VerifyError;
use splinter_sandbox::{
    Baseline, CodeResult, CommandCall, Sandbox, SandboxError, TreeSnapshot, WorkspaceEnvironment,
    WorkspaceHost,
};
use splinter_store::artifacts::ArtifactStore;
use splinter_store::error::StoreError;

pub use fixtures::Fixture;
pub use harness::{Harness, SuiteReport};
pub use held::Held;
pub use profile::{
    Qualified, RepositoryProfile, BASELINE_RUNS, CARGO_TEST, UNITTEST, UNITTEST_DISCOVER,
};
pub use splinter_eval::verifiers::hidden_strength::HiddenStrength;

/// The generator's name in provenance.
pub const WORKSPACE_GENERATOR: &str = "splinter-knowledge/workspace-tasks@1";

/// The most tasks one generation admits, by default.
pub const DEFAULT_MAX_TASKS: usize = 64;

/// The most mutants of one file considered, by default.
pub const DEFAULT_MUTANTS_PER_FILE: usize = 16;

/// How many times the baseline's time one run of a candidate's suite may
/// take, by default.
pub const DEFAULT_TIME_FACTOR: f64 = 10.0;

/// The least time one run of a candidate's suite may take, by default,
/// however quick the baseline.
pub const DEFAULT_MIN_TIME_BUDGET: Duration = Duration::from_secs(60);

/// How many times a failure must reproduce on the tree the solver is given.
pub const REPRODUCTIONS: usize = 2;

/// The least share of the faults seeded near a task's own, among those the
/// whole suite catches, that its hidden tests must catch for the task to be
/// admitted. Below it the hidden tests fail on the one seeded fault and let a
/// plausible wrong patch of its neighbours through, so a pass would say
/// little. Half is a floor, not a target: it refuses suites that miss most
/// neighbours and tolerates the equivalent mutants and the faults the visible
/// tests catch that the denominator cannot tell from a miss.
pub const MIN_HIDDEN_KILL_RATE: f64 = 0.5;

/// The fewest demonstrably wrong variants a refusal rests on. With fewer the
/// rate is recorded but a task is not refused on it: one variant caught or
/// missed is an anecdote, not a rate.
pub const MIN_VARIANTS_TO_REFUSE: usize = 2;

/// The most variants of a task's fault measured, each a run of the suite.
pub const VARIANTS_PER_TASK: usize = 6;

/// How far, in lines, from the seeded fault a variant lies: a wrong patch is
/// made about the code the failing test points at.
pub const VARIANT_WINDOW_LINES: usize = 3;

/// Why a baseline, a mutant or a fixture was not made into a task.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Reason {
    /// Tests fail on the baseline.
    RedBaseline,
    /// Tests differ between two runs of the baseline.
    FlakyBaseline,
    /// The baseline ran no test.
    NoTests,
    /// A run was stopped by a limit.
    TimedOut,
    /// The runner's report was cut at the output cap.
    OutputTruncated,
    /// A test id holds a character the runner reads as a pattern.
    UnselectableTests,
    /// No test fails on the mutant: it is equivalent to the original as far
    /// as the tests can tell (or lies in code they do not reach).
    EquivalentMutant,
    /// The mutant does not run the suite the baseline ran (a module no
    /// longer loads): a failure of everything says nothing about a test.
    BrokenBuild,
    /// A test that fails is not in a file of its own, or is in a file under
    /// test, so it cannot be taken out of the tree.
    TestsNotSeparable,
    /// The failure does not reproduce on every run, or a test that should
    /// pass does not.
    Flaky,
    /// The reverse patch is not a clean change to the one file, or does not
    /// make the task's check pass.
    NotRestorable,
    /// One run of the candidate's suite took longer than the budget.
    OverTimeBudget,
    /// The request would quote a line that was changed.
    LeaksChange,
    /// The request quotes or points at the hidden tests.
    NotSelfContained,
    /// The request is a near duplicate of one admitted for the same family.
    NearDuplicate,
    /// The hidden tests catch fewer than [`MIN_HIDDEN_KILL_RATE`] of the
    /// other faults seeded near this one: a plausible wrong patch would pass
    /// them.
    WeakHiddenTests,
    /// A task with the same key is already held.
    AlreadyHeld,
    /// The fixture's check does not fail on its tree, or fails on a command
    /// that should pass.
    NotFailingAsDescribed,
}

/// Why workspace tasks could not be generated.
#[derive(Debug, thiserror::Error)]
pub enum WorkspaceError {
    /// A parameter is out of range.
    #[error("invalid workspace parameter {name}: {reason}")]
    Parameter {
        /// The parameter.
        name: &'static str,
        /// Why it is refused.
        reason: String,
    },
    /// No rule gives a profile for the tree.
    #[error("{tree}: no test command is known for this tree; give a profile")]
    NoProfile {
        /// The tree.
        tree: PathBuf,
    },
    /// The baseline cannot be trusted to generate from.
    #[error("the baseline is refused ({reason:?}, {count}): {detail}")]
    BaselineRefused {
        /// The reason, counted.
        reason: Reason,
        /// How many tests it concerns.
        count: usize,
        /// What was found.
        detail: String,
    },
    /// What was asked cannot be done with the runner the profile names.
    #[error("unsupported: {what}")]
    Unsupported {
        /// What was asked.
        what: String,
    },
    /// A fixture is not laid out as expected.
    #[error("{path}: {reason}")]
    Fixture {
        /// The fixture's path.
        path: PathBuf,
        /// What is wrong.
        reason: String,
    },
    /// An archive could not be made or laid out.
    #[error("{what} failed: {detail}")]
    Archive {
        /// What was being done.
        what: String,
        /// What went wrong.
        detail: String,
    },
    /// A file operation failed.
    #[error("{path}: {source}")]
    Io {
        /// The file or directory.
        path: PathBuf,
        /// The underlying error.
        source: std::io::Error,
    },
    /// The sandbox failed.
    #[error(transparent)]
    Sandbox(#[from] SandboxError),
    /// The artifact store failed.
    #[error(transparent)]
    Store(#[from] StoreError),
    /// The workspace verifier could not run.
    #[error(transparent)]
    Verify(#[from] VerifyError),
    /// A task is not valid.
    #[error(transparent)]
    Task(#[from] ExperienceError),
    /// A record cannot be serialized.
    #[error("cannot serialize: {0}")]
    Serialize(#[from] serde_json::Error),
}

/// What bounds a generation.
#[derive(Clone, Debug, PartialEq)]
pub struct WorkspacePlan {
    /// The most tasks to admit.
    pub max_tasks: usize,
    /// The most mutants of one file to consider.
    pub mutants_per_file: usize,
    /// How many times the baseline's time one run of a candidate's suite may
    /// take.
    pub time_factor: f64,
    /// The least time one run may take, however quick the baseline.
    pub min_time_budget: Duration,
    /// Words in a shingle of the near-duplicate rule.
    pub shingle_words: usize,
    /// The overlap at which two requests of one family are near duplicates.
    pub max_overlap: f64,
}

impl Default for WorkspacePlan {
    fn default() -> Self {
        Self {
            max_tasks: DEFAULT_MAX_TASKS,
            mutants_per_file: DEFAULT_MUTANTS_PER_FILE,
            time_factor: DEFAULT_TIME_FACTOR,
            min_time_budget: DEFAULT_MIN_TIME_BUDGET,
            shingle_words: super::DEFAULT_SHINGLE_WORDS,
            max_overlap: super::DEFAULT_MAX_OVERLAP,
        }
    }
}

impl WorkspacePlan {
    /// The longest one run of a candidate's suite may take, given how long
    /// the baseline's took.
    #[must_use]
    pub fn time_budget(&self, baseline_secs: Option<f64>) -> Duration {
        let scaled = baseline_secs
            .filter(|secs| secs.is_finite() && *secs >= 0.0)
            .map(|secs| Duration::from_secs_f64(secs * self.time_factor));
        scaled.map_or(self.min_time_budget, |d| d.max(self.min_time_budget))
    }

    fn validate(&self) -> Result<(), WorkspaceError> {
        let bad = |name, reason: &str| {
            Err(WorkspaceError::Parameter {
                name,
                reason: reason.into(),
            })
        };
        if self.mutants_per_file == 0 {
            return bad("mutants_per_file", "at least one mutant per file");
        }
        if !self.time_factor.is_finite() || self.time_factor < 1.0 {
            return bad(
                "time_factor",
                "a candidate's time budget is at least the baseline's",
            );
        }
        if self.shingle_words == 0 || !(0.0..=1.0).contains(&self.max_overlap) {
            return bad(
                "max_overlap",
                "a shingle of one word or more, an overlap in [0, 1]",
            );
        }
        Ok(())
    }
}

/// A task admitted, with what identifies it.
#[derive(Clone, Debug)]
pub struct WorkspaceTask {
    /// The task. Its environment is the solver's snapshot (the hidden tests
    /// taken out); its privileged items are the workspace check and the
    /// reference patch (when the task has one).
    pub task: Task,
    /// The family the task belongs to, for reserving and grouping: the path
    /// of the file that was changed, or the fixture's family.
    pub family: String,
    /// What identifies the task across runs: a digest over the baseline
    /// snapshot, the change that seeded it and the family.
    pub key: Digest,
    /// The tree the solver is given.
    pub snapshot: TreeSnapshot,
    /// The hidden overlay, an archive in the artifact store, when the task
    /// has one.
    pub hidden: Option<Digest>,
}

/// What a generation admitted and what it refused, by reason.
#[derive(Clone, Debug, Default)]
pub struct WorkspaceGeneration {
    /// The tasks admitted, in the order candidates were considered.
    pub admitted: Vec<WorkspaceTask>,
    /// The candidates not admitted, counted by reason.
    pub refused: BTreeMap<Reason, usize>,
}

impl WorkspaceGeneration {
    /// How many candidates were refused for `reason`.
    #[must_use]
    pub fn count(&self, reason: Reason) -> usize {
        self.refused.get(&reason).copied().unwrap_or(0)
    }

    pub(super) fn refuse(&mut self, reason: Reason) {
        *self.refused.entry(reason).or_insert(0) += 1;
    }
}

/// What builds workspace tasks: the sandbox their commands run in, the host
/// tools that lay trees out and read changes back, the artifact store that
/// keeps snapshots and hidden overlays, and a scratch directory.
#[derive(Clone, Debug)]
pub struct WorkspaceGenerator {
    pub(super) sandbox: Arc<dyn Sandbox>,
    pub(super) host: Arc<WorkspaceHost>,
    pub(super) store: ArtifactStore,
    pub(super) scratch: PathBuf,
    pub(super) plan: WorkspacePlan,
}

impl WorkspaceGenerator {
    /// A generator whose tasks run in `sandbox`, with trees made under
    /// `scratch`, bounded by the default plan.
    #[must_use]
    pub fn new(
        sandbox: Arc<dyn Sandbox>,
        host: Arc<WorkspaceHost>,
        store: ArtifactStore,
        scratch: impl Into<PathBuf>,
    ) -> Self {
        Self {
            sandbox,
            host,
            store,
            scratch: scratch.into(),
            plan: WorkspacePlan::default(),
        }
    }

    /// The same generator under `plan`, refused when it is out of range.
    pub fn with_plan(mut self, plan: WorkspacePlan) -> Result<Self, WorkspaceError> {
        plan.validate()?;
        self.plan = plan;
        Ok(self)
    }

    /// The environment of the snapshot `snapshot` in this generator's
    /// sandbox.
    #[must_use]
    pub fn environment(&self, snapshot: &TreeSnapshot) -> WorkspaceEnvironment {
        WorkspaceEnvironment::new(snapshot.clone(), self.sandbox.clone(), self.host.clone())
    }

    /// The file holding the archive of `snapshot` in the artifact store.
    pub fn archive_of(&self, snapshot: &TreeSnapshot) -> Result<PathBuf, WorkspaceError> {
        Ok(self.store.path(&snapshot.digest)?)
    }

    /// Qualifies `profile` on `snapshot` (see
    /// [`RepositoryProfile::qualify`]), whose archive is in the store.
    pub fn qualify(
        &self,
        profile: &RepositoryProfile,
        snapshot: &TreeSnapshot,
    ) -> Result<Qualified, WorkspaceError> {
        profile.qualify(
            &self.environment(snapshot),
            &self.archive_of(snapshot)?,
            &self.scratch,
        )
    }
}

/// A tree laid out afresh from a snapshot's archive, removed when dropped.
pub(super) struct FreshTree {
    _dir: tempfile::TempDir,
    tree: PathBuf,
    baseline: Baseline,
}

impl FreshTree {
    /// `archive` laid out by `env` in a new directory under `scratch`.
    pub(super) fn lay(
        env: &WorkspaceEnvironment,
        archive: &Path,
        scratch: &Path,
    ) -> Result<Self, WorkspaceError> {
        std::fs::create_dir_all(scratch).map_err(|source| WorkspaceError::Io {
            path: scratch.to_path_buf(),
            source,
        })?;
        let dir = tempfile::Builder::new()
            .prefix("splinter-workspace-")
            .tempdir_in(scratch)
            .map_err(|source| WorkspaceError::Io {
                path: scratch.to_path_buf(),
                source,
            })?;
        let tree = dir.path().join("tree");
        let baseline = env.materialize(archive, &tree)?;
        Ok(Self {
            _dir: dir,
            tree,
            baseline,
        })
    }

    pub(super) fn path(&self) -> &Path {
        &self.tree
    }

    pub(super) fn baseline(&self) -> &Baseline {
        &self.baseline
    }
}

/// The variable set on every command the generator runs, so a run leaves no
/// bytecode behind that a later edit within the same second could hide
/// behind. The workspace verifier sets it too.
const NO_BYTECODE: &str = "PYTHONDONTWRITEBYTECODE=1";

/// The exit status `timeout` reports when it stopped its command.
pub(super) const TIMEOUT_EXIT: i32 = 124;

/// Runs `call` in `tree` with bytecode writing off, and says how long it
/// took. With `limit`, the command is stopped by `timeout` after that many
/// seconds (rounded up) and ends with [`TIMEOUT_EXIT`]: the sandbox's own wall
/// time is far longer than a budget set from the baseline, and a mutant that
/// loops would otherwise cost the whole of it.
pub(super) fn run_quiet(
    env: &WorkspaceEnvironment,
    tree: &Path,
    call: &CommandCall,
    limit: Option<Duration>,
) -> Result<(CodeResult, Duration), WorkspaceError> {
    let mut argv = Vec::new();
    if let Some(limit) = limit {
        argv.extend(
            ["timeout", "-k", "1"]
                .map(String::from)
                .into_iter()
                .chain([limit.as_secs_f64().ceil().max(1.0).to_string()]),
        );
    }
    argv.extend(["env".to_string(), NO_BYTECODE.to_string()]);
    argv.extend(call.argv.iter().cloned());
    let started = Instant::now();
    let result = env.run(
        tree,
        &CommandCall {
            argv,
            stdin: call.stdin.clone(),
        },
    )?;
    Ok((result, started.elapsed()))
}
