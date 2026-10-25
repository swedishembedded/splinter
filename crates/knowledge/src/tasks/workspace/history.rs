// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements executable tasks derived from a repository's
// own history, for its clients. If your team needs expertise in turning a
// codebase's commits into verified training tasks, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Tasks from a repository's history: a commit that changes source and tests
//! together is an executable task when its parent tree fails the commit's own
//! tests and the commit's tree passes them, reproducibly.
//!
//! For a revision range, the newest commits are considered first, up to the
//! request's bound. A commit is excluded, with a counted reason, when it is a
//! merge or has no parent, when every path it changes is vendored or
//! generated, when the change makes no difference beyond whitespace, when it
//! changes no test, or when it changes no source file.
//!
//! A candidate is materialised as two trees: the parent (what the solver is
//! given) and the commit (the oracle). The commit's suite must pass on every
//! run of [`ORACLE_RUNS`]; the parent, with the commit's changed test files
//! laid over it, must fail the tests the commit added or changed on every
//! run. A suite whose report differs between runs is flaky and refused; a
//! suite that cannot be read, or that reports no test of the commit's changed
//! files, is unbuildable or unverifiable. The tests that stay passing are the
//! verifier's pass-to-pass guard.
//!
//! Contamination is kept out by construction: the solver's snapshot is the
//! parent tree, so it holds neither the solution change nor the commit's test
//! evolution. The changed test files are an overlay only the verifier lays
//! over the tree, and are protected from edits; the solution is the task's
//! privileged reference patch. The request is built from the commit message
//! and the changed source paths, and is refused if it would quote an added
//! line of the solution or name a changed test file.
//!
//! Every admitted task records the repository and parent commits, the family
//! (the first changed source file), the solver environment's digest, the
//! verifier's evidence and its admission status. A second pass over the same
//! range admits nothing: a task's key is its commit, its parent and its
//! family.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use self::commits::{
    describe, diff_between, history_key, is_source_path, module_id, remember, Commit,
};
use serde::{Deserialize, Serialize};
use splinter_core::clock::Clock;
use splinter_core::digest::Digest;
use splinter_eval::verifiers::test_report::{TestExpectation, TestRunner};
use splinter_eval::verifiers::workspace::{Phase, WorkspaceCheck};

use super::harness::{Harness, SuiteReport};
use super::held::Held;
use super::pack::lay_over;
use super::profile::{is_test_path, list_files, RepositoryProfile};
use super::{
    run_quiet, FreshTree, WorkspaceError, WorkspaceGeneration, WorkspaceGenerator, WorkspaceTask,
    TIMEOUT_EXIT,
};
use crate::capture::{capture_tree, DirtyPolicy, TreeRequest};
use crate::tasks::dedup::Seen;

mod commits;

/// How many times each side of an oracle is run: a test that changes its
/// verdict between runs is flaky, and a task whose oracle is not stable is
/// not admitted.
pub const ORACLE_RUNS: usize = 2;

/// How many commits a range is considered for, by default, newest first.
pub const DEFAULT_MAX_COMMITS: usize = 16;

/// What to read the history of.
#[derive(Clone, Debug)]
pub struct HistoryRequest<'a> {
    /// The git repository: a work tree or a bare repository.
    pub repository: &'a Path,
    /// The revision range, as `git rev-list` understands it (`A..B`).
    pub range: &'a str,
    /// The command that runs the suite; detected from the parent tree when
    /// `None`.
    pub test_command: Option<Vec<String>>,
    /// The most commits considered, newest first.
    pub max_commits: usize,
}

impl<'a> HistoryRequest<'a> {
    /// The history of `range` in `repository`, bounded by
    /// [`DEFAULT_MAX_COMMITS`] and with the test command detected.
    #[must_use]
    pub fn new(repository: &'a Path, range: &'a str) -> Self {
        Self {
            repository,
            range,
            test_command: None,
            max_commits: DEFAULT_MAX_COMMITS,
        }
    }
}

/// Why a commit was not made into a task.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HistoryReason {
    /// A commit with no parent (the root).
    NoParent,
    /// A merge commit.
    Merge,
    /// Every change is whitespace.
    FormattingOnly,
    /// Every changed path is generated.
    Generated,
    /// Every changed path is third-party vendored code.
    Vendored,
    /// No test file changed.
    NoTestChange,
    /// No source file changed.
    NoSourceChange,
    /// A task with the same key is already held.
    Duplicate,
    /// The description repeats one admitted for the same family.
    NearDuplicate,
    /// The commit's suite cannot be read, is cut off, or reports no tests.
    Unbuildable,
    /// The commit's own suite fails on the commit's tree.
    CommitFails,
    /// The commit's tests do not fail on the parent tree: there is no fault
    /// to fix.
    ParentPasses,
    /// A suite's verdict differs between runs.
    Flaky,
    /// No runner reads the suite per test, or no test of the changed files
    /// ran, so the oracle cannot be stated.
    Unverifiable,
    /// The reference change does not make the check pass.
    NotRestorable,
    /// The description would quote an added line of the solution.
    LeaksChange,
    /// The description would name a changed test file.
    LeaksTest,
}

/// What one pass over a range considered and admitted.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct HistoryReport {
    /// The commits considered.
    pub attempted: usize,
    /// The tasks admitted.
    pub accepted: usize,
    /// The candidates refused, by reason.
    pub rejected: BTreeMap<HistoryReason, usize>,
    /// The candidates refused because a suite's verdict changed between runs
    /// (also counted in [`HistoryReport::rejected`]).
    pub flaky: usize,
    /// The candidates that could not be built or run (also counted in
    /// [`HistoryReport::rejected`]).
    pub unbuildable: usize,
    /// The candidates already held from an earlier pass.
    pub duplicate: usize,
    /// The candidates whose oracle could not be stated.
    pub unverifiable: usize,
}

impl HistoryReport {
    /// How many candidates were refused for `reason`.
    #[must_use]
    pub fn count(&self, reason: HistoryReason) -> usize {
        self.rejected.get(&reason).copied().unwrap_or(0)
    }

    fn refuse(&mut self, reason: HistoryReason) {
        *self.rejected.entry(reason).or_insert(0) += 1;
        match reason {
            HistoryReason::Flaky => self.flaky += 1,
            HistoryReason::Unbuildable => self.unbuildable += 1,
            HistoryReason::Duplicate => self.duplicate += 1,
            HistoryReason::Unverifiable => self.unverifiable += 1,
            _ => {}
        }
    }
}

/// Whether a task was admitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Admission {
    /// The oracle held on every run.
    Accepted,
}

/// What the verifier saw when the task was admitted.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HistoryEvidence {
    /// The tests that went from failing on the parent to passing on the
    /// commit, and are the task's hidden fail-to-pass tests.
    pub fail_to_pass: Vec<String>,
    /// How many tests were guarded as pass-to-pass.
    pub pass_to_pass: usize,
    /// How many times the commit's suite was run.
    pub commit_runs: usize,
    /// How many times the parent's tree, with the commit's tests laid over
    /// it, was run.
    pub parent_runs: usize,
    /// The tests that failed on the parent's tree, as the runner named them.
    pub parent_failed: Vec<String>,
}

/// What identifies an admitted history task.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HistoryRecord {
    /// The commit that made the change: the oracle.
    pub repo_sha: String,
    /// The commit whose tree the solver is given.
    pub parent_sha: String,
    /// The family the task belongs to: its first changed source file.
    pub family: String,
    /// The digest of the solver's workspace snapshot.
    pub environment: Digest,
    /// The verifier's evidence.
    pub evidence: HistoryEvidence,
    /// The admission status.
    pub status: Admission,
}

/// An admitted history task, with what identifies it.
#[derive(Clone, Debug)]
pub struct HistoryTask {
    /// The task and the archives the workspace verifier reads.
    pub task: WorkspaceTask,
    /// The registration: commits, family, environment and evidence.
    pub record: HistoryRecord,
}

/// What one pass over a range admitted and what it refused.
#[derive(Clone, Debug, Default)]
pub struct HistoryGeneration {
    /// The tasks admitted, newest commit first.
    pub admitted: Vec<HistoryTask>,
    /// The funnel.
    pub report: HistoryReport,
}

impl HistoryGeneration {
    /// The admitted tasks as the generation the rest of the pipeline reads,
    /// beside the funnel: the same [`WorkspaceTask`] shape a mutation-seeded
    /// generation has, so every later stage is unchanged.
    #[must_use]
    pub fn into_generation(self) -> (WorkspaceGeneration, HistoryReport) {
        let tasks = self.admitted.into_iter().map(|t| t.task).collect();
        (
            WorkspaceGeneration {
                admitted: tasks,
                refused: BTreeMap::new(),
            },
            self.report,
        )
    }
}

/// Why one candidate stopped: it is refused for a reason, or the pass itself
/// failed.
pub(super) enum Halt {
    Reason(HistoryReason),
    Fail(WorkspaceError),
}

impl<E: Into<WorkspaceError>> From<E> for Halt {
    fn from(error: E) -> Self {
        Self::Fail(error.into())
    }
}

impl WorkspaceGenerator {
    /// Tasks from the history of `request.repository` over `request.range`,
    /// held against `held` so an earlier pass adds nothing.
    ///
    /// Refused as a whole only when the history cannot be read at all; every
    /// other refusal is counted in the returned report.
    pub fn from_history(
        &self,
        request: &HistoryRequest<'_>,
        held: &Held,
        clock: &dyn Clock,
    ) -> Result<HistoryGeneration, WorkspaceError> {
        if request.range.is_empty() || request.range.starts_with('-') {
            return Err(WorkspaceError::Parameter {
                name: "range",
                reason: "a revision range may not be empty or start with '-'".into(),
            });
        }
        let commits = self.discover(request)?;
        let mut generation = HistoryGeneration::default();
        let mut seen: BTreeMap<String, Seen> = BTreeMap::new();
        for commit in commits {
            generation.report.attempted += 1;
            if let Some(reason) = self.excluded(request.repository, &commit) {
                generation.report.refuse(reason);
                continue;
            }
            match self.history_candidate(request, &commit, held, &mut seen, clock) {
                Ok(task) => {
                    generation.report.accepted += 1;
                    remember(&mut seen, &task.task);
                    generation.admitted.push(task);
                }
                Err(Halt::Reason(reason)) => generation.report.refuse(reason),
                Err(Halt::Fail(error)) => return Err(error),
            }
        }
        Ok(generation)
    }

    /// One commit qualified as an oracle and made into a task, or refused.
    fn history_candidate(
        &self,
        request: &HistoryRequest<'_>,
        commit: &Commit,
        held: &Held,
        seen: &mut BTreeMap<String, Seen>,
        clock: &dyn Clock,
    ) -> Result<HistoryTask, Halt> {
        let Some(parent_sha) = commit.parent().map(str::to_string) else {
            return Err(Halt::Reason(HistoryReason::NoParent));
        };
        let sources: Vec<String> = commit
            .changes
            .iter()
            .map(|c| c.path.as_str())
            .filter(|p| is_source_path(p))
            .map(str::to_string)
            .collect();
        let tests: Vec<String> = commit
            .changes
            .iter()
            .map(|c| c.path.as_str())
            .filter(|p| is_test_path(p))
            .map(str::to_string)
            .collect();
        let Some(family) = sources.first().cloned() else {
            return Err(Halt::Reason(HistoryReason::NoSourceChange));
        };
        let summaries: Vec<String> = commit
            .changes
            .iter()
            .filter(|change| is_source_path(&change.path))
            .map(|change| format!("{} {}", change.status, change.path))
            .collect();
        let key = history_key(commit, &family)?;
        if held.contains(&key) {
            return Err(Halt::Reason(HistoryReason::Duplicate));
        }

        // The solution, restricted to the source files: the tests are
        // evidence, not an answer the solver may see or a patch may touch.
        let solution = diff_between(request.repository, &parent_sha, &commit.sha, &sources)?;
        if solution.trim().is_empty() {
            return Err(Halt::Reason(HistoryReason::NoSourceChange));
        }
        let description =
            describe(commit, &sources, &summaries, &tests, &solution).map_err(Halt::Reason)?;
        let family_seen = seen
            .entry(family.clone())
            .or_insert_with(|| held.seen(&family, self.plan.shingle_words, self.plan.max_overlap));
        if family_seen.repeats(&description).is_some() {
            return Err(Halt::Reason(HistoryReason::NearDuplicate));
        }

        // Materialise the two trees: the parent the solver gets and the
        // commit the oracle is stated against.
        let capture = |revision: &str| {
            capture_tree(
                &TreeRequest {
                    repository: request.repository,
                    revision,
                    dirty: DirtyPolicy::Refuse,
                    terms: None,
                },
                &self.host,
                &self.store,
                &self.scratch,
                clock,
            )
            .map_err(WorkspaceError::from)
        };
        let parent_tree = capture(&parent_sha)?;
        let oracle_tree = capture(&commit.sha)?;
        let parent_env = self.environment(&parent_tree.snapshot);
        let oracle_env = self.environment(&oracle_tree.snapshot);
        let parent_lay = FreshTree::lay(&parent_env, &parent_tree.archive_path, &self.scratch)?;
        let oracle_lay = FreshTree::lay(&oracle_env, &oracle_tree.archive_path, &self.scratch)?;

        // How the suite runs. Only a runner read per test can name the tests
        // that go from failing to passing.
        let command = match &request.test_command {
            Some(command) if !command.is_empty() => command.clone(),
            _ => RepositoryProfile::detect(parent_lay.path())
                .map(|profile| profile.test_command)
                .map_err(|_| Halt::Reason(HistoryReason::Unverifiable))?,
        };
        let harness = Harness::of(&command).ok_or(Halt::Reason(HistoryReason::Unverifiable))?;
        let files = list_files(oracle_lay.path())?;
        let budget = self.plan.time_budget(None);

        // The commit's own suite passes, reproducibly.
        let oracle_reports =
            self.oracle_reports(&oracle_tree, &oracle_env, &harness, &command, budget)?;
        let oracle = &oracle_reports[0];
        if oracle.ran().next().is_none() {
            return Err(Halt::Reason(HistoryReason::Unbuildable));
        }
        if !oracle.failed.is_empty() {
            return Err(Halt::Reason(HistoryReason::CommitFails));
        }
        let test_files: Vec<String> = tests
            .iter()
            .filter(|path| oracle_lay.path().join(path).is_file())
            .cloned()
            .collect();
        let changed: BTreeSet<String> = test_files.iter().cloned().collect();
        let changed_tests: BTreeSet<String> = oracle
            .passed
            .iter()
            .filter(|id| {
                harness
                    .file_of(id, &files, &command)
                    .is_some_and(|file| changed.contains(&file))
            })
            .cloned()
            .collect();
        if test_files.is_empty() || changed_tests.is_empty() {
            return Err(Halt::Reason(HistoryReason::Unverifiable));
        }

        // The solver's tree: the parent tree, without the commit's tests in
        // the tree; they are the hidden overlay.
        let views = self.stage(
            parent_lay.path(),
            &[],
            Some((oracle_lay.path(), &test_files)),
        )?;
        let solver_env = self.environment(&views.snapshot);
        let parent_reports =
            self.parent_reports(&views, &solver_env, &harness, &command, budget)?;
        let parent = &parent_reports[0];
        if parent.ran().next().is_none() {
            return Err(Halt::Reason(HistoryReason::Unbuildable));
        }
        let mut fail_to_pass: BTreeSet<String> = changed_tests
            .iter()
            .filter(|id| parent.failed.contains(*id))
            .cloned()
            .collect();
        // A commit that adds a symbol its new tests import fails at module
        // load, so the runner names the module rather than each case; that is
        // still every one of the commit's tests failing on the parent.
        if fail_to_pass.is_empty() {
            let module_failure = harness == Harness::Unittest
                && test_files.iter().any(|file| {
                    let module = module_id(file);
                    parent
                        .failed
                        .iter()
                        .any(|id| id == &module || id.ends_with(&format!(".{module}")))
                });
            if module_failure {
                fail_to_pass.clone_from(&changed_tests);
            } else {
                return Err(Halt::Reason(if parent.missing(&changed_tests).is_empty() {
                    HistoryReason::ParentPasses
                } else {
                    HistoryReason::Unverifiable
                }));
            }
        }
        let pass_to_pass: BTreeSet<String> = parent
            .passed
            .iter()
            .filter(|id| !changed_tests.contains(*id))
            .cloned()
            .collect();

        // The check: the hidden fail-to-pass tests, the tests that stay
        // passing, and the test files a patch may not touch.
        let mut expectations = vec![expectation(
            Phase::FailToPass,
            harness.runner(),
            &fail_to_pass,
        )];
        if !pass_to_pass.is_empty() {
            expectations.push(expectation(
                Phase::PassToPass,
                harness.runner(),
                &pass_to_pass,
            ));
        }
        let protected: Vec<String> = list_files(parent_lay.path())?
            .into_iter()
            .filter(|path| is_test_path(path) && !changed.contains(path))
            .collect();
        let check = WorkspaceCheck {
            fail_to_pass: vec![harness.select(&command, &fail_to_pass)],
            pass_to_pass: if pass_to_pass.is_empty() {
                Vec::new()
            } else {
                vec![harness.select(&command, &pass_to_pass)]
            },
            hidden: views.hidden(),
            protected,
            hidden_paths: test_files,
            hidden_strength: None,
            expectations,
            pinned: Vec::new(),
        };
        let task = self.build_task(&solver_env, &check, &description, &family, Some(&solution))?;
        let environment = views.snapshot.digest.clone();
        self.keep(&views)?;
        if !self.passes(&solver_env, &task, &solution, clock)? {
            return Err(Halt::Reason(HistoryReason::NotRestorable));
        }
        let record = HistoryRecord {
            repo_sha: commit.sha.clone(),
            parent_sha,
            family: family.clone(),
            environment,
            evidence: HistoryEvidence {
                fail_to_pass: fail_to_pass.into_iter().collect(),
                pass_to_pass: pass_to_pass.len(),
                commit_runs: ORACLE_RUNS,
                parent_runs: ORACLE_RUNS,
                parent_failed: parent.failed.iter().cloned().collect(),
            },
            status: Admission::Accepted,
        };
        Ok(HistoryTask {
            task: WorkspaceTask {
                task,
                family,
                key,
                hidden: views.hidden(),
                snapshot: views.snapshot.clone(),
            },
            record,
        })
    }

    /// The commit tree's suite, run [`ORACLE_RUNS`] times, refused when it
    /// cannot be read or differs between runs.
    fn oracle_reports(
        &self,
        tree: &crate::capture::CapturedTree,
        env: &splinter_sandbox::WorkspaceEnvironment,
        harness: &Harness,
        command: &[String],
        budget: std::time::Duration,
    ) -> Result<Vec<SuiteReport>, Halt> {
        let mut reports = Vec::with_capacity(ORACLE_RUNS);
        for _ in 0..ORACLE_RUNS {
            let laid = FreshTree::lay(env, &tree.archive_path, &self.scratch)?;
            reports.push(run_suite(env, laid.path(), harness, command, budget)?);
        }
        if differs(&reports) {
            return Err(Halt::Reason(HistoryReason::Flaky));
        }
        Ok(reports)
    }

    /// The solver's tree with the commit's tests laid over it, run
    /// [`ORACLE_RUNS`] times.
    fn parent_reports(
        &self,
        views: &super::staging::Staged,
        env: &splinter_sandbox::WorkspaceEnvironment,
        harness: &Harness,
        command: &[String],
        budget: std::time::Duration,
    ) -> Result<Vec<SuiteReport>, Halt> {
        let mut reports = Vec::with_capacity(ORACLE_RUNS);
        for _ in 0..ORACLE_RUNS {
            let laid = FreshTree::lay(env, &views.archive, &self.scratch)?;
            if let Some(overlay) = &views.overlay {
                lay_over(&overlay.path, laid.path())?;
            }
            reports.push(run_suite(env, laid.path(), harness, command, budget)?);
        }
        if differs(&reports) {
            return Err(Halt::Reason(HistoryReason::Flaky));
        }
        Ok(reports)
    }
}

/// Runs `harness`'s whole suite in `tree`, refused when it cannot be read.
fn run_suite(
    env: &splinter_sandbox::WorkspaceEnvironment,
    tree: &Path,
    harness: &Harness,
    command: &[String],
    budget: std::time::Duration,
) -> Result<SuiteReport, Halt> {
    let call = harness.run_all(command);
    let (result, took) = run_quiet(env, tree, &call, Some(budget))?;
    if result.timed_out || result.exit_code == Some(TIMEOUT_EXIT) || took > budget {
        return Err(Halt::Reason(HistoryReason::Unbuildable));
    }
    let report = harness.read(&result);
    if report.unreadable() || report.elided || report.verdict.is_none() {
        return Err(Halt::Reason(HistoryReason::Unbuildable));
    }
    Ok(report)
}

/// The module a unittest test file is imported as: `tests/test_x.py` is
/// `tests.test_x`.
/// Whether two runs of a suite disagree about any test.
fn differs(reports: &[SuiteReport]) -> bool {
    reports
        .windows(2)
        .any(|pair| pair[0].passed != pair[1].passed || pair[0].failed != pair[1].failed)
}

/// The command of a phase that must be seen to run and pass exactly `ids`.
fn expectation(phase: Phase, runner: TestRunner, ids: &BTreeSet<String>) -> TestExpectation {
    TestExpectation {
        phase,
        index: 0,
        runner,
        tests: ids.clone(),
        min_tests: ids.len(),
    }
}
