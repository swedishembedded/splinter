// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements executable evaluation of code changes
// against a project's own test suite, for its clients. If your team needs
// expertise in test-driven benchmarks for code-editing agents, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Tasks seeded by faults: a mutant of a file under test is a bug whose
//! symptoms the project's own tests describe.
//!
//! A mutant becomes a task only after it has been shown to be a good one,
//! in this order, each step refusing with a [`Reason`]:
//!
//! 1. the key of the change is not held already;
//! 2. the suite, run on the mutant, still runs the tests the baseline ran
//!    (else it is a broken build), within the time budget, and some tests
//!    that passed now fail (else it is equivalent);
//! 3. every failing test lives in a test file of its own, which is taken out
//!    of the tree the solver is given and kept as the hidden overlay;
//! 4. on the tree the solver is given with the overlay laid over it, the
//!    failing tests fail on each of [`REPRODUCTIONS`] runs and the rest pass
//!    (a test that fails once and passes once is flaky, and is never named
//!    as one to fix);
//! 5. the request, composed from the symptoms, quotes neither a changed
//!    line nor the hidden tests, and is not a near duplicate of one already
//!    admitted for the same file;
//! 6. the hidden tests catch at least [`MIN_HIDDEN_KILL_RATE`] of the other
//!    faults seeded within a few lines of this one that the whole suite
//!    catches (the measurement is kept on the task's check);
//! 7. the reverse patch (the original file put back) is a clean change to
//!    that one file, and the workspace verifier passes it.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde::Serialize;
use splinter_core::clock::Clock;
use splinter_core::digest::{canonical_json, Digest};
use splinter_core::experience::{Privileged, PrivilegedKind};
use splinter_core::selfcontained::check_self_contained;
use splinter_eval::verifiers::mutation::{mutants, Mutant};
use splinter_eval::verifiers::workspace::WorkspaceCheck;
use splinter_sandbox::{CommandCall, Patch, TreeSnapshot, WorkspaceEnvironment};

use super::harness::{Harness, SuiteReport};
use super::held::Held;
use super::pack::lay_over;
use super::profile::{is_test_path, Qualified};
use super::staging::Staged;
use super::{
    instruction, run_quiet, FreshTree, HiddenStrength, Reason, WorkspaceError, WorkspaceGeneration,
    WorkspaceGenerator, WorkspaceTask, MIN_HIDDEN_KILL_RATE, MIN_VARIANTS_TO_REFUSE, REPRODUCTIONS,
    TIMEOUT_EXIT, VARIANTS_PER_TASK, VARIANT_WINDOW_LINES,
};
use crate::tasks::dedup::Seen;

/// The privileged kind of the hidden tests' text, as the self-containment
/// rule reads it.
const HIDDEN_TEST: &str = "hidden-test";

/// The shortest changed line, in characters, whose appearance in a request
/// is a leak. A shorter one (a bracket, `pass`) is a word the request may
/// use without saying anything of the change.
const MIN_LEAKED_LINE_CHARS: usize = 6;

/// The mutants of a file considered as variants of a fault, before the
/// window of lines and the per-task cap choose among them.
const VARIANT_POOL: usize = 64;

/// Why one candidate stopped: it is refused for a reason, or the generation
/// itself failed.
pub(super) enum Stop {
    Refuse(Reason),
    Fail(WorkspaceError),
}

impl<E: Into<WorkspaceError>> From<E> for Stop {
    fn from(error: E) -> Self {
        Self::Fail(error.into())
    }
}

/// What a candidate needs to know of the baseline it is a change to.
struct Parent<'a> {
    snapshot: &'a TreeSnapshot,
    archive: PathBuf,
    env: WorkspaceEnvironment,
    qualified: &'a Qualified,
    harness: Harness,
    budget: std::time::Duration,
    held: &'a Held,
}

impl Parent<'_> {
    fn command(&self) -> &[String] {
        &self.qualified.profile.test_command
    }

    fn quarantined(&self, report: &mut SuiteReport) {
        let flaky = &self.qualified.profile.flaky;
        report.passed.retain(|id| !flaky.contains(id));
        report.failed.retain(|id| !flaky.contains(id));
    }

    fn locate(&self, id: &str) -> Option<String> {
        self.harness
            .file_of(id, &self.qualified.files, self.command())
    }
}

/// The tests a mutant fails, and the files they live in.
struct Killed {
    failing: BTreeSet<String>,
    passing: BTreeSet<String>,
    files: BTreeSet<String>,
    docs: BTreeMap<String, String>,
}

impl WorkspaceGenerator {
    /// Tasks seeded by mutating the files under test of `qualified`'s
    /// profile in the tree `baseline`, whose archive is in the artifact
    /// store. A change whose key is in `held` is not made into a task again,
    /// nor a request that repeats one `held` holds for the same file.
    /// Generation stops at the plan's `max_tasks`; `clock` stamps the
    /// throwaway experience the reverse patch is verified through.
    ///
    /// Refused as a whole when `qualified` has no runner read per test.
    pub fn from_mutations(
        &self,
        baseline: &TreeSnapshot,
        qualified: &Qualified,
        held: &Held,
        clock: &dyn Clock,
    ) -> Result<WorkspaceGeneration, WorkspaceError> {
        let Some(harness) = qualified.harness else {
            return Err(WorkspaceError::Unsupported {
                what: format!(
                    "mutation-seeded tasks need a test runner read per test; {:?} is not one",
                    qualified.profile.test_command
                ),
            });
        };
        let env = self.environment(baseline);
        let archive = self.archive_of(baseline)?;
        let base = Parent {
            snapshot: baseline,
            budget: self.plan.time_budget(qualified.profile.baseline_secs),
            archive,
            env,
            qualified,
            harness,
            held,
        };
        let sources = self.sources(&base)?;
        let mut generation = WorkspaceGeneration::default();
        let mut seen: BTreeMap<String, Seen> = BTreeMap::new();
        for (path, text) in &sources {
            for mutant in mutants(text, self.plan.mutants_per_file) {
                if generation.admitted.len() >= self.plan.max_tasks {
                    return Ok(generation);
                }
                let seen = seen.entry(path.clone()).or_insert_with(|| {
                    held.seen(path, self.plan.shingle_words, self.plan.max_overlap)
                });
                match self.candidate(&base, path, text, &mutant, seen, clock) {
                    Ok(task) => {
                        seen.admit(&task.task.instruction);
                        generation.admitted.push(task);
                    }
                    Err(Stop::Refuse(reason)) => generation.refuse(reason),
                    Err(Stop::Fail(error)) => return Err(error),
                }
            }
        }
        Ok(generation)
    }

    /// The text of each file under test, in path order. A file that is not
    /// UTF-8 text, or is not in the tree, has no mutants.
    fn sources(&self, base: &Parent<'_>) -> Result<Vec<(String, String)>, WorkspaceError> {
        let tree = FreshTree::lay(&base.env, &base.archive, &self.scratch)?;
        let mut paths = base.qualified.profile.files_under_test.clone();
        paths.sort();
        paths.dedup();
        Ok(paths
            .into_iter()
            .filter(|path| !is_test_path(path))
            .filter_map(|path| {
                let text = std::fs::read_to_string(tree.path().join(&path)).ok()?;
                Some((path, text))
            })
            .collect())
    }

    /// One mutant of `path` (whose text is `original`), made into a task or
    /// refused.
    fn candidate(
        &self,
        base: &Parent<'_>,
        path: &str,
        original: &str,
        mutant: &Mutant,
        seen: &Seen,
        clock: &dyn Clock,
    ) -> Result<WorkspaceTask, Stop> {
        let mutated = FreshTree::lay(&base.env, &base.archive, &self.scratch)?;
        write(&mutated.path().join(path), &mutant.text)?;
        let change = base
            .env
            .diff_since(mutated.path(), mutated.baseline(), &[])?;
        let key = task_key(base.snapshot, &Digest::of(&change.diff), path)?;
        if base.held.contains(&key) {
            return Err(Stop::Refuse(Reason::AlreadyHeld));
        }
        let killed = self.kill(base, mutated.path())?;
        let views = self.stage_views(base, path, &mutant.text, &mutated, &killed)?;
        let env = self.environment(&views.snapshot);
        let (failing, passing) = self.reproduce(base, &env, &views, &killed)?;
        let hidden = hidden_text(mutated.path(), &killed.files);
        let text = instruction::compose(base.command(), &failing, &killed.docs);
        admissible(&text, original, &mutant.text, &hidden, seen)?;

        let strength = self.hidden_strength(base, path, original, mutant, &killed)?;
        if strength.variants >= MIN_VARIANTS_TO_REFUSE
            && strength
                .kill_rate()
                .is_some_and(|rate| rate < MIN_HIDDEN_KILL_RATE)
        {
            return Err(Stop::Refuse(Reason::WeakHiddenTests));
        }

        let fix = self.fix(&env, &views, path, original)?;
        let check = WorkspaceCheck {
            fail_to_pass: vec![base.harness.select(base.command(), &failing)],
            pass_to_pass: select_if_any(base, &passing),
            hidden: views.hidden(),
            protected: protected(base, &killed.files),
            hidden_paths: killed.files.iter().cloned().collect(),
            hidden_strength: Some(strength),
        };
        let task = self.build_task(&env, &check, &text, path, Some(&fix))?;
        self.keep(&views)?;
        if !self.passes(&env, &task, &fix, clock)? {
            return Err(Stop::Refuse(Reason::NotRestorable));
        }
        Ok(WorkspaceTask {
            task,
            family: path.to_string(),
            key,
            hidden: views.hidden(),
            snapshot: views.snapshot,
        })
    }

    /// What the hidden tests (the test files `killed` takes out of the
    /// solver's tree) do on the other faults seeded within
    /// [`VARIANT_WINDOW_LINES`] of `mutant`, at most [`VARIANTS_PER_TASK`] of
    /// them, each put in the baseline tree and run against the whole suite.
    /// A variant counts when some test catches it (else it may be equivalent
    /// to the original) and the suite still runs the baseline's tests; it is
    /// caught by the hidden tests when a failing test lives in a hidden file.
    /// A variant that runs past the budget or truncates the report is left
    /// out: it says nothing of the tests.
    fn hidden_strength(
        &self,
        base: &Parent<'_>,
        path: &str,
        original: &str,
        mutant: &Mutant,
        killed: &Killed,
    ) -> Result<HiddenStrength, Stop> {
        let mut strength = HiddenStrength {
            variants: 0,
            caught: 0,
        };
        let pool = mutants(original, VARIANT_POOL);
        let near = pool
            .iter()
            .filter(|v| {
                v.text != mutant.text && v.line.abs_diff(mutant.line) <= VARIANT_WINDOW_LINES
            })
            .take(VARIANTS_PER_TASK);
        for variant in near {
            let tree = FreshTree::lay(&base.env, &base.archive, &self.scratch)?;
            write(&tree.path().join(path), &variant.text)?;
            let call = base.harness.run_all(base.command());
            let report = match self.suite(base, &base.env, tree.path(), &call) {
                Ok(report) => report,
                Err(Stop::Refuse(_)) => continue,
                Err(fail) => return Err(fail),
            };
            let ran: BTreeSet<&String> = report.ran().collect();
            if report.failed.is_empty() || ran != base.qualified.tests.iter().collect() {
                continue;
            }
            strength.variants += 1;
            if report.failed.iter().any(|id| {
                base.locate(id)
                    .is_some_and(|file| killed.files.contains(&file))
            }) {
                strength.caught += 1;
            }
        }
        Ok(strength)
    }

    /// Runs the suite on the mutated tree and names what it kills.
    fn kill(&self, base: &Parent<'_>, tree: &Path) -> Result<Killed, Stop> {
        let report = self.suite(base, &base.env, tree, &base.harness.run_all(base.command()))?;
        let ran: BTreeSet<&String> = report.ran().collect();
        if ran != base.qualified.tests.iter().collect() {
            return Err(Stop::Refuse(Reason::BrokenBuild));
        }
        if report.failed.is_empty() {
            return Err(Stop::Refuse(Reason::EquivalentMutant));
        }
        let files_under_test: BTreeSet<&String> =
            base.qualified.profile.files_under_test.iter().collect();
        let mut files = BTreeSet::new();
        for id in &report.failed {
            match base.locate(id) {
                Some(file) if !files_under_test.contains(&file) => {
                    files.insert(file);
                }
                _ => return Err(Stop::Refuse(Reason::TestsNotSeparable)),
            }
        }
        let passing = report
            .passed
            .iter()
            .filter(|id| !report.failed.contains(*id))
            .cloned()
            .collect();
        Ok(Killed {
            failing: report.failed.clone(),
            passing,
            files,
            docs: report.docs,
        })
    }

    /// The tree the solver is given (the mutant without the files that kill
    /// it) and the hidden overlay (those files), archived.
    fn stage_views(
        &self,
        base: &Parent<'_>,
        path: &str,
        mutant: &str,
        mutated: &FreshTree,
        killed: &Killed,
    ) -> Result<Staged, Stop> {
        let solver = FreshTree::lay(&base.env, &base.archive, &self.scratch)?;
        write(&solver.path().join(path), mutant)?;
        for file in &killed.files {
            let hidden = solver.path().join(file);
            std::fs::remove_file(&hidden).map_err(|source| WorkspaceError::Io {
                path: hidden,
                source,
            })?;
        }
        let files: Vec<String> = killed.files.iter().cloned().collect();
        Ok(self.stage(solver.path(), &[], Some((mutated.path(), &files)))?)
    }

    /// Runs the failing and the passing tests on the solver's tree with the
    /// overlay laid over it, [`REPRODUCTIONS`] times. A failing test that
    /// passes once is flaky and dropped; none left is a refusal, as is a
    /// passing test that fails.
    fn reproduce(
        &self,
        base: &Parent<'_>,
        env: &WorkspaceEnvironment,
        views: &Staged,
        killed: &Killed,
    ) -> Result<(BTreeSet<String>, BTreeSet<String>), Stop> {
        let mut failing = killed.failing.clone();
        for _ in 0..REPRODUCTIONS {
            let tree = FreshTree::lay(env, &views.archive, &self.scratch)?;
            if let Some(overlay) = &views.overlay {
                lay_over(&overlay.path, tree.path())?;
            }
            let call = base.harness.select(base.command(), &failing);
            let report = self.suite(base, env, tree.path(), &call)?;
            if !failing
                .iter()
                .all(|id| report.passed.contains(id) || report.failed.contains(id))
            {
                return Err(Stop::Refuse(Reason::BrokenBuild));
            }
            failing.retain(|id| report.failed.contains(id));
            if failing.is_empty() {
                return Err(Stop::Refuse(Reason::Flaky));
            }
            if !killed.passing.is_empty() {
                let call = base.harness.select(base.command(), &killed.passing);
                let report = self.suite(base, env, tree.path(), &call)?;
                let ran: BTreeSet<&String> = report.ran().collect();
                if !report.failed.is_empty() || ran != killed.passing.iter().collect() {
                    return Err(Stop::Refuse(Reason::Flaky));
                }
            }
        }
        Ok((failing, killed.passing.clone()))
    }

    /// One run of `call` in `tree`, read per test, refused when it ran past
    /// the budget or its report was cut off.
    fn suite(
        &self,
        base: &Parent<'_>,
        env: &WorkspaceEnvironment,
        tree: &Path,
        call: &CommandCall,
    ) -> Result<SuiteReport, Stop> {
        let (result, took) = run_quiet(env, tree, call, Some(base.budget))?;
        if result.timed_out || result.exit_code == Some(TIMEOUT_EXIT) || took > base.budget {
            return Err(Stop::Refuse(Reason::OverTimeBudget));
        }
        if result.stderr_truncated {
            return Err(Stop::Refuse(Reason::OutputTruncated));
        }
        let mut report = base.harness.read(&result);
        base.quarantined(&mut report);
        Ok(report)
    }

    /// The change that puts `original` back in `path` of the solver's tree.
    fn fix(
        &self,
        env: &WorkspaceEnvironment,
        views: &Staged,
        path: &str,
        original: &str,
    ) -> Result<String, Stop> {
        let tree = FreshTree::lay(env, &views.archive, &self.scratch)?;
        write(&tree.path().join(path), original)?;
        let patch: Patch = env.diff_since(tree.path(), tree.baseline(), &[])?;
        if patch.files != [PathBuf::from(path)] {
            return Err(Stop::Refuse(Reason::NotRestorable));
        }
        String::from_utf8(patch.diff).map_err(|_| Stop::Refuse(Reason::NotRestorable))
    }
}

/// What identifies a change to `family` of the baseline `snapshot`: the
/// snapshot, the digest of the change and the family.
pub(super) fn task_key(
    snapshot: &TreeSnapshot,
    change: &Digest,
    family: &str,
) -> Result<Digest, WorkspaceError> {
    #[derive(Serialize)]
    struct Key<'a> {
        baseline: &'a Digest,
        change: &'a Digest,
        family: &'a str,
    }
    Ok(Digest::of(&canonical_json(&Key {
        baseline: &snapshot.digest,
        change,
        family,
    })?))
}

/// The commands' selection of `ids`, or none when there are none (an empty
/// selection would run everything).
fn select_if_any(base: &Parent<'_>, ids: &BTreeSet<String>) -> Vec<CommandCall> {
    if ids.is_empty() {
        Vec::new()
    } else {
        vec![base.harness.select(base.command(), ids)]
    }
}

/// The test files that stay in the tree, which a patch may not touch: every
/// file a baseline test lives in, except the hidden ones.
fn protected(base: &Parent<'_>, hidden: &BTreeSet<String>) -> Vec<String> {
    let files: BTreeSet<String> = base
        .qualified
        .tests
        .iter()
        .filter_map(|id| base.locate(id))
        .filter(|file| !hidden.contains(file))
        .collect();
    files.into_iter().collect()
}

/// The text of the hidden test files in `tree`.
fn hidden_text(tree: &Path, files: &BTreeSet<String>) -> Privileged {
    let content: String = files
        .iter()
        .filter_map(|file| std::fs::read(tree.join(file)).ok())
        .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
        .collect::<Vec<_>>()
        .join("\n");
    Privileged {
        kind: PrivilegedKind::Other(HIDDEN_TEST.into()),
        content,
        span: None,
    }
}

/// Whether the request `text` may be shown: it quotes no line that the
/// mutation changed, nothing of the hidden tests, and is not a near
/// duplicate of one admitted for the file.
fn admissible(
    text: &str,
    original: &str,
    mutated: &str,
    hidden: &Privileged,
    seen: &Seen,
) -> Result<(), Stop> {
    let lowered = text.to_lowercase();
    let leaks = changed_lines(original, mutated)
        .iter()
        .any(|line| lowered.contains(&line.to_lowercase()));
    if leaks {
        return Err(Stop::Refuse(Reason::LeaksChange));
    }
    if check_self_contained(text, &[hidden]).is_err() {
        return Err(Stop::Refuse(Reason::NotSelfContained));
    }
    if seen.repeats(text).is_some() {
        return Err(Stop::Refuse(Reason::NearDuplicate));
    }
    Ok(())
}

/// The lines, trimmed, that are in one of the two texts and not the other,
/// leaving out those too short to say anything.
fn changed_lines(a: &str, b: &str) -> BTreeSet<String> {
    let lines =
        |text: &str| -> BTreeSet<String> { text.lines().map(|l| l.trim().to_string()).collect() };
    let (a, b) = (lines(a), lines(b));
    a.symmetric_difference(&b)
        .filter(|l| l.chars().count() >= MIN_LEAKED_LINE_CHARS)
        .cloned()
        .collect()
}

fn write(path: &Path, text: &str) -> Result<(), WorkspaceError> {
    std::fs::write(path, text).map_err(|source| WorkspaceError::Io {
        path: path.to_path_buf(),
        source,
    })
}
