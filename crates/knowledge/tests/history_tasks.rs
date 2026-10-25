// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements executable tasks derived from a repository's
// own history, for its clients. If your team needs expertise in turning a
// codebase's commits into verified training tasks, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: a repository's history is a source of tasks. For a revision range,
//! a commit that changes source and tests together is a task only when its
//! parent tree fails the commit's own tests and the commit's tree passes
//! them, reproducibly.
//!
//! * Formatting-only, generated, vendored and merge commits are excluded,
//!   and so are commits that change no test and commits that change no
//!   source.
//! * The solver's tree is the parent tree: it holds neither the solution
//!   change nor the commit's test evolution. The tests are the hidden,
//!   protected evidence the verifier lays over the solver's tree.
//! * The request is built from the commit message and the changed source
//!   files, never from the diff and never from a hidden test's path.
//! * Admission is counted: attempted, accepted, rejected by reason, and the
//!   flaky, unbuildable, duplicate and unverifiable counts.
//! * Every admitted task records the repository and parent commits, its
//!   family, its environment digest, the verifier evidence and its status.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use common::{git, write, Repo, AT};
use splinter_core::annotation::Outcome;
use splinter_core::clock::FixedClock;
use splinter_core::digest::Digest;
use splinter_core::experience::{Experience, PrivilegedKind, Provenance, Task};
use splinter_eval::verifiers::workspace::{WorkspaceCheck, WorkspaceVerifier};
use splinter_eval::verifiers::Verifier;
use splinter_knowledge::tasks::workspace::{
    Admission, Held, HistoryGeneration, HistoryReason, HistoryRequest, HistoryTask,
    WorkspaceGenerator, WorkspacePlan, WorkspaceTask,
};
use splinter_sandbox::{CommandCall, Limits, ProcessSandbox, TreeSnapshot, WorkspaceEnvironment};

const CALC_BUGGY: &str = "\
def total(values):
    result = 0
    for value in values:
        result -= value
    return result
";

const CALC_FIXED: &str = "\
def total(values):
    result = 0
    for value in values:
        result += value
    return result
";

const CALC_DOUBLED: &str = "\
def total(values):
    result = 0
    for value in values:
        result += value
    return result


def double(value):
    return value * 2
";

const TEST_BASE: &str = "\
import unittest

from pkg.calc import total


class TotalTest(unittest.TestCase):
    def test_empty_values_cost_nothing(self):
        self.assertEqual(total([]), 0)
";

const TEST_FIXED: &str = "\
import unittest

from pkg.calc import total


class TotalTest(unittest.TestCase):
    def test_empty_values_cost_nothing(self):
        self.assertEqual(total([]), 0)

    def test_values_are_added_up(self):
        self.assertEqual(total([2, 3]), 5)
";

const TEST_DOUBLED: &str = "\
import unittest

from pkg.calc import double, total


class TotalTest(unittest.TestCase):
    def test_empty_values_cost_nothing(self):
        self.assertEqual(total([]), 0)

    def test_values_are_added_up(self):
        self.assertEqual(total([2, 3]), 5)

    def test_a_value_is_doubled(self):
        self.assertEqual(double(2), 4)
";

/// The fixed sources a commit changed, to look for in a description.
const SOLUTION_LINES: [&str; 3] = ["result += value", "def double(value)", "return value * 2"];

/// The generator a history spec drives, over a scratch repository.
struct Rig {
    repo: Repo,
    generator: WorkspaceGenerator,
    sandbox: Arc<ProcessSandbox>,
}

impl Rig {
    fn new(name: &str) -> Self {
        let repo = Repo::of_files(
            name,
            &[
                (
                    "pyproject.toml",
                    "[project]\nname = \"calc\"\nversion = \"1\"\n",
                ),
                ("pkg/__init__.py", ""),
                ("pkg/calc.py", CALC_BUGGY),
                ("tests/__init__.py", ""),
                ("tests/test_calc.py", TEST_BASE),
            ],
        );
        let calls = repo.scratch.0.join("calls");
        fs::create_dir_all(&calls).unwrap();
        let sandbox = Arc::new(ProcessSandbox::new(&calls, [], &[], Limits::default()));
        let generator = WorkspaceGenerator::new(
            sandbox.clone(),
            repo.host.clone(),
            repo.store.clone(),
            repo.scratch.0.join("work"),
        )
        .with_plan(WorkspacePlan::default())
        .unwrap();
        Self {
            repo,
            generator,
            sandbox,
        }
    }

    fn commit(&self, message: &str, files: &[(&str, &str)]) -> String {
        for (path, text) in files {
            write(&self.repo.repo.join(path), text);
        }
        git(&self.repo.repo, &["add", "-A", "-f"]);
        git(&self.repo.repo, &["commit", "-q", "-m", message]);
        git(&self.repo.repo, &["rev-parse", "HEAD"])
            .trim()
            .to_string()
    }

    fn head(&self) -> String {
        git(&self.repo.repo, &["rev-parse", "HEAD"])
            .trim()
            .to_string()
    }

    /// A three-commit history after the fixture commit: a fix with its test,
    /// a feature with its test, and a formatting-only change.
    fn history(&self) -> (String, String, String, String) {
        let base = self.head();
        let fix = self.commit(
            "Fix the basket total: add the values instead of subtracting\n\n\
             Before this change a basket of prices came out negative.\n",
            &[
                ("pkg/calc.py", CALC_FIXED),
                ("tests/test_calc.py", TEST_FIXED),
            ],
        );
        let feature = self.commit(
            "Add doubling to the calculator",
            &[
                ("pkg/calc.py", CALC_DOUBLED),
                ("tests/test_calc.py", TEST_DOUBLED),
            ],
        );
        let formatting = self.commit(
            "Reformat the calculator without changing behaviour",
            &[
                (
                    "pkg/calc.py",
                    "\
def total(values):

    result = 0

    for value in values:
        result += value

    return result


def double(value):

    return value * 2
",
                ),
                (
                    "tests/test_calc.py",
                    "\
import unittest

from pkg.calc import double, total


class TotalTest(unittest.TestCase):

    def test_empty_values_cost_nothing(self):
        self.assertEqual(total([]), 0)

    def test_values_are_added_up(self):
        self.assertEqual(total([2, 3]), 5)

    def test_a_value_is_doubled(self):
        self.assertEqual(double(2), 4)
",
                ),
            ],
        );
        (base, fix, feature, formatting)
    }

    /// Adds a vendored-only commit and a merge commit after `formatting`.
    fn vendored_and_merge(&self) {
        self.commit(
            "Vendor the rounding helper",
            &[(
                "vendor/rounding.py",
                "def half(value):\n    return value / 2\n",
            )],
        );
        let main = self.head();
        git(&self.repo.repo, &["checkout", "-q", "-b", "side"]);
        self.commit(
            "Reformat the vendored helper",
            &[(
                "vendor/rounding.py",
                "def half(value):\n\n    return value / 2\n",
            )],
        );
        git(&self.repo.repo, &["checkout", "-q", "main"]);
        git(
            &self.repo.repo,
            &[
                "merge",
                "--no-ff",
                "-q",
                "side",
                "-m",
                "Merge the side branch",
            ],
        );
        assert_eq!(self.head().len(), main.len());
    }

    fn generate(&self, range: &str, held: &Held) -> HistoryGeneration {
        self.generator
            .from_history(
                &HistoryRequest {
                    repository: &self.repo.repo,
                    range,
                    test_command: None,
                    max_commits: 32,
                },
                held,
                &FixedClock::new(AT),
            )
            .unwrap()
    }
}

fn check_of(task: &Task) -> WorkspaceCheck {
    WorkspaceCheck::of_task(task).unwrap().unwrap().1
}

/// The admitted task of the commit `sha`.
fn task_of<'a>(generation: &'a HistoryGeneration, sha: &str) -> &'a HistoryTask {
    generation
        .admitted
        .iter()
        .find(|t| t.record.repo_sha == sha)
        .unwrap_or_else(|| panic!("no admitted task from {sha}: {:#?}", generation.report))
}

/// The solver's tree of `task`, laid out under `dir`.
fn solver_tree(rig: &Rig, task: &WorkspaceTask, dir: &Path) -> PathBuf {
    let _ = fs::remove_dir_all(dir);
    let env = rig.generator.environment(&task.snapshot);
    env.materialize(&rig.generator.archive_of(&task.snapshot).unwrap(), dir)
        .unwrap();
    dir.to_path_buf()
}

/// The task's hidden overlay laid over its solver tree, as a verifier lays it.
fn with_hidden(rig: &Rig, task: &WorkspaceTask, dir: &Path) -> PathBuf {
    let tree = solver_tree(rig, task, dir);
    let check = check_of(&task.task);
    if let Some(hidden) = &check.hidden {
        let status = Command::new("tar")
            .args(["--extract", "--file"])
            .arg(rig.repo.store.path(hidden).unwrap())
            .arg("--directory")
            .arg(&tree)
            .status()
            .unwrap();
        assert!(status.success());
    } else {
        panic!("a history task has no hidden overlay");
    }
    tree
}

fn baseline_fails(rig: &Rig, task: &WorkspaceTask) -> bool {
    let check = check_of(&task.task);
    let dir = rig
        .repo
        .scratch
        .0
        .join(format!("base-{}", &task.key.hex()[..12]));
    let tree = with_hidden(rig, task, &dir);
    let env = rig.generator.environment(&task.snapshot);
    check.fail_to_pass.iter().all(|call| {
        let mut argv = vec!["env".to_string(), "PYTHONDONTWRITEBYTECODE=1".to_string()];
        argv.extend(call.argv.clone());
        env.run(&tree, &CommandCall { argv, stdin: None })
            .unwrap()
            .exit_code
            != Some(0)
    })
}

fn reference_passes(rig: &Rig, task: &Task) -> bool {
    let patch = task
        .privileged
        .iter()
        .find(|p| p.kind == PrivilegedKind::Reference)
        .unwrap();
    let verifier = WorkspaceVerifier::new(
        vec![WorkspaceEnvironment::new(
            snapshot_of(task),
            rig.sandbox.clone(),
            rig.repo.host.clone(),
        )],
        Arc::new(rig.repo.store.clone()),
        rig.repo.scratch.0.join("verify"),
    );
    let experience = Experience::answered_without_a_run(
        task.clone(),
        &patch.content,
        Provenance::new("spec", &FixedClock::new(AT)),
    )
    .unwrap();
    verifier.verify(task, &experience).unwrap().outcome == Outcome::Pass
}

/// The snapshot a task's environment is pinned to, read from its record.
fn snapshot_of(task: &Task) -> TreeSnapshot {
    let spec = &task.environment.spec;
    TreeSnapshot {
        digest: Digest::parse(spec["snapshot"].as_str().unwrap()).unwrap(),
        tree_id: spec["tree_id"].as_str().unwrap().to_string(),
    }
}

#[test]
fn a_fix_with_its_test_and_a_feature_with_its_test_are_admitted_as_oracles() {
    let rig = Rig::new("history-admit");
    let (base, fix, feature, _formatting) = rig.history();
    rig.vendored_and_merge();
    let range = format!("{base}..HEAD");
    let history = rig.generate(&range, &Held::none());

    assert_eq!(history.report.attempted, 6, "{:#?}", history.report);
    assert_eq!(history.report.accepted, 2, "{:#?}", history.report);
    assert_eq!(history.report.flaky, 0);
    assert_eq!(history.report.unbuildable, 0);
    assert_eq!(history.report.duplicate, 0);
    assert_eq!(history.report.unverifiable, 0);
    assert_eq!(
        history.report.attempted,
        history.report.accepted + history.report.rejected.values().sum::<usize>()
    );

    // The fix names the test it brought, and qualifies as an oracle.
    let fixed = task_of(&history, &fix);
    assert_eq!(fixed.record.repo_sha, fix);
    assert_eq!(fixed.record.parent_sha, base);
    assert_eq!(fixed.record.family, "pkg/calc.py");
    assert_eq!(fixed.record.status, Admission::Accepted);
    assert_eq!(fixed.record.environment, fixed.task.snapshot.digest);

    // The feature names its own test.
    let added = task_of(&history, &feature);
    assert_eq!(added.record.repo_sha, feature);

    // Each oracle: the parent tree fails the hidden test, the reference
    // change makes it pass.
    for task in [&fixed.task, &added.task] {
        assert!(baseline_fails(&rig, task), "{}", task.task.instruction);
        assert!(reference_passes(&rig, &task.task));
    }
}

#[test]
fn formatting_only_vendored_and_merge_commits_are_excluded_by_reason() {
    let rig = Rig::new("history-exclude");
    let (base, _fix, _feature, _formatting) = rig.history();
    rig.vendored_and_merge();
    let history = rig.generate(&format!("{base}..HEAD"), &Held::none());

    assert_eq!(
        history.report.count(HistoryReason::FormattingOnly),
        1,
        "{:#?}",
        history.report
    );
    assert_eq!(
        history.report.count(HistoryReason::Vendored),
        2,
        "{:#?}",
        history.report
    );
    assert_eq!(
        history.report.count(HistoryReason::Merge),
        1,
        "{:#?}",
        history.report
    );
    assert!(history
        .admitted
        .iter()
        .all(|t| t.record.family == "pkg/calc.py"));
}

#[test]
fn the_description_states_the_message_and_the_changed_source_files_not_the_diff() {
    let rig = Rig::new("history-description");
    let (base, fix, _feature, _formatting) = rig.history();
    let history = rig.generate(&format!("{base}..HEAD"), &Held::none());
    let fixed = task_of(&history, &fix);
    let instruction = &fixed.task.task.instruction;

    assert!(
        instruction.contains("Fix the basket total"),
        "{instruction}"
    );
    assert!(instruction.contains("pkg/calc.py"), "{instruction}");
    for line in SOLUTION_LINES {
        assert!(!instruction.contains(line), "{line:?} in {instruction}");
    }
    assert!(
        !instruction.contains("tests/"),
        "a hidden test's path is in the request: {instruction}"
    );
    assert!(!instruction.contains("diff --git"), "{instruction}");
    assert!(!instruction.contains("@@"), "{instruction}");
}

#[test]
fn the_solver_tree_is_the_parent_and_the_test_evolution_is_hidden_evidence() {
    let rig = Rig::new("history-contamination");
    let (base, fix, _feature, _formatting) = rig.history();
    let history = rig.generate(&format!("{base}..HEAD"), &Held::none());
    let fixed = task_of(&history, &fix);

    // The solver's tree is the parent: the bug is there and the fix is not.
    let dir = rig.repo.scratch.0.join("solver-history");
    let tree = solver_tree(&rig, &fixed.task, &dir);
    let source = fs::read_to_string(tree.join("pkg/calc.py")).unwrap();
    assert_eq!(
        source, CALC_BUGGY,
        "the solution change is in the solver's tree"
    );
    let tests = fs::read_to_string(tree.join("tests/test_calc.py")).unwrap();
    assert_eq!(tests, TEST_BASE, "test evolution is in the solver's tree");
    assert!(!tests.contains("test_values_are_added_up"));

    // The commit's tests are the hidden overlay, which the verifier sees.
    let hidden = with_hidden(&rig, &fixed.task, &rig.repo.scratch.0.join("hidden-tree"));
    let hidden_tests = fs::read_to_string(hidden.join("tests/test_calc.py")).unwrap();
    assert!(hidden_tests.contains("test_values_are_added_up"));
    let check = check_of(&fixed.task.task);
    assert_eq!(check.hidden_paths, ["tests/test_calc.py"]);
    assert!(!check.protected.contains(&"tests/test_calc.py".to_string()));
}

#[test]
fn a_second_pass_over_the_same_range_admits_nothing_and_counts_the_duplicates() {
    let rig = Rig::new("history-again");
    let (base, _fix, _feature, _formatting) = rig.history();
    let range = format!("{base}..HEAD");
    let first = rig.generate(&range, &Held::none());
    assert_eq!(first.report.accepted, 2, "{:#?}", first.report);

    let held: Held = first.admitted.iter().map(|t| &t.task).collect();
    let second = rig.generate(&range, &held);
    assert_eq!(second.report.accepted, 0);
    assert_eq!(second.report.duplicate, first.report.accepted);
    assert_eq!(
        second.report.count(HistoryReason::Duplicate),
        first.report.accepted
    );
}
