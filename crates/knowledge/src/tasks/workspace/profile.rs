// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements executable evaluation of code changes
// against a project's own test suite, for its clients. If your team needs
// expertise in test-driven benchmarks for code-editing agents, you can
// procure our services by sending an email to info@swedishembedded.com.

//! How a repository's tests are run, and whether its baseline can be
//! trusted to generate tasks from.
//!
//! A [`RepositoryProfile`] names the test command, the files tasks may
//! change (the files under test), how long the suite took and the tests
//! known to be flaky. It is given explicitly or [detected](RepositoryProfile::detect)
//! by rule. [`RepositoryProfile::qualify`] runs the suite twice, each time in
//! a fresh laying-out of the snapshot, and refuses a baseline that is red
//! (a test fails in both runs) or flaky (a test differs between the runs and
//! is not named in the profile's quarantine), with a counted reason.

use std::collections::BTreeSet;
use std::path::Path;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use splinter_eval::verifiers::test_report::Verdict;
use splinter_sandbox::{CommandCall, WorkspaceEnvironment};

use super::harness::Harness;
use super::{run_quiet, FreshTree, Reason, WorkspaceError};

/// How many times the baseline suite is run.
pub const BASELINE_RUNS: usize = 2;

/// The test command of a Python project whose tests import from the root.
pub const UNITTEST: [&str; 3] = ["python3", "-m", "unittest"];

/// The test command of a Python project whose `tests` directory is not a
/// package, so its modules are found by discovery from that directory.
pub const UNITTEST_DISCOVER: [&str; 6] = ["python3", "-m", "unittest", "discover", "-s", "tests"];

/// The test command of a Rust project. Offline: a task's tree has no
/// network.
pub const CARGO_TEST: [&str; 3] = ["cargo", "test", "--offline"];

/// Directories that are never part of a project's own files.
const SKIPPED_DIRS: [&str; 4] = [".git", "__pycache__", "target", "node_modules"];

/// Build files whose presence, beside a `tests` directory, says a project is
/// a Python project.
const PYTHON_MARKERS: [&str; 3] = ["pyproject.toml", "setup.py", "setup.cfg"];

/// How a repository's tests are run; see the module documentation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RepositoryProfile {
    /// The program and arguments that run the whole suite in the tree's root.
    pub test_command: Vec<String>,
    /// The files, relative to the tree, that tasks may change: what the
    /// tests exercise, not the tests themselves.
    pub files_under_test: Vec<String>,
    /// How long the suite took on the baseline, in seconds; absent until
    /// [`RepositoryProfile::qualify`] has measured it.
    #[serde(default)]
    pub baseline_secs: Option<f64>,
    /// Ids of tests known to be flaky. They are quarantined: left out of
    /// the baseline's comparison and never named by a task's check.
    #[serde(default)]
    pub flaky: Vec<String>,
}

impl RepositoryProfile {
    /// The profile the rules give for the tree laid out at `tree`:
    ///
    /// * a `pyproject.toml`, `setup.py` or `setup.cfg` beside a `tests`
    ///   directory is a Python project run by `python3 -m unittest` (with
    ///   `discover -s tests` when `tests` is not a package), its files
    ///   under test the `.py` files that are not tests;
    /// * a `Cargo.toml` is a Rust project run by `cargo test --offline`, its
    ///   files under test the `.rs` files below `src`.
    ///
    /// A tree that matches neither needs a profile given explicitly.
    pub fn detect(tree: &Path) -> Result<Self, WorkspaceError> {
        let files = list_files(tree)?;
        let has = |name: &str| files.contains(name);
        let python =
            PYTHON_MARKERS.iter().any(|m| has(m)) && files.iter().any(|f| f.starts_with("tests/"));
        let (test_command, own): (&[&str], fn(&str) -> bool) = if python {
            let importable_tests = has("tests/__init__.py")
                || files.iter().any(|f| !f.contains('/') && is_test_module(f));
            (
                if importable_tests {
                    &UNITTEST
                } else {
                    &UNITTEST_DISCOVER
                },
                is_python_source,
            )
        } else if has("Cargo.toml") {
            (&CARGO_TEST, is_rust_source)
        } else {
            return Err(WorkspaceError::NoProfile {
                tree: tree.to_path_buf(),
            });
        };
        Ok(Self {
            test_command: test_command.iter().map(|a| (*a).to_string()).collect(),
            files_under_test: files.iter().filter(|f| own(f)).cloned().collect(),
            baseline_secs: None,
            flaky: Vec::new(),
        })
    }

    /// Runs the suite [`BASELINE_RUNS`] times on `archive` (the snapshot
    /// `env` is pinned to), each in a fresh tree made under `scratch`, and
    /// returns what a task generator needs to know of it. Refused, with the
    /// number of tests concerned, when the baseline is red, flaky, empty,
    /// cut off by a limit or has tests the runner cannot select.
    pub fn qualify(
        &self,
        env: &WorkspaceEnvironment,
        archive: &Path,
        scratch: &Path,
    ) -> Result<Qualified, WorkspaceError> {
        if self.test_command.is_empty() {
            return Err(WorkspaceError::Parameter {
                name: "test_command",
                reason: "the test command is empty".into(),
            });
        }
        let harness = Harness::of(&self.test_command);
        let call = harness.map_or_else(
            || CommandCall {
                argv: self.test_command.clone(),
                stdin: None,
            },
            |h| h.run_all(&self.test_command),
        );
        let mut runs = Vec::with_capacity(BASELINE_RUNS);
        let mut files = BTreeSet::new();
        for _ in 0..BASELINE_RUNS {
            let tree = FreshTree::lay(env, archive, scratch)?;
            files = list_files(tree.path())?;
            let (result, took) = run_quiet(env, tree.path(), &call, None)?;
            if result.timed_out {
                return Err(refused(
                    Reason::TimedOut,
                    1,
                    "the suite was stopped by a limit",
                ));
            }
            runs.push((result, took));
        }
        let baseline_secs = runs
            .iter()
            .map(|(_, took)| took.as_secs_f64())
            .fold(0.0, f64::max);
        let tests = match harness {
            Some(h) => self.compare_reports(h, &runs)?,
            None => self.compare_statuses(&runs)?,
        };
        Ok(Qualified {
            profile: Self {
                baseline_secs: Some(baseline_secs),
                ..self.clone()
            },
            tests,
            files,
            harness,
        })
    }

    /// The tests that passed in both runs; the refusal when a run was cut
    /// off, nothing ran, a test the runner cannot select exists, a test
    /// differs between the runs, or one fails in both.
    fn compare_reports(
        &self,
        harness: Harness,
        runs: &[(splinter_sandbox::CodeResult, Duration)],
    ) -> Result<BTreeSet<String>, WorkspaceError> {
        let quarantined: BTreeSet<&String> = self.flaky.iter().collect();
        let mut reports = Vec::new();
        for (result, _) in runs {
            let mut report = harness.read(result);
            if report.unreadable() {
                return Err(refused(
                    Reason::OutputTruncated,
                    1,
                    "the runner's output passed the output cap and its summary was not among what was kept",
                ));
            }
            if report.elided {
                // The summary decides pass or fail, but a baseline names its
                // tests one by one and the middle of the list is gone.
                return Err(match report.verdict {
                    Some(Verdict::Fail) => refused(
                        Reason::RedBaseline,
                        report.failed.len().max(1),
                        "the suite fails on the baseline (its report passed the output cap)",
                    ),
                    _ => refused(
                        Reason::OutputTruncated,
                        1,
                        "the suite passes but its per-test report passed the output cap; \
                         raise --test-output-cap",
                    ),
                });
            }
            let failures = report.failed.len();
            report.passed.retain(|id| !quarantined.contains(id));
            report.failed.retain(|id| !quarantined.contains(id));
            if report.ran().next().is_none() {
                return Err(refused(Reason::NoTests, 0, "no test ran"));
            }
            reports.push((report, result.exit_code, failures));
        }
        let unselectable = reports[0]
            .0
            .ran()
            .filter(|id| !harness.selectable(id))
            .count();
        if unselectable > 0 {
            return Err(refused(
                Reason::UnselectableTests,
                unselectable,
                "a test id holds a character the runner reads as a pattern",
            ));
        }
        let (first, second) = (&reports[0].0, &reports[1].0);
        let differing: BTreeSet<&String> = first
            .passed
            .symmetric_difference(&second.passed)
            .chain(first.failed.symmetric_difference(&second.failed))
            .collect();
        if !differing.is_empty() {
            return Err(refused(
                Reason::FlakyBaseline,
                differing.len(),
                "tests differ between two runs of the same tree; name them in the profile's \
                 flaky list to quarantine them",
            ));
        }
        if !first.failed.is_empty() {
            return Err(refused(
                Reason::RedBaseline,
                first.failed.len(),
                "tests fail on the baseline",
            ));
        }
        // A nonzero status explained by a failure of a quarantined test is not
        // a red baseline.
        let unexplained = reports
            .iter()
            .find_map(|(_, code, failures)| code.filter(|c| *c != 0 && *failures == 0));
        if let Some(code) = unexplained {
            return Err(refused(
                Reason::RedBaseline,
                1,
                &format!("the suite exits with status {code} though no test failed"),
            ));
        }
        Ok(first.passed.clone())
    }

    /// The refusal when two runs of a suite read by status alone differ or
    /// fail; no tests are known by name.
    fn compare_statuses(
        &self,
        runs: &[(splinter_sandbox::CodeResult, Duration)],
    ) -> Result<BTreeSet<String>, WorkspaceError> {
        let codes: Vec<Option<i32>> = runs.iter().map(|(r, _)| r.exit_code).collect();
        if codes.windows(2).any(|pair| pair[0] != pair[1]) {
            return Err(refused(
                Reason::FlakyBaseline,
                1,
                "the suite's status differs between two runs of the same tree",
            ));
        }
        if codes[0] != Some(0) {
            return Err(refused(
                Reason::RedBaseline,
                1,
                &format!("the suite ends with {:?}", codes[0]),
            ));
        }
        Ok(BTreeSet::new())
    }
}

/// A baseline that is known to be sound: the profile with the time its suite
/// took, the tests that pass on it and the files of the tree.
#[derive(Clone, Debug)]
pub struct Qualified {
    /// The profile, its `baseline_secs` measured.
    pub profile: RepositoryProfile,
    /// The ids of the tests that passed in both runs, quarantined tests
    /// left out. Empty when the runner is not read per test.
    pub tests: BTreeSet<String>,
    /// Every file of the tree, relative to its root.
    pub files: BTreeSet<String>,
    /// The runner, when it is read per test.
    pub harness: Option<Harness>,
}

fn refused(reason: Reason, count: usize, detail: &str) -> WorkspaceError {
    WorkspaceError::BaselineRefused {
        reason,
        count,
        detail: detail.to_string(),
    }
}

/// Every file below `tree` (symbolic links not followed), relative to it and
/// `/`-separated, leaving out version control, bytecode caches and build
/// output.
pub(super) fn list_files(tree: &Path) -> Result<BTreeSet<String>, WorkspaceError> {
    let mut files = BTreeSet::new();
    walk(tree, "", &mut files)?;
    Ok(files)
}

fn walk(dir: &Path, prefix: &str, files: &mut BTreeSet<String>) -> Result<(), WorkspaceError> {
    let io = |source| WorkspaceError::Io {
        path: dir.to_path_buf(),
        source,
    };
    for entry in std::fs::read_dir(dir).map_err(io)? {
        let entry = entry.map_err(io)?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let kind = entry.file_type().map_err(io)?;
        if kind.is_dir() {
            if !SKIPPED_DIRS.contains(&name.as_str()) {
                walk(&entry.path(), &format!("{prefix}{name}/"), files)?;
            }
        } else if kind.is_file() {
            files.insert(format!("{prefix}{name}"));
        }
    }
    Ok(())
}

/// Whether `path` is a Python module of the project's own, not a test or a
/// build script.
fn is_python_source(path: &str) -> bool {
    path.ends_with(".py")
        && !is_test_path(path)
        && !matches!(file_name(path), "setup.py" | "conftest.py" | "__init__.py")
}

fn is_rust_source(path: &str) -> bool {
    path.starts_with("src/") && path.ends_with(".rs") && !is_test_path(path)
}

fn file_name(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

/// Whether the file is named like a test module: `test_*.py` or `*_test.py`.
fn is_test_module(path: &str) -> bool {
    let name = file_name(path);
    name.ends_with(".py") && (name.starts_with("test_") || name.ends_with("_test.py"))
}

/// Whether `path` is a test: a test module, or anything below a directory
/// named for tests.
pub(super) fn is_test_path(path: &str) -> bool {
    is_test_module(path)
        || path
            .split('/')
            .any(|part| matches!(part, "tests" | "test" | "testing"))
        || file_name(path) == "conftest.py"
}
