// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements executable evaluation of code changes
// against a project's own test suite, for its clients. If your team needs
// expertise in test-driven benchmarks for code-editing agents, you can
// procure our services by sending an email to info@swedishembedded.com.

//! A test runner as the generator drives it: run every test and read each
//! one's outcome, run a chosen few, and find the file a test lives in.
//!
//! A task needs to say which tests must go from failing to passing and which
//! must keep passing, so the generator reads the runner's per-test report.
//! The runner is recognised from the profile's test command; only
//! `python3 -m unittest` and `cargo test` are read per test. A command of
//! another runner can still qualify a baseline, by its exit status alone.
//!
//! A test's id is what the runner calls it. For cargo it is the source file
//! of the test binary the test lives in, then `::`, then libtest's name for
//! the test (`tests/total.rs::prices_are_added_up`; a unit test is reported
//! under its crate root, `src/lib.rs::module::tests::name`). The file is
//! what [`Harness::file_of`] finds; libtest selects by the name alone, so
//! two binaries that hold a test of the same name are selected together.

use std::collections::{BTreeMap, BTreeSet};

use splinter_eval::verifiers::test_report::{TestRunner, Verdict};
use splinter_sandbox::{CodeResult, CommandCall};

/// The test runners read per test.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Harness {
    /// `python3 -m unittest`, with or without `discover`.
    Unittest,
    /// `cargo test`, read from libtest's line per test. Doc-tests are not
    /// read: their names hold spaces and cannot be selected.
    Cargo,
}

/// What one run of a suite reported, per test.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SuiteReport {
    /// The tests that ran and passed.
    pub passed: BTreeSet<String>,
    /// The tests that failed or errored. A fixture or a module that could
    /// not be loaded is named here too, as the runner names it.
    pub failed: BTreeSet<String>,
    /// The first line of each test's docstring, for the tests that have one.
    pub docs: BTreeMap<String, String>,
    /// The exception message of each failed test, where the runner prints
    /// one per test (`unittest`); never a traceback frame.
    pub messages: BTreeMap<String, String>,
    /// Whether the middle of the runner's output was elided at the output
    /// cap, so that the tests above are only those the head and tail show.
    pub elided: bool,
    /// How the run ended by the runner's own summary; `None` when no
    /// summary was kept, which is the only case a run is unreadable.
    pub verdict: Option<Verdict>,
}

impl SuiteReport {
    /// Whether the run cannot be read at all: its output was cut and the
    /// summary that closes it is not among what was kept.
    #[must_use]
    pub fn unreadable(&self) -> bool {
        self.elided && self.verdict.is_none()
    }

    /// Every test that ran, passed or failed.
    pub fn ran(&self) -> impl Iterator<Item = &String> {
        self.passed.iter().chain(&self.failed)
    }

    /// The tests of `expected` that the run did not report, neither as
    /// passed nor as failed. A test that was asked for and did not run says
    /// nothing, and is not a pass.
    #[must_use]
    pub fn missing<'a>(&self, expected: impl IntoIterator<Item = &'a String>) -> BTreeSet<String> {
        expected
            .into_iter()
            .filter(|id| !self.passed.contains(*id) && !self.failed.contains(*id))
            .cloned()
            .collect()
    }
}

/// The program that cuts unittest cases out of a module by the parser's line
/// ranges (see [`Harness::cut_cases`]).
const CUT_UNITTEST_CASES: &str = include_str!("cut_unittest_cases.py");

/// Characters of a test id that the selection patterns would read as
/// wildcards.
const PATTERN_CHARS: [char; 3] = ['*', '?', '['];

impl Harness {
    /// The runner `argv` invokes, when it is one that is read per test.
    #[must_use]
    pub fn of(argv: &[String]) -> Option<Self> {
        let program = argv.first()?.rsplit('/').next()?;
        let word = |at: usize| argv.get(at).map(String::as_str);
        let python = program == "python" || program.starts_with("python3");
        if python && word(1) == Some("-m") && word(2) == Some("unittest") {
            Some(Self::Unittest)
        } else if program == "cargo" && word(1) == Some("test") {
            Some(Self::Cargo)
        } else {
            None
        }
    }

    /// Runs the whole suite, reporting each test.
    #[must_use]
    pub fn run_all(self, base: &[String]) -> CommandCall {
        match self {
            Self::Unittest => call(verbose(base)),
            Self::Cargo => call(no_fail_fast(base)),
        }
    }

    /// Runs the tests `ids` and no others, reporting each.
    ///
    /// Each id is an anchored pattern, so `test_sum` does not select
    /// `test_sum_twice`; an id may not hold a pattern character (see
    /// [`Harness::selectable`]).
    #[must_use]
    pub fn select(self, base: &[String], ids: &BTreeSet<String>) -> CommandCall {
        match self {
            Self::Unittest => {
                let mut argv = verbose(base);
                for id in ids {
                    argv.push("-k".into());
                    argv.push(format!("*{id}"));
                }
                call(argv)
            }
            Self::Cargo => {
                let mut argv = no_fail_fast(base);
                argv.extend(["--".into(), "--exact".into()]);
                argv.extend(ids.iter().map(|id| cargo_name(id).to_string()));
                call(argv)
            }
        }
    }

    /// Whether [`Harness::cut_cases`] knows this runner's test source.
    #[must_use]
    pub fn cuts_cases(self) -> bool {
        self == Self::Unittest
    }

    /// The call that takes the cases `ids` out of the test file `file` of the
    /// tree it runs in, leaving the rest of the file; `None` for a runner
    /// whose source this does not know how to cut.
    #[must_use]
    pub fn cut_cases(self, file: &str, ids: &BTreeSet<String>) -> Option<CommandCall> {
        match self {
            Self::Unittest => {
                let mut argv: Vec<String> = ["python3", "-I", "-c", CUT_UNITTEST_CASES, file]
                    .map(String::from)
                    .into();
                argv.extend(ids.iter().cloned());
                Some(call(argv))
            }
            Self::Cargo => None,
        }
    }

    /// Whether the test `id` can be named to [`Harness::select`] exactly.
    #[must_use]
    pub fn selectable(self, id: &str) -> bool {
        match self {
            Self::Unittest => !id.contains(PATTERN_CHARS),
            // Names are matched whole, so nothing is a pattern; a name the
            // command line would read as a flag or split cannot be given.
            Self::Cargo => {
                let name = cargo_name(id);
                !name.is_empty() && !name.starts_with('-') && !name.contains(char::is_whitespace)
            }
        }
    }

    /// Reads what the runner printed.
    #[must_use]
    pub fn read(self, result: &CodeResult) -> SuiteReport {
        let report = self.runner().read(result);
        SuiteReport {
            passed: report.passed,
            failed: report.failed,
            docs: report.docs,
            messages: report.messages,
            elided: report.elided,
            verdict: report.verdict,
        }
    }

    /// The reader of this runner's report, which the verifier also uses to
    /// require that the tests of a task were seen to run.
    #[must_use]
    pub fn runner(self) -> TestRunner {
        match self {
            Self::Unittest => TestRunner::Unittest,
            Self::Cargo => TestRunner::Cargo,
        }
    }

    /// The file, relative to the tree, that holds the test `id`: the longest
    /// module path of the id that is one of `files`, found from the tree's
    /// root or from the start directory of a `discover` run.
    #[must_use]
    pub fn file_of(self, id: &str, files: &BTreeSet<String>, base: &[String]) -> Option<String> {
        if self == Self::Cargo {
            return id
                .split_once("::")
                .map(|(file, _)| file)
                .filter(|file| files.contains(*file))
                .map(String::from);
        }
        let roots = discovery_roots(base);
        let parts: Vec<&str> = id.split('.').collect();
        (1..parts.len()).rev().find_map(|k| {
            let module = format!("{}.py", parts[..k].join("/"));
            roots
                .iter()
                .map(|root| format!("{root}{module}"))
                .find(|path| files.contains(path))
        })
    }
}

/// `base` asking for every test to run though one fails.
fn no_fail_fast(base: &[String]) -> Vec<String> {
    let mut argv = base.to_vec();
    if !argv.iter().any(|a| a == "--no-fail-fast") {
        argv.push("--no-fail-fast".into());
    }
    argv
}

/// libtest's name of the test `id` of cargo's: what follows the binary's
/// source file.
fn cargo_name(id: &str) -> &str {
    id.split_once("::").map_or(id, |(_, name)| name)
}

/// `base` asking for a line per test.
fn verbose(base: &[String]) -> Vec<String> {
    let mut argv = base.to_vec();
    if !argv.iter().any(|a| a == "-v") {
        argv.push("-v".into());
    }
    argv
}

fn call(argv: Vec<String>) -> CommandCall {
    CommandCall { argv, stdin: None }
}

/// The directories a test module may be found from: the root, and the
/// start directory a `discover` command names.
fn discovery_roots(base: &[String]) -> Vec<String> {
    let mut roots = vec![String::new()];
    if let Some(start) = base
        .iter()
        .position(|a| a == "-s")
        .and_then(|i| base.get(i + 1))
    {
        let start = start.trim_start_matches("./").trim_end_matches('/');
        if !start.is_empty() && start != "." {
            roots.push(format!("{start}/"));
        }
    }
    roots
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cmd(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|p| (*p).to_string()).collect()
    }

    #[test]
    fn only_unittest_and_cargo_test_are_read_per_test() {
        assert_eq!(
            Harness::of(&cmd(&["python3", "-m", "unittest"])),
            Some(Harness::Unittest)
        );
        assert_eq!(
            Harness::of(&cmd(&["/usr/bin/python3.12", "-m", "unittest", "discover"])),
            Some(Harness::Unittest)
        );
        assert_eq!(
            Harness::of(&cmd(&["cargo", "test", "--offline"])),
            Some(Harness::Cargo)
        );
        assert_eq!(Harness::of(&cmd(&["cargo", "build"])), None);
        assert_eq!(Harness::of(&cmd(&["python3", "-m", "pytest"])), None);
    }

    #[test]
    fn the_verbose_report_is_read_per_test() {
        let stderr = "\
test_one (tests.test_a.A.test_one)
Adds one. ... ok
test_sk (tests.test_a.A.test_sk) ... skipped 'no'
test_twelve (tests.test_a.A.test_twelve) ... ERROR
test_two (tests.test_a.A.test_two) ... FAIL
test_old (tests.test_b.B) ... ok

======================================================================
ERROR: test_twelve (tests.test_a.A.test_twelve)
----------------------------------------------------------------------
Traceback (most recent call last):
RuntimeError: x

======================================================================
FAIL: test_two (tests.test_a.A.test_two)
----------------------------------------------------------------------
AssertionError: 1 != 2

Ran 5 tests in 0.002s
";
        let report = Harness::Unittest.read(&CodeResult {
            stdout: String::new(),
            stderr: stderr.into(),
            exit_code: Some(1),
            signal: None,
            timed_out: false,
            stdout_truncated: false,
            stderr_truncated: false,
        });
        let ids = |set: &BTreeSet<String>| set.iter().cloned().collect::<Vec<_>>();
        assert_eq!(
            ids(&report.passed),
            ["tests.test_a.A.test_one", "tests.test_b.B.test_old"]
        );
        assert_eq!(
            ids(&report.failed),
            ["tests.test_a.A.test_twelve", "tests.test_a.A.test_two"]
        );
        assert_eq!(
            report
                .docs
                .get("tests.test_a.A.test_one")
                .map(String::as_str),
            Some("Adds one.")
        );
    }

    #[test]
    fn a_test_is_found_in_its_module_from_the_root_or_the_start_directory() {
        let files: BTreeSet<String> = ["tests/test_a.py", "pkg/core.py"].map(String::from).into();
        let at = |id: &str, base: &[&str]| Harness::Unittest.file_of(id, &files, &cmd(base));
        let plain = ["python3", "-m", "unittest"];
        let discover = ["python3", "-m", "unittest", "discover", "-s", "tests"];
        assert_eq!(
            at("tests.test_a.A.test_one", &plain).as_deref(),
            Some("tests/test_a.py")
        );
        assert_eq!(
            at("test_a.A.test_one", &discover).as_deref(),
            Some("tests/test_a.py")
        );
        assert_eq!(at("test_a.A.test_one", &plain), None);
    }

    #[test]
    fn a_selection_anchors_each_id_at_its_end() {
        let ids: BTreeSet<String> = ["t.A.test_x".to_string()].into();
        let call = Harness::Unittest.select(&cmd(&["python3", "-m", "unittest"]), &ids);
        assert_eq!(
            call.argv,
            ["python3", "-m", "unittest", "-v", "-k", "*t.A.test_x"]
        );
        assert!(!Harness::Unittest.selectable("t.A.test_x[1]"));
    }

    #[test]
    fn a_cargo_report_pairs_each_binary_with_its_tests_and_skips_doc_tests() {
        let stderr = "\
   Compiling shop v0.1.0
     Running unittests src/lib.rs (target/debug/deps/shop-1)
     Running tests/total.rs (target/debug/deps/total-2)
   Doc-tests shop
error: test failed, to rerun pass `--test total`
";
        let stdout = "\
running 1 test
test util::tests::inner ... ok

test result: ok. 1 passed; 0 failed

running 4 tests
test adds ... ok
test boom - should panic ... ok
test skipped ... ignored
test wrong ... FAILED

failures:

---- wrong stdout ----
test fake ... FAILED

running 1 test
test src/lib.rs - total (line 3) ... ok
";
        let report = Harness::Cargo.read(&CodeResult {
            stdout: stdout.into(),
            stderr: stderr.into(),
            exit_code: Some(101),
            signal: None,
            timed_out: false,
            stdout_truncated: false,
            stderr_truncated: false,
        });
        let ids = |set: &BTreeSet<String>| set.iter().cloned().collect::<Vec<_>>();
        assert_eq!(
            ids(&report.passed),
            [
                "src/lib.rs::util::tests::inner",
                "tests/total.rs::adds",
                "tests/total.rs::boom"
            ]
        );
        assert_eq!(ids(&report.failed), ["tests/total.rs::wrong"]);
    }

    #[test]
    fn cargo_selects_by_exact_name_and_finds_the_binary_file() {
        let ids: BTreeSet<String> = ["tests/t.rs::a::b", "tests/t.rs::c"]
            .map(String::from)
            .into();
        let base = cmd(&["cargo", "test", "--offline"]);
        assert_eq!(
            Harness::Cargo.select(&base, &ids).argv,
            [
                "cargo",
                "test",
                "--offline",
                "--no-fail-fast",
                "--",
                "--exact",
                "a::b",
                "c"
            ]
        );
        assert_eq!(
            Harness::Cargo.run_all(&base).argv,
            ["cargo", "test", "--offline", "--no-fail-fast"]
        );
        let files: BTreeSet<String> = ["tests/t.rs", "src/lib.rs"].map(String::from).into();
        assert_eq!(
            Harness::Cargo
                .file_of("tests/t.rs::a::b", &files, &base)
                .as_deref(),
            Some("tests/t.rs")
        );
        assert_eq!(Harness::Cargo.file_of("gone.rs::a", &files, &base), None);
        assert!(Harness::Cargo.selectable("tests/t.rs::a::b"));
        assert!(!Harness::Cargo.selectable("tests/t.rs::a b"));
    }

    #[test]
    fn a_test_that_was_asked_for_and_not_reported_is_missing() {
        let report = SuiteReport {
            passed: ["a".to_string()].into(),
            failed: ["b".to_string()].into(),
            docs: BTreeMap::new(),
            ..SuiteReport::default()
        };
        let asked: BTreeSet<String> = ["a", "b", "c"].map(String::from).into();
        assert_eq!(
            report.missing(&asked).into_iter().collect::<Vec<_>>(),
            ["c"]
        );
    }
}
