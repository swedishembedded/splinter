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
}

impl SuiteReport {
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
        match self {
            Self::Unittest => read_unittest(&result.stderr),
            Self::Cargo => read_cargo(&result.stdout, &result.stderr),
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

/// A header line of the verbose report: `name (id)`, then ` ... status` when
/// the test has no docstring. Returns the name, the id's text and the rest.
fn header(line: &str) -> Option<(&str, &str, Option<&str>)> {
    if line.starts_with([' ', '\t', '=', '-']) {
        return None;
    }
    let (name, rest) = line.split_once(' ')?;
    let rest = rest.strip_prefix('(')?;
    if let Some((inner, status)) = rest.split_once(") ... ") {
        return Some((name, inner, Some(status)));
    }
    rest.strip_suffix(')').map(|inner| (name, inner, None))
}

/// The id a header names: older interpreters print the class only, so the
/// test's name completes it.
fn compose(name: &str, inner: &str) -> String {
    if inner.ends_with(&format!(".{name}")) {
        inner.to_string()
    } else {
        format!("{inner}.{name}")
    }
}

/// `FAIL: name (id)` or `ERROR: name (id)` of the summary the runner prints
/// after the per-test lines.
fn failure(line: &str) -> Option<String> {
    let rest = line
        .strip_prefix("FAIL: ")
        .or_else(|| line.strip_prefix("ERROR: "))?;
    let (name, inner, _) = header(rest)?;
    Some(compose(name, inner))
}

fn read_unittest(stderr: &str) -> SuiteReport {
    let lines: Vec<&str> = stderr.lines().collect();
    let mut seen = BTreeSet::new();
    let mut skipped = BTreeSet::new();
    let mut report = SuiteReport::default();
    let mut summary = false;
    let mut at = 0;
    while at < lines.len() {
        let line = lines[at];
        at += 1;
        summary |= line.starts_with("======");
        if summary {
            report.failed.extend(failure(line));
            continue;
        }
        let Some((name, inner, status)) = header(line) else {
            continue;
        };
        let id = compose(name, inner);
        let status = match status {
            Some(status) => status,
            // A test with a docstring prints it on the next line, with its
            // status.
            None => match lines.get(at).and_then(|next| next.rsplit_once(" ... ")) {
                Some((doc, status)) => {
                    report.docs.insert(id.clone(), doc.trim().to_string());
                    at += 1;
                    status
                }
                None => "",
            },
        };
        if status.starts_with("skipped") {
            skipped.insert(id.clone());
        }
        seen.insert(id);
    }
    report.passed = seen
        .into_iter()
        .filter(|id| !report.failed.contains(id) && !skipped.contains(id))
        .collect();
    report
}

/// The source file of the test binary a `Running` line of cargo's names:
/// `Running unittests src/lib.rs (target/...)` or `Running tests/x.rs (...)`.
fn binary_source(line: &str) -> Option<&str> {
    let rest = line.trim_start().strip_prefix("Running ")?;
    let rest = rest.strip_prefix("unittests ").unwrap_or(rest);
    rest.rsplit_once(" (").map(|(file, _)| file)
}

/// libtest's outcome of one test: `test name ... ok`. A should-panic test
/// is printed `name - should panic`, which is not part of its name.
fn libtest_line(line: &str) -> Option<(&str, &str)> {
    let (name, status) = line.strip_prefix("test ")?.rsplit_once(" ... ")?;
    Some((name.strip_suffix(" - should panic").unwrap_or(name), status))
}

/// Cargo prints each test binary's `Running` line on stderr and the binary's
/// libtest report on stdout, so the two are paired by order: the nth block
/// of tests (`running N tests`) is the nth binary. Blocks past the last
/// binary are doc-tests.
fn read_cargo(stdout: &str, stderr: &str) -> SuiteReport {
    let binaries: Vec<&str> = stderr.lines().filter_map(binary_source).collect();
    let mut report = SuiteReport::default();
    let mut block: Option<usize> = None;
    // Past `failures:` a block quotes the output of its failed tests, which
    // may print anything, a line shaped like a result included.
    let mut quoting = false;
    for line in stdout.lines() {
        if line.starts_with("running ") && (line.ends_with(" tests") || line.ends_with(" test")) {
            block = Some(block.map_or(0, |b| b + 1));
            quoting = false;
            continue;
        }
        quoting |= line == "failures:";
        if quoting {
            continue;
        }
        let (Some(binary), Some((name, status))) =
            (block.and_then(|b| binaries.get(b)), libtest_line(line))
        else {
            continue;
        };
        let id = format!("{binary}::{name}");
        match status {
            "ok" => {
                report.passed.insert(id);
            }
            "FAILED" => {
                report.failed.insert(id);
            }
            // Ignored tests did not run.
            _ => {}
        }
    }
    report
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
        };
        let asked: BTreeSet<String> = ["a", "b", "c"].map(String::from).into();
        assert_eq!(
            report.missing(&asked).into_iter().collect::<Vec<_>>(),
            ["c"]
        );
    }
}
