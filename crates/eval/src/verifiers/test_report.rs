// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verifiers that prove a project's tests
// really ran before they count a pass, for its clients. If your team needs
// expertise in tamper-resistant evaluation of code-editing agents, you can
// procure our services by sending an email to info@swedishembedded.com.

//! What a test runner said it did, read back from its output.
//!
//! An exit status of 0 proves only that a process ended calmly. A patch can
//! make that happen without a single test running (exit early, replace the
//! runner, print a success log), so a command that carries a
//! [`TestExpectation`] passes only when the runner's own report shows:
//!
//! - every expected test reported as run and passed;
//! - at least the expected number of tests run, so a suite cannot shrink;
//! - one report, in the stream the runner writes it to, whose per-test lines
//!   agree with its own count;
//! - nothing that looks like a report in the other stream, where a program
//!   under test prints, and no output cut at the cap.
//!
//! The expected names are not shown to the solver, so a printed log cannot
//! name them. What a log printed by the program under test, in the runner's
//! own stream, before the program ends the process, could still do for tests
//! whose names the solver knows is a limit of reading output; a canary
//! covers the exit status, the mutants of the conformance suite the rest.
//!
//! Readers exist for `python3 -m unittest` (verbose) and `cargo test`.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use splinter_sandbox::CodeResult;

use super::workspace::Phase;

/// The test runners whose report is read.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TestRunner {
    /// `python3 -m unittest`, run verbosely; the report is on stderr.
    Unittest,
    /// `cargo test`; the report is on stdout.
    Cargo,
}

/// The tests a command must be seen to run: attached to a command of a
/// workspace check by phase and position.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TestExpectation {
    /// The phase of the command this describes.
    pub phase: Phase,
    /// The command's position within its phase.
    pub index: usize,
    /// How its output is read.
    pub runner: TestRunner,
    /// Ids that must be reported as run and passed.
    #[serde(default)]
    pub tests: BTreeSet<String>,
    /// The fewest tests the command may run in all (the baseline's number
    /// for a suite that guards behaviour), so tests cannot be dropped.
    #[serde(default)]
    pub min_tests: usize,
}

/// Why a command's report did not prove its tests ran and passed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Flaw {
    /// No report in the runner's stream: it never ran, or ended early.
    NoReport,
    /// The per-test lines and the runner's own count disagree, or a block
    /// the runner opens was never closed.
    Inconsistent,
    /// More than one summary: a second report was printed.
    Duplicated,
    /// Report lines in the stream the runner does not write them to.
    ForeignReport,
    /// The report was cut at the output cap.
    Truncated,
    /// A test failed or errored.
    TestFailed,
    /// An expected test was not reported as run and passed.
    ExpectedTestMissing,
    /// Fewer tests ran than the least allowed.
    TooFewTests,
}

/// What a command's output showed against its expectation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Observation {
    /// Tests the report shows ran.
    pub run: usize,
    /// Tests the report shows failed.
    pub failed: usize,
    /// Expected tests not reported as passed.
    pub missing: usize,
    /// What kept the output from proving the tests ran; empty when it did.
    pub flaws: BTreeSet<Flaw>,
}

impl Observation {
    /// Whether the output proves the expected tests ran and passed.
    #[must_use]
    pub fn proven(&self) -> bool {
        self.flaws.is_empty()
    }
}

/// What a runner's output holds, per test.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TestReport {
    /// Tests that ran and passed.
    pub passed: BTreeSet<String>,
    /// Tests that failed or errored (a module that could not be loaded is
    /// named here as the runner names it).
    pub failed: BTreeSet<String>,
    /// Tests that were skipped or ignored: run in the runner's count, but
    /// not passed.
    pub skipped: BTreeSet<String>,
    /// Doc-tests that passed. They ran, but their names cannot be selected,
    /// so they are never an expected test.
    pub doc_tests: BTreeSet<String>,
    /// The first line of each test's docstring, where it has one.
    pub docs: BTreeMap<String, String>,
    /// Tests the runner says it ran, summed over its summaries; `None` when
    /// it printed none.
    pub declared: Option<usize>,
    /// What is wrong with the report as a whole.
    pub flaws: BTreeSet<Flaw>,
}

impl TestReport {
    /// Every test that ran: passed, failed, skipped or a doc-test.
    pub fn ran(&self) -> impl Iterator<Item = &String> {
        self.passed
            .iter()
            .chain(&self.failed)
            .chain(&self.skipped)
            .chain(&self.doc_tests)
    }
}

impl TestRunner {
    /// The runner `argv` invokes, when it is one that is read.
    #[must_use]
    pub fn of(argv: &[String]) -> Option<Self> {
        let program = argv.first()?.rsplit('/').next()?;
        let python = program == "python" || program.starts_with("python3");
        if python
            && argv.get(1).map(String::as_str) == Some("-m")
            && argv.get(2).map(String::as_str) == Some("unittest")
        {
            Some(Self::Unittest)
        } else if program == "cargo" && argv.iter().skip(1).any(|a| a == "test") {
            Some(Self::Cargo)
        } else {
            None
        }
    }

    /// `argv` made to print what this runner's report needs: unittest names
    /// its tests only when verbose.
    #[must_use]
    pub fn reporting(self, argv: &[String]) -> Vec<String> {
        let mut argv = argv.to_vec();
        if self == Self::Unittest && !argv.iter().any(|a| a == "-v") {
            if let Some(at) = argv.iter().position(|a| a == "unittest") {
                argv.insert(at + 1, "-v".into());
            }
        }
        argv
    }

    /// Reads what the runner printed.
    #[must_use]
    pub fn read(self, result: &CodeResult) -> TestReport {
        let (report, foreign_stream, truncated) = match self {
            Self::Unittest => (
                read_unittest(&result.stderr),
                &result.stdout,
                result.stderr_truncated,
            ),
            Self::Cargo => (
                read_cargo(&result.stdout, &result.stderr),
                &result.stderr,
                result.stdout_truncated,
            ),
        };
        let mut report = report;
        if truncated {
            report.flaws.insert(Flaw::Truncated);
        }
        let mimicked = foreign_stream.lines().any(|line| match self {
            Self::Unittest => is_unittest_summary(line) || unittest_header(line).is_some(),
            Self::Cargo => cargo_test_line(line).is_some() || line.starts_with("test result: "),
        });
        if mimicked {
            report.flaws.insert(Flaw::ForeignReport);
        }
        report
    }
}

impl TestExpectation {
    /// What `result` shows against this expectation.
    #[must_use]
    pub fn observe(&self, result: &CodeResult) -> Observation {
        let report = self.runner.read(result);
        let mut flaws = report.flaws.clone();
        if !report.failed.is_empty() {
            flaws.insert(Flaw::TestFailed);
        }
        let missing = self
            .tests
            .iter()
            .filter(|id| !report.passed.contains(*id))
            .count();
        if missing > 0 {
            flaws.insert(Flaw::ExpectedTestMissing);
        }
        let run = report.ran().count();
        if run < self.min_tests {
            flaws.insert(Flaw::TooFewTests);
        }
        Observation {
            run,
            failed: report.failed.len(),
            missing,
            flaws,
        }
    }
}

// ---------------------------------------------------------------- unittest

/// A header line of the verbose report: `name (id)`, then ` ... status` when
/// the test has no docstring. Returns the name, the id's text and the rest.
fn unittest_header(line: &str) -> Option<(&str, &str, Option<&str>)> {
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
fn unittest_failure(line: &str) -> Option<String> {
    let rest = line
        .strip_prefix("FAIL: ")
        .or_else(|| line.strip_prefix("ERROR: "))?;
    let (name, inner, _) = unittest_header(rest)?;
    Some(compose(name, inner))
}

/// `Ran 3 tests in 0.002s`: the count it names.
fn unittest_ran(line: &str) -> Option<usize> {
    let rest = line.strip_prefix("Ran ")?;
    let (count, rest) = rest.split_once(' ')?;
    let rest = rest.strip_prefix("test")?;
    let rest = rest.strip_prefix('s').unwrap_or(rest);
    rest.starts_with(" in ").then_some(())?;
    count.parse().ok()
}

fn is_unittest_summary(line: &str) -> bool {
    unittest_ran(line).is_some()
}

/// Whether a status text ends a test's line: what the runner prints once the
/// test is over (a test's own printing may come between name and status).
fn terminal_status(status: &str) -> bool {
    ["ok", "FAIL", "ERROR"].contains(&status)
        || ["skipped", "expected failure", "unexpected success"]
            .iter()
            .any(|p| status.starts_with(p))
}

fn read_unittest(stderr: &str) -> TestReport {
    let lines: Vec<&str> = stderr.lines().collect();
    let mut report = TestReport::default();
    let mut tests = 0_usize;
    let mut summaries = Vec::new();
    let mut in_failures = false;
    let mut at = 0;
    while at < lines.len() {
        let line = lines[at];
        at += 1;
        if let Some(n) = unittest_ran(line) {
            summaries.push(n);
            in_failures = true;
            continue;
        }
        in_failures |= line.starts_with("======");
        if in_failures {
            report.failed.extend(unittest_failure(line));
            continue;
        }
        let Some((name, inner, status)) = unittest_header(line) else {
            continue;
        };
        let id = compose(name, inner);
        let mut status = match status {
            Some(status) => status.to_string(),
            // A test with a docstring prints it on the next line, with its
            // status.
            None => match lines.get(at).and_then(|next| next.rsplit_once(" ... ")) {
                Some((doc, status)) => {
                    report.docs.insert(id.clone(), doc.trim().to_string());
                    at += 1;
                    status.to_string()
                }
                None => String::new(),
            },
        };
        // Output the test itself printed sits between its name and status.
        while !terminal_status(&status) && at < lines.len() {
            let next = lines[at];
            if unittest_header(next).is_some() || unittest_ran(next).is_some() {
                break;
            }
            status = next.to_string();
            at += 1;
        }
        tests += 1;
        if status.starts_with("skipped") {
            report.skipped.insert(id);
        } else if status == "ok" {
            report.passed.insert(id);
        } else {
            report.failed.insert(id);
        }
    }
    report.passed.retain(|id| !report.failed.contains(id));
    match summaries.as_slice() {
        [] => {
            report.flaws.insert(Flaw::NoReport);
        }
        [n] => {
            report.declared = Some(*n);
            if *n != tests {
                report.flaws.insert(Flaw::Inconsistent);
            }
        }
        [n, ..] => {
            report.declared = Some(*n);
            report.flaws.insert(Flaw::Duplicated);
        }
    }
    report
}

// ------------------------------------------------------------------- cargo

/// `test name ... status` of libtest: the name and the status.
fn cargo_test_line(line: &str) -> Option<(&str, &str)> {
    let rest = line.strip_prefix("test ")?;
    let (name, status) = rest.rsplit_once(" ... ")?;
    ["ok", "FAILED", "ignored"]
        .iter()
        .any(|s| status.starts_with(s))
        .then_some((name, status))
}

/// `running 3 tests` or `running 1 test`: the count.
fn cargo_running(line: &str) -> Option<usize> {
    let rest = line.strip_prefix("running ")?;
    let (count, rest) = rest.split_once(' ')?;
    matches!(rest, "test" | "tests")
        .then(|| count.parse().ok())
        .flatten()
}

/// `test result: ok. 3 passed; 0 failed; 1 ignored; ...`: the tests it
/// counts (passed, failed and ignored).
fn cargo_result(line: &str) -> Option<usize> {
    let rest = line.strip_prefix("test result: ")?;
    let (_, counts) = rest.split_once(". ")?;
    let mut total = 0;
    for part in counts.split(';') {
        let mut words = part.split_whitespace();
        let (Some(n), Some(label)) = (words.next(), words.next()) else {
            continue;
        };
        if matches!(label, "passed" | "failed" | "ignored") {
            total += n.parse::<usize>().ok()?;
        }
    }
    Some(total)
}

/// The source file of the test binary a `Running` line of cargo's names:
/// `Running unittests src/lib.rs (target/...)` or `Running tests/x.rs (...)`.
fn cargo_binary_source(line: &str) -> Option<&str> {
    let rest = line.trim_start().strip_prefix("Running ")?;
    let rest = rest.strip_prefix("unittests ").unwrap_or(rest);
    rest.rsplit_once(" (").map(|(file, _)| file)
}

/// libtest prints a should-panic test as `name - should panic`, which is not
/// part of its name.
fn cargo_test_name(name: &str) -> &str {
    name.strip_suffix(" - should panic").unwrap_or(name)
}

/// A doc-test is named `src/lib.rs - item (line 3)`: spaces make it
/// impossible to select by name, so it is counted but not identified.
fn is_doc_test(name: &str) -> bool {
    name.contains(" - ") && name.contains(" (line ")
}

/// Reads `cargo test`'s report. Cargo prints each test binary's `Running`
/// line on stderr and the binary's libtest report on stdout, so the two are
/// paired by order: the nth block (`running N tests`) is the nth binary. A
/// test is identified as `<source file of its binary>::<libtest name>`
/// (`tests/total.rs::adds`), so that equal names in two binaries stay apart;
/// when stderr names no binary for a block the bare libtest name is used.
fn read_cargo(stdout: &str, stderr: &str) -> TestReport {
    let binaries: Vec<&str> = stderr.lines().filter_map(cargo_binary_source).collect();
    let mut report = TestReport::default();
    let mut declared = 0_usize;
    let mut counted = 0_usize;
    let mut blocks = 0_usize;
    let mut closed = 0_usize;
    let mut block_lines = 0_usize;
    let mut block_declared = 0_usize;
    // Past `failures:` a block quotes the output of its failed tests, which
    // may print anything, a line shaped like a result included.
    let mut quoting = false;
    for line in stdout.lines() {
        if let Some(n) = cargo_running(line) {
            blocks += 1;
            block_declared = n;
            block_lines = 0;
            declared += n;
            quoting = false;
        } else if let Some(total) = cargo_result(line) {
            closed += 1;
            if total != block_lines || total != block_declared {
                report.flaws.insert(Flaw::Inconsistent);
            }
        } else if line == "failures:" {
            quoting = true;
        } else if let (false, Some((name, status))) = (quoting, cargo_test_line(line)) {
            block_lines += 1;
            counted += 1;
            let name = cargo_test_name(name);
            let binary = blocks.checked_sub(1).and_then(|b| binaries.get(b));
            let doc_test = binary.is_none() && is_doc_test(name);
            let id = match binary {
                Some(binary) => format!("{binary}::{name}"),
                None => name.to_string(),
            };
            if status.starts_with("ok") {
                if doc_test {
                    report.doc_tests.insert(id);
                } else {
                    report.passed.insert(id);
                }
            } else if status.starts_with("ignored") {
                report.skipped.insert(id);
            } else {
                report.failed.insert(id);
            }
        }
    }
    if blocks == 0 {
        report.flaws.insert(Flaw::NoReport);
    } else {
        report.declared = Some(declared);
        if closed != blocks || counted != declared {
            report.flaws.insert(Flaw::Inconsistent);
        }
    }
    report.passed.retain(|id| !report.failed.contains(id));
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(stdout: &str, stderr: &str) -> CodeResult {
        CodeResult {
            stdout: stdout.into(),
            stderr: stderr.into(),
            exit_code: Some(0),
            signal: None,
            timed_out: false,
            stdout_truncated: false,
            stderr_truncated: false,
        }
    }

    fn expect(runner: TestRunner, tests: &[&str], min_tests: usize) -> TestExpectation {
        TestExpectation {
            phase: Phase::FailToPass,
            index: 0,
            runner,
            tests: tests.iter().map(|t| (*t).to_string()).collect(),
            min_tests,
        }
    }

    const GOOD: &str = "test_a (m.T.test_a) ... ok\ntest_b (m.T.test_b) ... ok\n\n\
        ----------------------------------------------------------------------\n\
        Ran 2 tests in 0.001s\n\nOK\n";

    #[test]
    fn unittest_output_that_names_every_expected_test_is_proven() {
        let seen = expect(TestRunner::Unittest, &["m.T.test_a"], 2).observe(&result("", GOOD));
        assert!(seen.proven(), "{seen:?}");
        assert_eq!((seen.run, seen.failed, seen.missing), (2, 0, 0));
    }

    #[test]
    fn what_is_not_a_report_proves_nothing() {
        let want = expect(TestRunner::Unittest, &["m.T.test_a"], 0);
        // Plain "OK" with no test lines, as a patch that prints it would leave.
        let seen = want.observe(&result("", "OK\n"));
        assert!(seen.flaws.contains(&Flaw::NoReport), "{seen:?}");
        // A suite of other tests only.
        let seen = expect(TestRunner::Unittest, &["m.T.test_z"], 0).observe(&result("", GOOD));
        assert!(seen.flaws.contains(&Flaw::ExpectedTestMissing), "{seen:?}");
        // Fewer tests than the baseline had.
        let seen = expect(TestRunner::Unittest, &[], 3).observe(&result("", GOOD));
        assert!(seen.flaws.contains(&Flaw::TooFewTests), "{seen:?}");
    }

    #[test]
    fn a_report_printed_in_the_wrong_stream_or_twice_or_cut_is_not_believed() {
        let want = expect(TestRunner::Unittest, &["m.T.test_a"], 0);
        let seen = want.observe(&result(GOOD, GOOD));
        assert!(seen.flaws.contains(&Flaw::ForeignReport), "{seen:?}");
        let seen = want.observe(&result("", &format!("{GOOD}{GOOD}")));
        assert!(seen.flaws.contains(&Flaw::Duplicated), "{seen:?}");
        let mut cut = result("", GOOD);
        cut.stderr_truncated = true;
        assert!(want.observe(&cut).flaws.contains(&Flaw::Truncated));
        let lying = GOOD.replace("Ran 2 tests", "Ran 9 tests");
        let seen = want.observe(&result("", &lying));
        assert!(seen.flaws.contains(&Flaw::Inconsistent), "{seen:?}");
    }

    #[test]
    fn a_failed_skipped_or_noisy_test_is_read_as_what_it_was() {
        let stderr =
            "test_a (m.T.test_a) ... ok\ntest_b (m.T.test_b) ... printed by the test\nFAIL\n\
            test_c (m.T.test_c) ... skipped 'no'\n\n======\nFAIL: test_b (m.T.test_b)\n\
            ------\nAssertionError\n\nRan 3 tests in 0.001s\n\nFAILED (failures=1, skipped=1)\n";
        let seen = expect(TestRunner::Unittest, &["m.T.test_a", "m.T.test_c"], 0)
            .observe(&result("", stderr));
        assert_eq!((seen.run, seen.failed, seen.missing), (3, 1, 1), "{seen:?}");
        assert!(!seen.flaws.contains(&Flaw::Inconsistent), "{seen:?}");
        assert!(seen.flaws.contains(&Flaw::TestFailed));
    }

    const CARGO_GOOD: &str = "\nrunning 2 tests\ntest tests::a ... ok\ntest tests::b ... ok\n\n\
        test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s\n\n\
        running 0 tests\n\ntest result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s\n";

    #[test]
    fn cargo_output_is_read_per_test_over_all_its_binaries() {
        let want = expect(TestRunner::Cargo, &["tests::a"], 2);
        let seen = want.observe(&result(CARGO_GOOD, "Compiling x\n"));
        assert!(seen.proven(), "{seen:?}");
        let seen = want.observe(&result("test result: ok. 2 passed; 0 failed;\n", ""));
        assert!(seen.flaws.contains(&Flaw::NoReport), "{seen:?}");
        let seen = want.observe(&result("", CARGO_GOOD));
        assert!(seen.flaws.contains(&Flaw::ForeignReport), "{seen:?}");
        let cut = CARGO_GOOD.replace("test result: ok. 2 passed", "test result: ok. 3 passed");
        assert!(want
            .observe(&result(&cut, ""))
            .flaws
            .contains(&Flaw::Inconsistent));
    }

    #[test]
    fn runners_are_recognised_from_the_command() {
        let argv = |parts: &[&str]| parts.iter().map(|p| (*p).to_string()).collect::<Vec<_>>();
        assert_eq!(
            TestRunner::of(&argv(&["python3", "-m", "unittest", "m"])),
            Some(TestRunner::Unittest)
        );
        assert_eq!(
            TestRunner::of(&argv(&["cargo", "test", "--offline"])),
            Some(TestRunner::Cargo)
        );
        assert_eq!(TestRunner::of(&argv(&["python3", "-m", "pytest"])), None);
        assert_eq!(
            TestRunner::Unittest.reporting(&argv(&["python3", "-m", "unittest", "m"])),
            argv(&["python3", "-m", "unittest", "-v", "m"])
        );
    }
}
