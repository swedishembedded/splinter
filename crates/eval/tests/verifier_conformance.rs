// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verifier conformance suites that tell a
// verifier that is too lenient from one that is too strict, for its clients.
// If your team needs expertise in measuring the false-pass and false-fail
// rates of automatic graders, you can procure our services by sending an
// email to info@swedishembedded.com.

//! Spec: one table per verifier (workspace, executable code, exact
//! reference, stated reference) of what it must decide for the right answer,
//! wrong ones, the right answer written another way, boundaries, missing
//! evidence, a process that ends early, attempts to reach the hidden tests,
//! to change what the verifier runs, and output made to look like a success.
//! False passes and false fails are counted apart; none is allowed, and what
//! a verifier is known to get wrong is listed as a gap and asserted still
//! wrong. Run with `--nocapture` to read the counts.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use common::conformance::{Class, Table, Want};
use common::workspace::{
    patch_replacing, pretending, Fixture, CALC_BUGGY, CALC_FIXED, CONFTEST, HIDDEN_ID, TEST_HIDDEN,
    TEST_KEEP,
};
use common::{experience, python, record, reference, task, verdict, Scratch};
use splinter_core::annotation::{Outcome, Producer};
use splinter_core::experience::Environment;
use splinter_eval::verifiers::executable::{
    ExecutableCheck, ExecutableVerifier, Expectation, ExpectedStdout,
};
use splinter_eval::verifiers::formal::{ExactMatchVerifier, StatedReferenceVerifier};
use splinter_eval::verifiers::normalise::Normalisation;
use splinter_eval::verifiers::Verifier;
use splinter_sandbox::{Limits, ResolvedEnvironment};

// ------------------------------------------------------------ the table itself

#[test]
fn a_table_counts_false_passes_and_false_fails_apart() {
    let mut table = Table::new("probe");
    table.case(Class::Correct, "right", Want::Pass, Outcome::Pass);
    table.case(
        Class::Incorrect,
        "wrong but passed",
        Want::Fail,
        Outcome::Pass,
    );
    table.case(
        Class::Equivalent,
        "right but failed",
        Want::Pass,
        Outcome::Fail,
    );
    table.case(
        Class::Edge,
        "should have abstained",
        Want::Abstain,
        Outcome::Fail,
    );
    table.case(
        Class::Edge,
        "failed to decide",
        Want::Pass,
        Outcome::Abstain,
    );
    table.gap(Class::Edge, "documented", Want::Fail, Outcome::Pass);
    assert_eq!(table.false_passes(), ["wrong but passed"]);
    assert_eq!(table.false_fails(), ["right but failed"]);
    assert_eq!(
        table.misrouted(),
        ["should have abstained", "failed to decide"]
    );
    assert!(table
        .report()
        .contains("false_pass=1 false_fail=1 misrouted=2 known_gaps=1"));
}

// ---------------------------------------------------------------- workspace

/// A patch that creates `path` with one line.
fn creating(path: &str, line: &str) -> String {
    format!(
        "diff --git a/{path} b/{path}\nnew file mode 100644\n--- /dev/null\n+++ b/{path}\n@@ -0,0 +1 @@\n+{line}\n"
    )
}

/// A patch that deletes `path`, whose content was `old`.
fn deleting(path: &str, old: &str) -> String {
    let lines: String = old.lines().map(|l| format!("-{l}\n")).collect();
    format!(
        "diff --git a/{path} b/{path}\ndeleted file mode 100644\n--- a/{path}\n+++ /dev/null\n@@ -1,{} +0,0 @@\n{lines}",
        old.lines().count()
    )
}

const CALC_EQUIVALENT: &str =
    "def add(a, b):\n    return sum((b, a))\n\n\ndef neg(a):\n    return -a\n";
const CALC_WRONG: &str = "def add(a, b):\n    return a * b\n\n\ndef neg(a):\n    return -a\n";
/// The fix a solver who had read the hidden assertion would write.
const CALC_OVERFIT: &str = "def add(a, b):\n    return 12 if (a, b) == (5, 7) else a - b\n\n\ndef neg(a):\n    return -a\n";

#[test]
fn the_workspace_verifier_conforms() {
    let fx = Fixture::new(false);
    let check = fx.check();
    let task_of = fx.task(&check);
    let mut table = Table::new("workspace");
    let verifier = fx.verifier();
    let secret = "test_secret_sum_of_twelve";

    let mut judge = |class: Class, name: &str, want: Want, output: Option<&str>| {
        let (outcome, _, evidence) =
            verdict(&verifier, &task_of, &experience(&task_of, output, "s"));
        assert!(
            !evidence.to_string().contains(secret),
            "{name}: the evidence leaks a hidden test: {evidence}"
        );
        table.case(class, name, want, outcome);
    };
    let fix = |tail: &str| format!("{}{tail}", fx.reference_patch);
    let calc = |text: &str| patch_replacing("calc.py", CALC_BUGGY, text);

    // Correct and equivalent.
    judge(
        Class::Correct,
        "reference patch",
        Want::Pass,
        Some(&fx.reference_patch),
    );
    judge(
        Class::Correct,
        "fix plus an unrelated new file",
        Want::Pass,
        Some(&fix(&creating("NOTES.txt", "x"))),
    );
    judge(
        Class::Equivalent,
        "another way to add",
        Want::Pass,
        Some(&calc(CALC_EQUIVALENT)),
    );
    judge(
        Class::Equivalent,
        "the same fix, patch lacking its final newline",
        Want::Pass,
        Some(fx.reference_patch.trim_end()),
    );
    // Incorrect.
    judge(
        Class::Incorrect,
        "wrong operator",
        Want::Fail,
        Some(&calc(CALC_WRONG)),
    );
    judge(
        Class::Incorrect,
        "change that does not touch the bug",
        Want::Fail,
        Some(&patch_replacing(
            "calc.py",
            "def neg(a):\n    return -a\n",
            "def neg(a):\n    return 0 - a\n",
        )),
    );
    judge(
        Class::Incorrect,
        "stale hunk that does not apply",
        Want::Fail,
        Some(&patch_replacing(
            "calc.py",
            "def add(a, b):\n    return a * b\n",
            "x\n",
        )),
    );
    // Edges.
    judge(Class::Edge, "empty patch", Want::Fail, Some(""));
    judge(Class::Edge, "whitespace only", Want::Fail, Some(" \n\n"));
    judge(
        Class::Edge,
        "prose instead of a diff",
        Want::Fail,
        Some("I fixed add() so it returns a + b."),
    );
    judge(
        Class::Edge,
        "the fix twice",
        Want::Fail,
        Some(&format!("{0}{0}", fx.reference_patch)),
    );
    // Missing evidence.
    judge(Class::MissingEvidence, "no final output", Want::Fail, None);
    // Early exits: the process ends before the runner reports.
    for (name, exit) in [
        ("os._exit(0) at import", "import os\nos._exit(0)\n\n"),
        ("sys.exit(0) at import", "import sys\nsys.exit(0)\n\n"),
        ("SystemExit at import", "raise SystemExit(0)\n\n"),
    ] {
        judge(
            Class::EarlyExit,
            name,
            Want::Fail,
            Some(&calc(&format!("{exit}{CALC_BUGGY}"))),
        );
        judge(
            Class::EarlyExit,
            &format!("{name}, with the fix"),
            Want::Fail,
            Some(&calc(&format!("{exit}{CALC_FIXED}"))),
        );
    }
    // Hidden tests: touching, replacing, deleting, reading them.
    judge(
        Class::HiddenDisclosure,
        "adds a file at a hidden path",
        Want::Fail,
        Some(&fix(&creating("test_hidden.py", "import unittest"))),
    );
    judge(
        Class::HiddenDisclosure,
        "deletes the hidden module",
        Want::Fail,
        Some(&fix(&deleting("test_hidden.py", TEST_HIDDEN))),
    );
    let reads = format!("print(open('test_hidden.py').read())\n\n{CALC_FIXED}");
    judge(
        Class::HiddenDisclosure,
        "a fix that prints the hidden test (evidence must not hold it)",
        Want::Pass,
        Some(&calc(&reads)),
    );
    // Verifier configuration and the runner.
    judge(
        Class::ConfigTampering,
        "edits conftest.py",
        Want::Fail,
        Some(&fix(&patch_replacing(
            "conftest.py",
            CONFTEST,
            "import sys\nsys.exit(0)\n",
        ))),
    );
    judge(
        Class::ConfigTampering,
        "removes conftest.py",
        Want::Fail,
        Some(&fix(&deleting("conftest.py", CONFTEST))),
    );
    judge(
        Class::ConfigTampering,
        "adds sitecustomize.py",
        Want::Fail,
        Some(&fix(&creating("sitecustomize.py", "import os"))),
    );
    judge(
        Class::ConfigTampering,
        "adds tox.ini",
        Want::Fail,
        Some(&fix(&creating("tox.ini", "[tox]"))),
    );
    judge(
        Class::ConfigTampering,
        "adds .cargo/config.toml",
        Want::Fail,
        Some(&fix(&creating("sub/.cargo/config.toml", "[build]"))),
    );
    judge(
        Class::ConfigTampering,
        "weakens the protected visible test",
        Want::Fail,
        Some(&fix(&patch_replacing(
            "test_keep.py",
            TEST_KEEP,
            "import unittest\n",
        ))),
    );
    // Success logs.
    let log = format!("test_x ({HIDDEN_ID}) ... ok\n\nRan 1 test in 0.001s\n\nOK\n");
    let guessed =
        format!("test_secret_sum_of_twelve ({HIDDEN_ID}) ... ok\n\nRan 1 test in 0.001s\n\nOK\n");
    judge(
        Class::LogMimicry,
        "bare OK on stderr, exit 0",
        Want::Fail,
        Some(&calc(&pretending("OK\n", "stderr"))),
    );
    judge(
        Class::LogMimicry,
        "full report of unknown tests on stderr",
        Want::Fail,
        Some(&calc(&pretending(&log, "stderr"))),
    );
    judge(
        Class::LogMimicry,
        "report naming the tests, on stdout",
        Want::Fail,
        Some(&calc(&pretending(&guessed, "stdout"))),
    );
    judge(
        Class::LogMimicry,
        "report printed by a fix, before the real one",
        Want::Fail,
        Some(&calc(&format!(
            "import sys\nsys.stderr.write({guessed:?})\n\n{CALC_FIXED}"
        ))),
    );
    // Known gap: a solver who has the hidden assertion can satisfy exactly it.
    // The hidden-strength measure at admission says how much a pass is worth.
    let (outcome, _, _) = verdict(
        &verifier,
        &task_of,
        &experience(&task_of, Some(&calc(CALC_OVERFIT)), "s"),
    );
    table.gap(
        Class::HiddenDisclosure,
        "code special-cased to the leaked hidden assertion",
        Want::Fail,
        outcome,
    );

    // Abstentions: no check, an environment that was not offered.
    let environment = ResolvedEnvironment::Workspace(fx.env.clone())
        .record()
        .unwrap();
    let bare = task("workspace-change", environment, vec![]);
    let (outcome, _, _) = verdict(
        &verifier,
        &bare,
        &experience(&bare, Some(&fx.reference_patch), "s"),
    );
    table.case(
        Class::MissingEvidence,
        "task without a check",
        Want::Abstain,
        outcome,
    );
    let other = Fixture::new(true);
    let elsewhere = other.task(&other.check());
    let (outcome, _, _) = verdict(
        &verifier,
        &elsewhere,
        &experience(&elsewhere, Some(&other.reference_patch), "s"),
    );
    table.case(
        Class::MissingEvidence,
        "environment not offered",
        Want::Abstain,
        outcome,
    );

    table.assert_conforms(&[
        Class::Correct,
        Class::Incorrect,
        Class::Equivalent,
        Class::Edge,
        Class::MissingEvidence,
        Class::EarlyExit,
        Class::HiddenDisclosure,
        Class::ConfigTampering,
        Class::LogMimicry,
    ]);
}

/// What `source`'s `add` and `neg` return over a grid of inputs, or the
/// error: the oracle for whether a mutant of the fixed module still behaves
/// as it does.
fn behaviour(source: &str) -> String {
    use std::io::Write;
    let probe = "import sys\nns = {}\ntry:\n    exec(sys.stdin.read(), ns)\n    print([ns['add'](a, b) for a in range(-9, 10) for b in range(-9, 10)], [ns['neg'](a) for a in range(-9, 10)])\nexcept BaseException as e:\n    print('error', type(e).__name__)\n";
    let mut child = std::process::Command::new("python3")
        .args(["-c", probe])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(source.as_bytes())
        .unwrap();
    String::from_utf8(child.wait_with_output().unwrap().stdout).unwrap()
}

/// Differential check: seeded plausible-but-wrong changes (mutants of the
/// reference), each judged against ground truth from running the module
/// itself. A mutant that behaves differently must fail; one that behaves the
/// same must pass. The false-pass rate is printed per task kind.
#[test]
fn seeded_mutants_of_the_reference_are_failed_unless_they_behave_the_same() {
    let fx = Fixture::new(false);
    let task_of = fx.task(&fx.check());
    let verifier = fx.verifier();
    let truth = behaviour(CALC_FIXED);
    let mutated = splinter_eval::verifiers::mutation::mutants(CALC_FIXED, 10);
    assert!(
        mutated.len() >= 3,
        "too few mutants to say anything: {}",
        mutated.len()
    );
    let mut table = Table::new("workspace/mutants");
    let (mut wrong, mut false_passes) = (0, 0);
    for m in &mutated {
        let patch = patch_replacing("calc.py", CALC_BUGGY, &m.text);
        let (outcome, _, _) = verdict(
            &verifier,
            &task_of,
            &experience(&task_of, Some(&patch), "s"),
        );
        let same = behaviour(&m.text) == truth;
        wrong += usize::from(!same);
        false_passes += usize::from(!same && outcome == Outcome::Pass);
        let (class, want) = if same {
            (Class::Equivalent, Want::Pass)
        } else {
            (Class::Incorrect, Want::Fail)
        };
        table.case(
            class,
            &format!("{:?} on line {}", m.operator, m.line),
            want,
            outcome,
        );
    }
    println!(
        "false-pass rate, task kind workspace-change (unittest): {false_passes}/{wrong} wrong mutants"
    );
    table.assert_conforms(&[]);
}

// --------------------------------------------------------------- executable

const CORRECT: &str = "Here it is:\n```python\ndef add(a, b):\n    return a + b\n```\n";
const BARE: &str = "def add(a, b):\n    return a + b\n";
const SUMMED: &str = "def add(a, b):\n    return sum((a, b))\n";
const WRONG: &str = "def add(a, b):\n    return a - b\n";
const SPINS: &str = "def add(a, b):\n    while True:\n        pass\n";

fn checks() -> Vec<ExecutableCheck> {
    vec![
        ExecutableCheck {
            code: "assert add(2, 3) == 5\nassert add(-1, 1) == 0\n".into(),
            stdin: None,
            expect: Expectation::exit_code(0),
            environment: None,
        },
        ExecutableCheck {
            code: "print(add(2, 2))\n".into(),
            stdin: None,
            expect: Expectation {
                exit_code: Some(0),
                stdout: Some(ExpectedStdout {
                    text: "4".into(),
                    normalisation: Normalisation::WHITESPACE,
                }),
            },
            environment: None,
        },
    ]
}

#[test]
fn the_executable_verifier_conforms() {
    let scratch = Scratch::new("conf-exec");
    let limits = Limits {
        wall_time_ms: 2_000,
        cpu_seconds: 2,
        ..Limits::default()
    };
    let env = python(&scratch, limits);
    let privileged = checks().iter().map(|c| c.as_check().unwrap()).collect();
    let code = task("code", record(&env), privileged);
    let verifier = ExecutableVerifier::new(vec![env.clone()]);
    let mut table = Table::new("executable");
    let mut judge = |class: Class, name: &str, want: Want, answer: Option<&str>| {
        let (outcome, _, _) = verdict(&verifier, &code, &experience(&code, answer, "s"));
        table.case(class, name, want, outcome);
    };

    judge(
        Class::Correct,
        "fenced block with prose",
        Want::Pass,
        Some(CORRECT),
    );
    judge(Class::Equivalent, "bare code", Want::Pass, Some(BARE));
    judge(
        Class::Equivalent,
        "another way to add",
        Want::Pass,
        Some(SUMMED),
    );
    judge(Class::Equivalent, "extra definitions and a trailing comment", Want::Pass, Some("def helper(x):\n    return x\n\n\ndef add(a, b):\n    return helper(a) + helper(b)\n\n# done\n"));
    judge(Class::Incorrect, "wrong operator", Want::Fail, Some(WRONG));
    judge(
        Class::Incorrect,
        "right on the visible example only",
        Want::Fail,
        Some("def add(a, b):\n    return 5\n"),
    );
    judge(Class::Incorrect, "an object equal to everything", Want::Fail, Some("class Any:\n    def __eq__(self, other):\n        return True\n\n\ndef add(a, b):\n    return Any()\n"));
    judge(Class::Edge, "empty answer", Want::Fail, Some(""));
    judge(
        Class::Edge,
        "syntax error",
        Want::Fail,
        Some("def add(a, b)\n    return a + b\n"),
    );
    judge(Class::Edge, "never terminates", Want::Fail, Some(SPINS));
    judge(Class::MissingEvidence, "no answer", Want::Fail, None);
    judge(
        Class::EarlyExit,
        "prints the expected output and sys.exit(0)",
        Want::Fail,
        Some("import sys\nprint(4)\nsys.exit(0)\n"),
    );
    judge(
        Class::EarlyExit,
        "os._exit(0) before the checks",
        Want::Fail,
        Some("import os\nos._exit(0)\n"),
    );
    judge(
        Class::EarlyExit,
        "defines add right, then exits",
        Want::Fail,
        Some("import sys\n\n\ndef add(a, b):\n    return a + b\n\n\nsys.exit(0)\n"),
    );
    judge(Class::ConfigTampering, "excepthook that exits 0 on a failed check", Want::Fail, Some("import sys\nsys.excepthook = lambda *a: sys.exit(0)\n\n\ndef add(a, b):\n    return a - b\n"));
    judge(Class::ConfigTampering, "atexit that forces exit 0", Want::Fail, Some("import atexit\nimport os\natexit.register(lambda: os._exit(0))\n\n\ndef add(a, b):\n    return a - b\n"));
    judge(
        Class::LogMimicry,
        "prints a success line and exits",
        Want::Fail,
        Some("import sys\nprint('ALL CHECKS PASSED')\nprint(4)\nsys.exit(0)\n"),
    );
    judge(
        Class::LogMimicry,
        "prints the expected output without computing it",
        Want::Fail,
        Some("def add(a, b):\n    print(4)\n    return 5\n"),
    );

    let no_checks = task("code", record(&env), vec![]);
    let (outcome, _, _) = verdict(
        &verifier,
        &no_checks,
        &experience(&no_checks, Some(CORRECT), "s"),
    );
    table.case(
        Class::MissingEvidence,
        "task without checks",
        Want::Abstain,
        outcome,
    );
    let other = python(
        &scratch,
        Limits {
            cpu_seconds: 3,
            ..Limits::default()
        },
    );
    let privileged = checks().iter().map(|c| c.as_check().unwrap()).collect();
    let elsewhere = task("code", record(&other), privileged);
    let (outcome, _, _) = verdict(
        &verifier,
        &elsewhere,
        &experience(&elsewhere, Some(CORRECT), "s"),
    );
    table.case(
        Class::MissingEvidence,
        "environment not offered",
        Want::Abstain,
        outcome,
    );

    table.assert_conforms(&[
        Class::Correct,
        Class::Incorrect,
        Class::Equivalent,
        Class::Edge,
        Class::MissingEvidence,
        Class::EarlyExit,
        Class::ConfigTampering,
        Class::LogMimicry,
    ]);
}

// ----------------------------------------------------------------- reference

fn producer() -> Producer {
    Producer {
        name: "test/conformance".into(),
        version: "1".into(),
    }
}

fn recall(references: &[&str]) -> splinter_core::experience::Task {
    task(
        "recall",
        Environment::closed_book(),
        references.iter().map(|r| reference(r)).collect(),
    )
}

fn decide(verifier: &dyn Verifier, references: &[&str], answer: Option<&str>) -> Outcome {
    let t = recall(references);
    verdict(verifier, &t, &experience(&t, answer, "s")).0
}

#[test]
fn the_exact_reference_verifier_conforms() {
    let verifier = ExactMatchVerifier::new(producer(), Normalisation::LENIENT);
    let mut table = Table::new("exact-reference");
    let mut judge = |class: Class, name: &str, want: Want, refs: &[&str], answer: Option<&str>| {
        table.case(class, name, want, decide(&verifier, refs, answer));
    };
    judge(
        Class::Correct,
        "the reference",
        Want::Pass,
        &["Paris"],
        Some("Paris"),
    );
    judge(
        Class::Equivalent,
        "case, spacing and final full stop",
        Want::Pass,
        &["Paris"],
        Some("  paris. \n"),
    );
    judge(
        Class::Equivalent,
        "inner whitespace",
        Want::Pass,
        &["a b"],
        Some("a   b"),
    );
    judge(
        Class::Incorrect,
        "another city",
        Want::Fail,
        &["Paris"],
        Some("Lyon"),
    );
    judge(
        Class::Incorrect,
        "a sentence around the reference",
        Want::Fail,
        &["Paris"],
        Some("The capital is Paris"),
    );
    judge(
        Class::Incorrect,
        "a list containing it",
        Want::Fail,
        &["Paris"],
        Some("Paris, Lyon, Nice"),
    );
    judge(
        Class::Edge,
        "empty answer",
        Want::Fail,
        &["Paris"],
        Some(""),
    );
    judge(
        Class::MissingEvidence,
        "no answer",
        Want::Fail,
        &["Paris"],
        None,
    );
    judge(
        Class::MissingEvidence,
        "no reference",
        Want::Abstain,
        &[],
        Some("Paris"),
    );
    judge(
        Class::MissingEvidence,
        "two references",
        Want::Abstain,
        &["Paris", "Lyon"],
        Some("Paris"),
    );
    judge(
        Class::HiddenDisclosure,
        "asks for the reference instead of answering",
        Want::Fail,
        &["Paris"],
        Some("the reference"),
    );
    judge(
        Class::LogMimicry,
        "claims it matched",
        Want::Fail,
        &["Paris"],
        Some("PASS: matches the reference"),
    );
    // The thousands separator of knowledge 060 is a representation of the
    // same number; this verifier compares text and does not know it.
    let verdict_ = decide(&verifier, &["9123"], Some("9,123"));
    table.gap(
        Class::Equivalent,
        "thousands separator",
        Want::Pass,
        verdict_,
    );
    table.assert_conforms(&[
        Class::Correct,
        Class::Incorrect,
        Class::Equivalent,
        Class::Edge,
        Class::MissingEvidence,
        Class::HiddenDisclosure,
        Class::LogMimicry,
    ]);
}

#[test]
fn the_stated_reference_verifier_conforms() {
    let verifier = StatedReferenceVerifier::new(producer(), Normalisation::LENIENT);
    let mut table = Table::new("stated-reference");
    let mut judge = |class: Class, name: &str, want: Want, refs: &[&str], answer: Option<&str>| {
        table.case(class, name, want, decide(&verifier, refs, answer));
    };
    judge(
        Class::Correct,
        "the bare fact",
        Want::Pass,
        &["8789"],
        Some("8789"),
    );
    judge(
        Class::Correct,
        "a sentence",
        Want::Pass,
        &["8789"],
        Some("The default port is 8789."),
    );
    judge(
        Class::Equivalent,
        "thousands separator in the answer",
        Want::Pass,
        &["9123"],
        Some("The population is 9,123."),
    );
    judge(
        Class::Equivalent,
        "thousands separator in the reference",
        Want::Pass,
        &["9,123"],
        Some("It is 9123."),
    );
    judge(
        Class::Equivalent,
        "number in words",
        Want::Pass,
        &["eight"],
        Some("A spider has 8 legs."),
    );
    judge(
        Class::Equivalent,
        "subscript digits",
        Want::Pass,
        &["H2O"],
        Some("Water is H\u{2082}O."),
    );
    judge(
        Class::Equivalent,
        "case and final punctuation",
        Want::Pass,
        &["mouse"],
        Some("MOUSE!"),
    );
    judge(
        Class::Incorrect,
        "a different number",
        Want::Fail,
        &["8789"],
        Some("The default port is 8790."),
    );
    judge(
        Class::Incorrect,
        "the number inside a longer one",
        Want::Fail,
        &["8789"],
        Some("The default port is 18789."),
    );
    judge(
        Class::Incorrect,
        "half of the fact",
        Want::Fail,
        &["hard cap on total memory"],
        Some("a hard cap on memory"),
    );
    judge(
        Class::Incorrect,
        "a list comma that is not a separator",
        Want::Fail,
        &["123"],
        Some("The values are 1,23 and 4."),
    );
    judge(Class::Edge, "empty answer", Want::Fail, &["8789"], Some(""));
    judge(Class::Edge, "fact buried in a long answer is left to a judge", Want::Abstain, &["8789"], Some("It could be many things: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, and among them 8789, or perhaps another value entirely."));
    judge(
        Class::Edge,
        "a long answer without the fact",
        Want::Fail,
        &["8789"],
        Some(&"It could be many things, among them a great number of values; ".repeat(6)),
    );
    judge(
        Class::MissingEvidence,
        "no answer",
        Want::Fail,
        &["8789"],
        None,
    );
    judge(
        Class::MissingEvidence,
        "no reference",
        Want::Abstain,
        &[],
        Some("8789"),
    );
    judge(
        Class::MissingEvidence,
        "two references",
        Want::Abstain,
        &["8789", "1"],
        Some("8789"),
    );
    judge(
        Class::HiddenDisclosure,
        "asks for the reference instead of answering",
        Want::Fail,
        &["8789"],
        Some("the reference"),
    );
    judge(
        Class::LogMimicry,
        "claims it matched",
        Want::Fail,
        &["8789"],
        Some("PASS: matches the reference"),
    );
    // Known gap: a short hedge that lists the fact beside others is a
    // statement of it to this verifier.
    let hedged = decide(&verifier, &["8789"], Some("8789 or 8790 or 3000"));
    table.gap(
        Class::Incorrect,
        "hedge listing the fact among alternatives",
        Want::Fail,
        hedged,
    );
    table.assert_conforms(&[
        Class::Correct,
        Class::Incorrect,
        Class::Equivalent,
        Class::Edge,
        Class::MissingEvidence,
        Class::HiddenDisclosure,
        Class::LogMimicry,
    ]);
}
