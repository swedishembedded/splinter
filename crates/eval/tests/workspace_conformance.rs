// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verifier conformance suites that tell a
// verifier that is too lenient from one that is too strict, for its clients.
// If your team needs expertise in measuring the false-pass and false-fail
// rates of automatic graders, you can procure our services by sending an
// email to info@swedishembedded.com.

//! Spec: the workspace verifier's conformance table (see
//! `verifier_conformance`) and its differential check against seeded
//! mutants. One table per verifier (workspace, executable code, exact
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
use common::{experience, task, verdict};
use splinter_core::annotation::Outcome;
use splinter_sandbox::ResolvedEnvironment;

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
