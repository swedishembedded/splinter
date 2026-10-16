// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verifiers that grade code changes by
// running a project's own tests in a clean tree, for its clients. If your
// team needs expertise in executable evaluation of code-editing agents, you
// can procure our services by sending an email to info@swedishembedded.com.

//! Spec: the workspace verifier lays the task's snapshot out in a fresh
//! directory, applies the candidate patch, lays the hidden tests over the
//! result and runs the fail-to-pass and pass-to-pass commands. The reference
//! patch passes; an empty patch, a patch that does not make the failing
//! tests pass, a patch that touches a protected or hidden path, and a patch
//! that weakens a hidden test all fail; an environment it was not offered is
//! an abstention; the evidence holds digests, never test text.
//!
//! `grade` is the same judgement for a caller that needs more than a
//! verdict: each command's own result, and the files the change touched.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::path::PathBuf;

use common::workspace::{
    patch_replacing, pretending, Fixture, CALC_BUGGY, CONFTEST, HIDDEN_ID, TEST_BYPASSED,
    TEST_HIDDEN, TEST_KEEP, TEST_WEAKENED,
};
use common::{experience, task, verdict};
use splinter_core::annotation::{Outcome, Strength};
use splinter_eval::verifiers::workspace::{grade, Phase, Refusal, WorkspaceCheck};
use splinter_eval::verifiers::workspace_canary::canary_task;
use splinter_sandbox::{Patch, ResolvedEnvironment};

#[test]
fn the_reference_patch_passes_and_the_evidence_holds_digests_only() {
    let fx = Fixture::new(false);
    let check = fx.check();
    let task = fx.task(&check);
    let exp = experience(&task, Some(&fx.reference_patch), "s");
    let (outcome, strength, evidence) = verdict(&fx.verifier(), &task, &exp);
    assert_eq!(outcome, Outcome::Pass, "{evidence}");
    assert_eq!(strength, Strength::Executable);
    let commands = evidence["commands"].as_array().unwrap();
    assert_eq!(commands.len(), 2);
    for run in commands {
        assert_eq!(run["exit_code"], 0);
        assert!(run["command"].as_str().unwrap().starts_with("blake3:"));
        assert!(run["stdout"].as_str().unwrap().starts_with("blake3:"));
        assert!(run["stderr"].as_str().unwrap().starts_with("blake3:"));
    }
    assert_eq!(evidence["hidden"], fx.hidden.as_str());
    let text = evidence.to_string();
    for secret in ["test_secret_sum_of_twelve", "assertEqual", "add(5, 7)"] {
        assert!(!text.contains(secret), "evidence leaks {secret}: {text}");
    }
}

#[test]
fn an_empty_or_missing_patch_fails() {
    let fx = Fixture::new(false);
    let task = fx.task(&fx.check());
    for output in [Some(""), Some("  \n"), None] {
        let (outcome, _, evidence) =
            verdict(&fx.verifier(), &task, &experience(&task, output, "s"));
        assert_eq!(outcome, Outcome::Fail, "{output:?}: {evidence}");
    }
}

#[test]
fn a_patch_that_does_not_apply_or_does_not_fix_the_bug_fails() {
    let fx = Fixture::new(false);
    let task = fx.task(&fx.check());
    let stale = patch_replacing("calc.py", "def add(a, b):\n    return a * b\n", "x\n");
    let unrelated = patch_replacing(
        "calc.py",
        "def neg(a):\n    return -a\n",
        "def neg(a):\n    return 0 - a\n",
    );
    for output in [stale, unrelated] {
        let (outcome, _, evidence) = verdict(
            &fx.verifier(),
            &task,
            &experience(&task, Some(&output), "s"),
        );
        assert_eq!(outcome, Outcome::Fail, "{evidence}");
    }
}

#[test]
fn a_patch_touching_a_protected_or_hidden_path_fails_even_when_it_fixes_the_bug() {
    let fx = Fixture::new(false);
    let task = fx.task(&fx.check());
    let protected = format!(
        "{}{}",
        fx.reference_patch,
        patch_replacing("test_keep.py", TEST_KEEP, "import unittest\n")
    );
    let hidden = format!(
        "{}diff --git a/test_hidden.py b/test_hidden.py\nnew file mode 100644\n--- /dev/null\n+++ b/test_hidden.py\n@@ -0,0 +1 @@\n+import unittest\n",
        fx.reference_patch
    );
    for output in [protected, hidden] {
        let (outcome, _, evidence) = verdict(
            &fx.verifier(),
            &task,
            &experience(&task, Some(&output), "s"),
        );
        assert_eq!(outcome, Outcome::Fail, "{evidence}");
        assert!(evidence["refused"].is_string(), "{evidence}");
        assert!(!evidence.to_string().contains("test_hidden"), "{evidence}");
    }
}

#[test]
fn a_patch_that_deletes_the_failing_test_still_fails() {
    let fx = Fixture::new(true);
    let task = fx.task(&fx.check());
    let delete = format!(
        "{}diff --git a/test_hidden.py b/test_hidden.py\ndeleted file mode 100644\n--- a/test_hidden.py\n+++ /dev/null\n@@ -1,{n} +0,0 @@\n{lines}",
        "",
        n = TEST_HIDDEN.lines().count(),
        lines = TEST_HIDDEN.lines().map(|l| format!("-{l}\n")).collect::<String>()
    );
    let (outcome, _, evidence) = verdict(
        &fx.verifier(),
        &task,
        &experience(&task, Some(&delete), "s"),
    );
    assert_eq!(outcome, Outcome::Fail, "{evidence}");
}

#[test]
fn the_hidden_tests_are_laid_over_the_patched_tree() {
    // The hidden path is not declared, so nothing refuses the patch: only
    // the overlay stands between a weakened test and a pass.
    let fx = Fixture::new(true);
    let mut check = fx.check();
    check.hidden_paths.clear();
    let task = fx.task(&check);
    let weakened = patch_replacing("test_hidden.py", TEST_HIDDEN, TEST_WEAKENED);
    let (outcome, _, evidence) = verdict(
        &fx.verifier(),
        &task,
        &experience(&task, Some(&weakened), "s"),
    );
    assert_eq!(outcome, Outcome::Fail, "{evidence}");

    let with_fix = format!("{}{weakened}", fx.reference_patch);
    let (outcome, _, evidence) = verdict(
        &fx.verifier(),
        &task,
        &experience(&task, Some(&with_fix), "s"),
    );
    assert_eq!(outcome, Outcome::Pass, "{evidence}");
}

#[test]
fn an_environment_that_was_not_offered_is_an_abstention() {
    let fx = Fixture::new(false);
    let other = Fixture::new(true);
    let task = other.task(&other.check());
    let (outcome, _, evidence) = verdict(
        &fx.verifier(),
        &task,
        &experience(&task, Some(&other.reference_patch), "s"),
    );
    assert_eq!(outcome, Outcome::Abstain, "{evidence}");
}

#[test]
fn a_task_without_a_workspace_check_is_an_abstention() {
    let fx = Fixture::new(false);
    let environment = ResolvedEnvironment::Workspace(fx.env.clone())
        .record()
        .unwrap();
    let task = task("workspace-change", environment, vec![]);
    let (outcome, _, _) = verdict(
        &fx.verifier(),
        &task,
        &experience(&task, Some(&fx.reference_patch), "s"),
    );
    assert_eq!(outcome, Outcome::Abstain);
}

fn patch_of(text: &str, files: &[&str]) -> Patch {
    Patch {
        diff: text.as_bytes().to_vec(),
        files: files.iter().map(PathBuf::from).collect(),
    }
}

#[test]
fn grading_reports_each_command_with_its_result_and_the_files_the_change_touched() {
    let fx = Fixture::new(false);
    let check = fx.check();
    let scratch = fx.root.0.join("grade");
    let graded = grade(
        &fx.env,
        fx.archives.as_ref(),
        &scratch,
        &check,
        &patch_of(&fx.reference_patch, &["calc.py"]),
    )
    .unwrap();
    assert!(graded.passed(), "{graded:#?}");
    let phases: Vec<Phase> = graded.commands.iter().map(|c| c.phase).collect();
    assert_eq!(phases, [Phase::FailToPass, Phase::PassToPass]);
    assert!(
        graded
            .commands
            .iter()
            .all(|c| c.result.exit_code == Some(0)),
        "each command keeps its own result"
    );
    let touched: Vec<_> = graded.touched.iter().map(|t| t.path.as_str()).collect();
    assert_eq!(touched, ["calc.py"]);
    let calc = &graded.touched[0];
    assert!(calc.before.is_some() && calc.after.is_some() && calc.before != calc.after);
    assert!(
        std::fs::read_dir(&scratch).map_or(true, |mut left| left.next().is_none()),
        "the tree the change was graded in is removed"
    );
}

#[test]
fn grading_an_empty_change_runs_the_commands_on_the_snapshot_as_it_is() {
    let fx = Fixture::new(false);
    let graded = grade(
        &fx.env,
        fx.archives.as_ref(),
        &fx.root.0.join("grade"),
        &fx.check(),
        &patch_of("", &[]),
    )
    .unwrap();
    assert!(graded.refusal.is_none() && graded.touched.is_empty());
    assert!(
        !graded.passed(),
        "the seeded failure is red before any work"
    );
    let first = &graded.commands[0];
    assert!(first.phase == Phase::FailToPass && !first.passed);
    assert!(
        first.result.stderr.contains("AssertionError") || first.result.stdout.contains("Assertion"),
        "{first:#?}"
    );
}

#[test]
fn grading_sets_aside_a_change_that_cannot_apply_or_touches_off_limits_paths() {
    let fx = Fixture::new(false);
    let scratch = fx.root.0.join("grade");
    let stale = patch_replacing("calc.py", "def add(a, b):\n    return a * b\n", "x\n");
    let graded = grade(
        &fx.env,
        fx.archives.as_ref(),
        &scratch,
        &fx.check(),
        &patch_of(&stale, &["calc.py"]),
    )
    .unwrap();
    assert!(matches!(graded.refusal, Some(Refusal::DoesNotApply)));
    assert!(graded.commands.is_empty() && !graded.passed());

    let protected = format!(
        "{}{}",
        fx.reference_patch,
        patch_replacing("test_keep.py", TEST_KEEP, "import unittest\n")
    );
    let graded = grade(
        &fx.env,
        fx.archives.as_ref(),
        &scratch,
        &fx.check(),
        &patch_of(&protected, &["calc.py", "test_keep.py"]),
    )
    .unwrap();
    assert!(
        matches!(&graded.refusal, Some(Refusal::OffLimits(paths)) if paths == &["test_keep.py"]),
        "{graded:#?}"
    );
    assert!(
        graded.commands.is_empty(),
        "no command runs on a fenced change"
    );
}

/// A change to the code under test that tampers with the test runner: the
/// process exits 0 on its first run in a tree and 1 on its second, whatever
/// the tests said, so a command and its negation both "pass". It is the
/// shape of cheat a canary exists to catch.
const CALC_TAMPERING: &str = "import atexit\nimport os\n\n_count = \".runs\"\n_n = len(open(_count).read()) if os.path.exists(_count) else 0\nopen(_count, \"a\").write(\"x\")\natexit.register(lambda: os._exit(_n % 2))\n\n\ndef add(a, b):\n    return a - b\n\n\ndef neg(a):\n    return -a\n";

#[test]
fn a_canary_cannot_be_passed_by_doing_the_work_but_can_by_tampering_with_the_runner() {
    let fx = Fixture::new(false);
    let check = fx.check();
    let real = fx.task(&check);
    let canary = canary_task(&real).unwrap();
    assert_ne!(
        canary.task.id, real.task.id,
        "a canary is a task of its own"
    );
    assert_eq!(
        canary.instruction, real.instruction,
        "it reads like the real task"
    );

    // The honest fix passes the real task and fails its canary: the hidden
    // tests are now required to both pass and fail.
    let honest = experience(&real, Some(&fx.reference_patch), "s");
    assert_eq!(verdict(&fx.verifier(), &real, &honest).0, Outcome::Pass);
    let honest = experience(&canary, Some(&fx.reference_patch), "s");
    assert_eq!(verdict(&fx.verifier(), &canary, &honest).0, Outcome::Fail);
    // Doing nothing fails it too: the original assertion is still required.
    let nothing = experience(&canary, Some(""), "s");
    assert_ne!(verdict(&fx.verifier(), &canary, &nothing).0, Outcome::Pass);

    // The runner cheat no longer passes it: the report shows the fix is
    // absent, whatever the exit status says.
    let cheat = patch_replacing("calc.py", CALC_BUGGY, CALC_TAMPERING);
    for task in [&real, &canary] {
        let (outcome, _, evidence) =
            verdict(&fx.verifier(), task, &experience(task, Some(&cheat), "s"));
        assert_eq!(outcome, Outcome::Fail, "{evidence}");
    }

    // Fixing the code and tampering with the exit status is what still
    // passes the canary: the tests ran and passed, and so did their negation.
    let both = patch_replacing(
        "calc.py",
        CALC_BUGGY,
        &CALC_TAMPERING.replace("a - b", "a + b"),
    );
    let (outcome, _, evidence) = verdict(
        &fx.verifier(),
        &canary,
        &experience(&canary, Some(&both), "s"),
    );
    assert_eq!(outcome, Outcome::Pass, "{evidence}");
}

#[test]
fn a_task_without_a_workspace_check_has_no_canary() {
    let fx = Fixture::new(false);
    let environment = ResolvedEnvironment::Workspace(fx.env.clone())
        .record()
        .unwrap();
    assert!(canary_task(&task("workspace-change", environment, vec![])).is_err());
}

fn failed_with(fx: &Fixture, check: &WorkspaceCheck, patch: &str) -> serde_json::Value {
    let task = fx.task(check);
    let (outcome, _, evidence) =
        verdict(&fx.verifier(), &task, &experience(&task, Some(patch), "s"));
    assert_eq!(outcome, Outcome::Fail, "{evidence}");
    evidence
}

#[test]
fn a_patch_that_prints_a_success_log_and_exits_zero_fails() {
    let fx = Fixture::new(false);
    let check = fx.check();
    let bare = patch_replacing("calc.py", CALC_BUGGY, &pretending("OK\n", "stderr"));
    failed_with(&fx, &check, &bare);

    // Even a full report in the runner's own stream names no hidden test.
    let log = "test_x (test_hidden.Hidden.test_x) ... ok\n\nRan 1 test in 0.001s\n\nOK\n";
    let full = patch_replacing("calc.py", CALC_BUGGY, &pretending(log, "stderr"));
    failed_with(&fx, &check, &full);
}

#[test]
fn a_log_that_names_the_expected_tests_still_fails_when_it_is_not_the_runners() {
    // The names are guessed or leaked; the report is in the program's
    // stdout, which is not where the runner writes it.
    let fx = Fixture::new(false);
    let log =
        format!("test_secret_sum_of_twelve ({HIDDEN_ID}) ... ok\n\nRan 1 test in 0.001s\n\nOK\n");
    let patch = patch_replacing("calc.py", CALC_BUGGY, &pretending(&log, "stdout"));
    let evidence = failed_with(&fx, &fx.check(), &patch);
    assert!(
        evidence.to_string().contains("foreign_report"),
        "the evidence says why: {evidence}"
    );
}

#[test]
fn a_patch_that_bypasses_or_deletes_the_hidden_tests_fails() {
    // The hidden module is in the snapshot and no overlay restores it: the
    // exit status stays 0 when its test is swapped for one that passes.
    let fx = Fixture::new(true);
    let mut check = fx.check();
    check.hidden = None;
    check.hidden_paths.clear();
    let with_fix = |tail: String| format!("{}{tail}", fx.reference_patch);
    let swapped = with_fix(patch_replacing(
        "test_hidden.py",
        TEST_HIDDEN,
        TEST_BYPASSED,
    ));
    let evidence = failed_with(&fx, &check, &swapped);
    assert!(
        evidence.to_string().contains("expected_test_missing"),
        "{evidence}"
    );

    // A deleted module with a command that runs whatever it finds.
    check.fail_to_pass[0].argv = ["python3", "-m", "unittest", "discover", "-p", "test_h*.py"]
        .map(String::from)
        .to_vec();
    let deleted = with_fix(format!(
        "diff --git a/test_hidden.py b/test_hidden.py\ndeleted file mode 100644\n--- a/test_hidden.py\n+++ /dev/null\n@@ -1,{} +0,0 @@\n{}",
        TEST_HIDDEN.lines().count(),
        TEST_HIDDEN.lines().map(|l| format!("-{l}\n")).collect::<String>()
    ));
    failed_with(&fx, &check, &deleted);
}

#[test]
fn a_patch_that_changes_the_runner_configuration_fails_before_anything_runs() {
    let fx = Fixture::new(false);
    let check = fx.check();
    let edited = format!(
        "{}{}",
        fx.reference_patch,
        patch_replacing("conftest.py", CONFTEST, "import sys\nsys.exit(0)\n")
    );
    let added = format!(
        "{}diff --git a/sitecustomize.py b/sitecustomize.py\nnew file mode 100644\n--- /dev/null\n+++ b/sitecustomize.py\n@@ -0,0 +1 @@\n+import os\n",
        fx.reference_patch
    );
    let removed = format!(
        "{}diff --git a/conftest.py b/conftest.py\ndeleted file mode 100644\n--- a/conftest.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-{}",
        fx.reference_patch, CONFTEST
    );
    for patch in [edited, added, removed] {
        let evidence = failed_with(&fx, &check, &patch);
        assert_eq!(
            evidence["refused"], "runner_configuration_changed",
            "{evidence}"
        );
        assert!(evidence["commands"].is_null(), "nothing ran: {evidence}");
    }
}

#[test]
fn a_path_the_check_pins_must_stay_byte_identical() {
    let fx = Fixture::new(false);
    let mut check = fx.check();
    check.pinned = vec!["calc.py".into()];
    let evidence = failed_with(&fx, &check, &fx.reference_patch);
    assert_eq!(
        evidence["refused"], "runner_configuration_changed",
        "{evidence}"
    );
}

#[test]
fn the_evidence_counts_the_tests_observed_and_names_none() {
    let fx = Fixture::new(false);
    let task = fx.task(&fx.check());
    let exp = experience(&task, Some(&fx.reference_patch), "s");
    let (outcome, _, evidence) = verdict(&fx.verifier(), &task, &exp);
    assert_eq!(outcome, Outcome::Pass, "{evidence}");
    for run in evidence["commands"].as_array().unwrap() {
        assert_eq!(run["observed"]["run"], 1, "{run}");
        assert_eq!(run["observed"]["missing"], 0, "{run}");
    }
    assert!(!evidence.to_string().contains("test_neg"), "{evidence}");
}

#[test]
fn a_command_without_expectations_is_graded_by_its_exit_status_and_says_so() {
    let fx = Fixture::new(false);
    let mut check = fx.check();
    check.expectations.clear();
    let task = fx.task(&check);
    let exp = experience(&task, Some(&fx.reference_patch), "s");
    let (outcome, _, evidence) = verdict(&fx.verifier(), &task, &exp);
    assert_eq!(outcome, Outcome::Pass, "{evidence}");
    assert!(evidence["commands"][0]["observed"].is_null(), "{evidence}");
}
