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

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use common::{experience, task, verdict, Scratch};
use splinter_core::annotation::{Outcome, Strength};
use splinter_core::digest::Digest;
use splinter_core::experience::Task;
use splinter_eval::verifiers::workspace::{ArchiveSource, WorkspaceCheck, WorkspaceVerifier};
use splinter_eval::verifiers::VerifyError;
use splinter_sandbox::host::DEFAULT_SEARCH_PATH;
use splinter_sandbox::{
    CommandCall, Limits, ProcessSandbox, ResolvedEnvironment, WorkspaceEnvironment, WorkspaceHost,
};

const CALC_BUGGY: &str = "def add(a, b):\n    return a - b\n\n\ndef neg(a):\n    return -a\n";
const CALC_FIXED: &str = "def add(a, b):\n    return a + b\n\n\ndef neg(a):\n    return -a\n";
const TEST_KEEP: &str = "import unittest\n\nfrom calc import neg\n\n\nclass Keep(unittest.TestCase):\n    def test_neg(self):\n        self.assertEqual(neg(4), -4)\n";
const TEST_HIDDEN: &str = "import unittest\n\nfrom calc import add\n\n\nclass Hidden(unittest.TestCase):\n    def test_secret_sum_of_twelve(self):\n        self.assertEqual(add(5, 7), 12)\n";
const TEST_WEAKENED: &str = "import unittest\n\n\nclass Hidden(unittest.TestCase):\n    def test_secret_sum_of_twelve(self):\n        self.assertTrue(True)\n";

/// Runs `git` with a fixed identity and no host configuration.
fn git(dir: &Path, args: &[&str]) -> String {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(["-c", "user.name=t", "-c", "user.email=t@example.com"])
        .args(["-c", "commit.gpgsign=false", "-c", "core.autocrlf=false"])
        .args(args)
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_CONFIG_SYSTEM", "/dev/null")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

fn write(path: &Path, text: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

/// Archives resolved from a fixed table of digest to file.
struct Table(BTreeMap<Digest, PathBuf>);

impl ArchiveSource for Table {
    fn path(&self, digest: &Digest) -> Result<PathBuf, VerifyError> {
        self.0
            .get(digest)
            .cloned()
            .ok_or_else(|| VerifyError::Parameter {
                name: "archive",
                reason: format!("no archive {digest}"),
            })
    }
}

/// A small Python project with a bug, a visible passing test, and a hidden
/// test that fails until the bug is fixed.
struct Fixture {
    root: Scratch,
    env: WorkspaceEnvironment,
    archives: Arc<Table>,
    hidden: Digest,
    /// The change that fixes the bug, as `git diff` writes it.
    reference_patch: String,
}

impl Fixture {
    /// `keep_hidden_in_snapshot`: the snapshot also holds the hidden test
    /// (a generator that failed to remove it), so a patch can delete it.
    fn new(keep_hidden_in_snapshot: bool) -> Self {
        static SEQ: AtomicUsize = AtomicUsize::new(0);
        let root = Scratch::new(&format!("ws-{}", SEQ.fetch_add(1, Ordering::Relaxed)));
        let origin = root.0.join("origin");
        std::fs::create_dir_all(&origin).unwrap();
        git(&origin, &["init", "-q", "-b", "main"]);
        write(&origin.join("calc.py"), CALC_BUGGY);
        write(&origin.join("test_keep.py"), TEST_KEEP);
        if keep_hidden_in_snapshot {
            write(&origin.join("test_hidden.py"), TEST_HIDDEN);
        }
        git(&origin, &["add", "-A", "-f"]);
        git(&origin, &["commit", "-q", "-m", "origin"]);
        let snapshot_tar = root.0.join("snapshot.tar");
        git(
            &origin,
            &[
                "archive",
                "--format=tar",
                "-o",
                snapshot_tar.to_str().unwrap(),
                "HEAD",
            ],
        );
        write(&origin.join("calc.py"), CALC_FIXED);
        let reference_patch = git(&origin, &["diff"]);

        let overlay = root.0.join("overlay");
        write(&overlay.join("test_hidden.py"), TEST_HIDDEN);
        let hidden_tar = root.0.join("hidden.tar");
        let made = Command::new("tar")
            .args(["--create", "--file"])
            .arg(&hidden_tar)
            .arg("--directory")
            .arg(&overlay)
            .arg("test_hidden.py")
            .status()
            .unwrap();
        assert!(made.success());

        let host = Arc::new(WorkspaceHost::resolve(DEFAULT_SEARCH_PATH, &root.0).unwrap());
        let snapshot = host.snapshot(&snapshot_tar).unwrap();
        let calls = root.0.join("calls");
        std::fs::create_dir_all(&calls).unwrap();
        let sandbox = ProcessSandbox::new(&calls, BTreeMap::new(), &[], Limits::default());
        let env = WorkspaceEnvironment::new(snapshot.clone(), Arc::new(sandbox), host);
        let hidden = Digest::of(&std::fs::read(&hidden_tar).unwrap());
        let archives = Arc::new(Table(BTreeMap::from([
            (snapshot.digest, snapshot_tar),
            (hidden.clone(), hidden_tar),
        ])));
        Self {
            root,
            env,
            archives,
            hidden,
            reference_patch,
        }
    }

    fn verifier(&self) -> WorkspaceVerifier {
        WorkspaceVerifier::new(
            vec![self.env.clone()],
            self.archives.clone(),
            self.root.0.join("verify"),
        )
    }

    fn check(&self) -> WorkspaceCheck {
        let unittest = |module: &str| CommandCall {
            argv: ["python3", "-m", "unittest", module]
                .map(String::from)
                .to_vec(),
            stdin: None,
        };
        WorkspaceCheck {
            fail_to_pass: vec![unittest("test_hidden")],
            pass_to_pass: vec![unittest("test_keep")],
            hidden: Some(self.hidden.clone()),
            protected: vec!["test_keep.py".into()],
            hidden_paths: vec!["test_hidden.py".into()],
        }
    }

    fn task(&self, check: &WorkspaceCheck) -> Task {
        let environment = ResolvedEnvironment::Workspace(self.env.clone())
            .record()
            .unwrap();
        task(
            "workspace-change",
            environment,
            vec![check.as_check().unwrap()],
        )
    }
}

fn patch_replacing(path: &str, old: &str, new: &str) -> String {
    let old_lines: String = old.lines().map(|l| format!("-{l}\n")).collect();
    let new_lines: String = new.lines().map(|l| format!("+{l}\n")).collect();
    format!(
        "diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1,{} +1,{} @@\n{old_lines}{new_lines}",
        old.lines().count(),
        new.lines().count()
    )
}

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
