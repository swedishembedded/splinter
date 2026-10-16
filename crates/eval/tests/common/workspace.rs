// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verifiers that grade code changes by
// running a project's own tests in a clean tree, for its clients. If your
// team needs expertise in executable evaluation of code-editing agents, you
// can procure our services by sending an email to info@swedishembedded.com.

//! A small Python project with a bug, a visible passing test and a hidden
//! test that fails until the bug is fixed, in a workspace environment: what
//! the workspace verifier specs and the conformance suite grade patches
//! against.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used, dead_code)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use super::{task, Scratch};
use splinter_core::digest::Digest;
use splinter_core::experience::Task;
use splinter_eval::verifiers::test_report::{TestExpectation, TestRunner};
use splinter_eval::verifiers::workspace::{
    ArchiveSource, Phase, WorkspaceCheck, WorkspaceVerifier,
};
use splinter_eval::verifiers::VerifyError;
use splinter_sandbox::host::DEFAULT_SEARCH_PATH;
use splinter_sandbox::{
    CommandCall, Limits, ProcessSandbox, ResolvedEnvironment, WorkspaceEnvironment, WorkspaceHost,
};

pub const CALC_BUGGY: &str = "def add(a, b):\n    return a - b\n\n\ndef neg(a):\n    return -a\n";
pub const CALC_FIXED: &str = "def add(a, b):\n    return a + b\n\n\ndef neg(a):\n    return -a\n";
pub const TEST_KEEP: &str = "import unittest\n\nfrom calc import neg\n\n\nclass Keep(unittest.TestCase):\n    def test_neg(self):\n        self.assertEqual(neg(4), -4)\n";
pub const TEST_HIDDEN: &str = "import unittest\n\nfrom calc import add\n\n\nclass Hidden(unittest.TestCase):\n    def test_secret_sum_of_twelve(self):\n        self.assertEqual(add(5, 7), 12)\n";
pub const CONFTEST: &str = "# runner configuration the solver may not change\n";
pub const HIDDEN_ID: &str = "test_hidden.Hidden.test_secret_sum_of_twelve";
pub const KEEP_ID: &str = "test_keep.Keep.test_neg";
/// The hidden module with its one test replaced by a test that passes.
pub const TEST_BYPASSED: &str = "import unittest\n\n\nclass Hidden(unittest.TestCase):\n    def test_other(self):\n        self.assertTrue(True)\n";
pub const TEST_WEAKENED: &str = "import unittest\n\n\nclass Hidden(unittest.TestCase):\n    def test_secret_sum_of_twelve(self):\n        self.assertTrue(True)\n";

/// Runs `git` with a fixed identity and no host configuration.
pub fn git(dir: &Path, args: &[&str]) -> String {
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

pub fn write(path: &Path, text: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

/// Archives resolved from a fixed table of digest to file.
pub struct Table(BTreeMap<Digest, PathBuf>);

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
pub struct Fixture {
    pub root: Scratch,
    pub env: WorkspaceEnvironment,
    pub archives: Arc<Table>,
    pub hidden: Digest,
    /// The change that fixes the bug, as `git diff` writes it.
    pub reference_patch: String,
}

impl Fixture {
    /// `keep_hidden_in_snapshot`: the snapshot also holds the hidden test
    /// (a generator that failed to remove it), so a patch can delete it.
    pub fn new(keep_hidden_in_snapshot: bool) -> Self {
        static SEQ: AtomicUsize = AtomicUsize::new(0);
        let root = Scratch::new(&format!("ws-{}", SEQ.fetch_add(1, Ordering::Relaxed)));
        let origin = root.0.join("origin");
        std::fs::create_dir_all(&origin).unwrap();
        git(&origin, &["init", "-q", "-b", "main"]);
        write(&origin.join("calc.py"), CALC_BUGGY);
        write(&origin.join("test_keep.py"), TEST_KEEP);
        write(&origin.join("conftest.py"), CONFTEST);
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

    pub fn verifier(&self) -> WorkspaceVerifier {
        WorkspaceVerifier::new(
            vec![self.env.clone()],
            self.archives.clone(),
            self.root.0.join("verify"),
        )
    }

    pub fn check(&self) -> WorkspaceCheck {
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
            hidden_strength: None,
            expectations: vec![
                TestExpectation {
                    phase: Phase::FailToPass,
                    index: 0,
                    runner: TestRunner::Unittest,
                    tests: [HIDDEN_ID.to_string()].into(),
                    min_tests: 1,
                },
                TestExpectation {
                    phase: Phase::PassToPass,
                    index: 0,
                    runner: TestRunner::Unittest,
                    tests: [KEEP_ID.to_string()].into(),
                    min_tests: 1,
                },
            ],
            pinned: Vec::new(),
        }
    }

    pub fn task(&self, check: &WorkspaceCheck) -> Task {
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

pub fn patch_replacing(path: &str, old: &str, new: &str) -> String {
    let old_lines: String = old.lines().map(|l| format!("-{l}\n")).collect();
    let new_lines: String = new.lines().map(|l| format!("+{l}\n")).collect();
    format!(
        "diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1,{} +1,{} @@\n{old_lines}{new_lines}",
        old.lines().count(),
        new.lines().count()
    )
}

/// Source that makes the process look like a runner that finished well:
/// printed at import of the code under test, which every test run imports.
pub fn pretending(report: &str, stream: &str) -> String {
    format!(
        "import os\nimport sys\n\nsys.{stream}.write({report:?})\nsys.{stream}.flush()\nos._exit(0)\n\n{CALC_BUGGY}"
    )
}
