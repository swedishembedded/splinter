// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements executable evaluation of code changes
// against a project's own test suite, for its clients. If your team needs
// expertise in test-driven benchmarks for code-editing agents, you can
// procure our services by sending an email to info@swedishembedded.com.

//! What the workspace specs share: a small Python project in a git
//! repository, scratch directories, and the host tools and artifact store a
//! captured tree needs.

// Each spec binary uses a subset of these helpers, and a helper outside a
// #[test] fn unwraps too: a panic is the failure report.
#![allow(dead_code, clippy::unwrap_used)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use splinter_core::clock::FixedClock;
use splinter_knowledge::capture::{capture_tree, CapturedTree, DirtyPolicy, TreeRequest};
use splinter_knowledge::tasks::workspace::{
    Held, RepositoryProfile, WorkspaceGeneration, WorkspaceGenerator, WorkspacePlan,
};
use splinter_sandbox::host::DEFAULT_SEARCH_PATH;
use splinter_sandbox::WorkspaceHost;
use splinter_sandbox::{Limits, ProcessSandbox};
use splinter_store::artifacts::ArtifactStore;
use splinter_store::workspace::Workspace;
use splinter_store::StateRoot;

pub const AT: &str = "2026-10-13T08:00:00.000Z";

pub const PRICING: &str = "\
def total(prices):
    \"\"\"Sum of prices.\"\"\"
    result = 0
    for price in prices:
        result += price
    return result


def discount(price, percent):
    if percent > 100:
        raise ValueError(\"percent over 100\")
    return price - price * percent // 100
";

pub const STATUS: &str = "\
def is_ready(flag):
    verbose = False
    return bool(flag)
";

pub const TEST_PRICING: &str = "\
import unittest

from shop.pricing import discount, total


class TotalTest(unittest.TestCase):
    def test_empty_basket_costs_nothing(self):
        self.assertEqual(total([]), 0)

    def test_prices_are_added_up(self):
        self.assertEqual(total([2, 3]), 5)


class DiscountTest(unittest.TestCase):
    def test_ten_percent_off(self):
        self.assertEqual(discount(200, 10), 180)

    def test_over_hundred_percent_is_refused(self):
        with self.assertRaises(ValueError):
            discount(10, 101)
";

pub const TEST_STATUS: &str = "\
import unittest

from shop.status import is_ready


class StatusTest(unittest.TestCase):
    def test_a_truthy_flag_is_ready(self):
        self.assertTrue(is_ready(1))
";

/// Lines the mutations change; none may appear in a request.
pub const CHANGED_LINES: [&str; 6] = [
    "result = 0",
    "result += price",
    "percent > 100",
    "price * percent // 100",
    "return bool(flag)",
    "verbose = False",
];

pub struct Scratch(pub PathBuf);

impl Scratch {
    pub fn new(name: &str) -> Self {
        static N: AtomicU32 = AtomicU32::new(0);
        let path = std::env::temp_dir().join(format!(
            "splinter-wst-{name}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

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
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, text).unwrap();
}

/// A small Python project in a git repository.
pub fn project(dir: &Path, extra: &[(&str, &str)]) {
    let files = [
        (
            "pyproject.toml",
            "[project]\nname = \"shop\"\nversion = \"1\"\n",
        ),
        ("shop/__init__.py", ""),
        ("shop/pricing.py", PRICING),
        ("shop/status.py", STATUS),
        ("tests/__init__.py", ""),
        ("tests/test_pricing.py", TEST_PRICING),
        ("tests/test_status.py", TEST_STATUS),
    ];
    let all: Vec<(&str, &str)> = files.iter().chain(extra).copied().collect();
    project_of(dir, &all);
}

/// A git repository holding exactly `files`, committed.
pub fn project_of(dir: &Path, files: &[(&str, &str)]) {
    fs::create_dir_all(dir).unwrap();
    git(dir, &["init", "-q", "-b", "main"]);
    for (path, text) in files {
        write(&dir.join(path), text);
    }
    git(dir, &["add", "-A", "-f"]);
    git(dir, &["commit", "-q", "-m", "shop"]);
}

/// A project in a git repository, with the host tools and the artifact
/// store its snapshots are kept in.
pub struct Repo {
    pub scratch: Scratch,
    pub store: ArtifactStore,
    pub host: Arc<WorkspaceHost>,
    pub repo: PathBuf,
}

impl Repo {
    pub fn new(name: &str, extra: &[(&str, &str)]) -> Self {
        Self::laid(name, |repo| project(repo, extra))
    }

    /// A repository holding exactly `files`.
    pub fn of_files(name: &str, files: &[(&str, &str)]) -> Self {
        Self::laid(name, |repo| project_of(repo, files))
    }

    fn laid(name: &str, lay: impl FnOnce(&Path)) -> Self {
        let scratch = Scratch::new(name);
        let repo = scratch.0.join("repo");
        lay(&repo);
        let root = StateRoot::new(scratch.0.join("state"));
        let store = ArtifactStore::new(&Workspace::at(&root), &root);
        let scratch_root = scratch.0.join("host");
        fs::create_dir_all(&scratch_root).unwrap();
        let host = Arc::new(WorkspaceHost::resolve(DEFAULT_SEARCH_PATH, scratch_root).unwrap());
        Self {
            scratch,
            store,
            host,
            repo,
        }
    }

    pub fn request<'a>(&'a self, revision: &'a str) -> TreeRequest<'a> {
        TreeRequest {
            repository: &self.repo,
            revision,
            dirty: DirtyPolicy::Refuse,
            terms: None,
        }
    }

    pub fn capture(&self, revision: &str) -> CapturedTree {
        self.try_capture(&self.request(revision)).unwrap()
    }

    pub fn try_capture(
        &self,
        request: &TreeRequest<'_>,
    ) -> Result<CapturedTree, splinter_knowledge::capture::TreeError> {
        capture_tree(
            request,
            &self.host,
            &self.store,
            &self.scratch.0.join("capture"),
            &FixedClock::new(AT),
        )
    }
}

impl Repo {
    /// The tasks generated from the repository's `HEAD`, considering
    /// `mutants_per_file` mutants of each file under test.
    pub fn generate(&self, mutants_per_file: usize) -> WorkspaceGeneration {
        let calls = self.scratch.0.join("calls");
        fs::create_dir_all(&calls).unwrap();
        let sandbox = ProcessSandbox::new(&calls, [], &[], Limits::default());
        let generator = WorkspaceGenerator::new(
            Arc::new(sandbox),
            self.host.clone(),
            self.store.clone(),
            self.scratch.0.join("work"),
        )
        .with_plan(WorkspacePlan {
            mutants_per_file,
            ..WorkspacePlan::default()
        })
        .unwrap();
        let tree = self.capture("HEAD");
        let laid = self.scratch.0.join("laid");
        generator
            .environment(&tree.snapshot)
            .materialize(&tree.archive_path, &laid)
            .unwrap();
        let profile = RepositoryProfile::detect(&laid).unwrap();
        let qualified = generator.qualify(&profile, &tree.snapshot).unwrap();
        generator
            .from_mutations(
                &tree.snapshot,
                &qualified,
                &Held::none(),
                &FixedClock::new(AT),
            )
            .unwrap()
    }

    /// The files of the solver's tree of `task`, laid out under `dir`.
    pub fn solver_tree(
        &self,
        task: &splinter_knowledge::tasks::workspace::WorkspaceTask,
        dir: &Path,
    ) {
        let _ = fs::remove_dir_all(dir);
        let calls = self.scratch.0.join("calls");
        let sandbox = ProcessSandbox::new(&calls, [], &[], Limits::default());
        let generator = WorkspaceGenerator::new(
            Arc::new(sandbox),
            self.host.clone(),
            self.store.clone(),
            self.scratch.0.join("work"),
        );
        generator
            .environment(&task.snapshot)
            .materialize(&generator.archive_of(&task.snapshot).unwrap(), dir)
            .unwrap();
    }
}
