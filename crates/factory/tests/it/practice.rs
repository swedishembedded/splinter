// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training tasks made from a codebase's own
// tests, for its clients. If your team needs expertise in teaching models
// your repositories, you can procure our services by sending an email to
// info@swedishembedded.com.

//! `task.mutate`: a revision of a repository made into Harbor tasks by its
//! own tests. Small faults are seeded into the files under test and each is
//! kept only when the project's tests catch it; whole families (the file a
//! fault is in) are held out for evaluation, so no task trained on shares a
//! file with one examined.

use std::process::Command;

use splinter_factory::run::tasks::tasks_of;

use crate::factory_run::{resolved, run};
use crate::fakes::{FakeBackend, FakeTrainer, Outcome};
use crate::repo::Repo;

const PRICING: &str = "def total(prices):\n    \"\"\"Sum of prices.\"\"\"\n    result = 0\n    for price in prices:\n        result += price\n    return result\n\n\ndef discount(price, percent):\n    if percent > 100:\n        raise ValueError(\"percent over 100\")\n    return price - price * percent // 100\n";
const STATUS: &str = "def is_ready(flag, count):\n    return bool(flag) and count > 0\n";
const TEST_PRICING: &str = "import unittest\n\nfrom shop.pricing import discount, total\n\n\nclass TotalTest(unittest.TestCase):\n    def test_empty_basket_costs_nothing(self):\n        self.assertEqual(total([]), 0)\n\n    def test_prices_are_added_up(self):\n        self.assertEqual(total([2, 3]), 5)\n\n\nclass DiscountTest(unittest.TestCase):\n    def test_ten_percent_off(self):\n        self.assertEqual(discount(200, 10), 180)\n\n    def test_over_hundred_percent_is_refused(self):\n        with self.assertRaises(ValueError):\n            discount(10, 101)\n";
const TEST_STATUS: &str = "import unittest\n\nfrom shop.status import is_ready\n\n\nclass StatusTest(unittest.TestCase):\n    def test_a_truthy_flag_with_work_is_ready(self):\n        self.assertTrue(is_ready(1, 1))\n\n    def test_a_falsy_flag_is_not(self):\n        self.assertFalse(is_ready(0, 1))\n\n    def test_no_work_is_not_ready(self):\n        self.assertFalse(is_ready(1, 0))\n";

fn git(dir: &std::path::Path, args: &[&str]) -> String {
    let out = Command::new("git")
        .args([
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "-c",
            "commit.gpgsign=false",
        ])
        .args(args)
        .current_dir(dir)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap().trim().to_owned()
}

/// A task repository holding a small Python project as a git repository of
/// its own, and the commit it is at.
fn repository() -> (Repo, String) {
    let repo = Repo::new();
    for (path, text) in [
        (
            "project/pyproject.toml",
            "[project]\nname = \"shop\"\nversion = \"1\"\n",
        ),
        ("project/shop/__init__.py", ""),
        ("project/shop/pricing.py", PRICING),
        ("project/shop/status.py", STATUS),
        ("project/tests/__init__.py", ""),
        ("project/tests/test_pricing.py", TEST_PRICING),
        ("project/tests/test_status.py", TEST_STATUS),
    ] {
        repo.write(path, text);
    }
    let project = repo.root.join("project");
    git(&project, &["init", "-q", "-b", "main"]);
    git(&project, &["add", "-A"]);
    git(&project, &["commit", "-q", "-m", "shop"]);
    let commit = git(&project, &["rev-parse", "HEAD"]);
    (repo, commit)
}

#[test]
fn a_repositorys_own_tests_make_its_tasks_and_hold_out_whole_files_for_evaluation() {
    let (repo, commit) = repository();
    // seed: the families held out for evaluation are the seed's draw.
    let recipe = format!(
        "schema: splinter.recipe/v2
name: practised
stages:
  - id: practised
    uses: task.mutate
    with: {{ repository: project, revision: {commit}, max_tasks: 12, eval_percent: 50, seed: 3 }}
  - {{ id: train, uses: tasks.pin, inputs: {{ collection: $stages.practised.tasks }}, with: {{ split: train }} }}
  - {{ id: held, uses: tasks.pin, inputs: {{ collection: $stages.practised.tasks }}, with: {{ split: eval }} }}
"
    );
    let dir = tempfile::tempdir().unwrap();
    let backend = FakeBackend::with(Outcome::Pass);
    run(
        &resolved(&repo, &recipe),
        dir.path(),
        &backend,
        &FakeTrainer::default(),
    )
    .unwrap();
    let (train, held) = (
        tasks_of(&dir.path().join("train")).unwrap(),
        tasks_of(&dir.path().join("held")).unwrap(),
    );
    assert!(
        !train.tasks.is_empty() && !held.tasks.is_empty(),
        "{train:?} {held:?}"
    );
    for task in train.tasks.iter().chain(&held.tasks) {
        let folder = dir.path().join("practised").join(&task.id);
        for file in [
            "task.toml",
            "instruction.md",
            "tests/test.sh",
            "environment/Dockerfile",
        ] {
            assert!(folder.join(file).is_file(), "{} has {file}", task.id);
        }
    }
    let family = |id: &str| {
        id.split('/')
            .nth(1)
            .unwrap()
            .rsplit_once('-')
            .unwrap()
            .0
            .to_owned()
    };
    let trained: std::collections::BTreeSet<_> =
        train.tasks.iter().map(|t| family(&t.id)).collect();
    assert!(
        held.tasks.iter().all(|t| !trained.contains(&family(&t.id))),
        "no file is both trained on and examined: {trained:?} {:?}",
        held.tasks.iter().map(|t| &t.id).collect::<Vec<_>>()
    );
}

#[test]
fn a_revision_is_named_by_its_commit() {
    let (repo, _) = repository();
    let recipe = "schema: splinter.recipe/v2\nname: practised\nstages:\n  - id: practised\n    uses: task.mutate\n    with: { repository: project, revision: main, seed: 3 }\n";
    let refused =
        splinter_factory::resolve::resolve(&repo.root, recipe, crate::repo::connections())
            .err()
            .map(|e| format!("{e:?}"))
            .unwrap_or_default();
    assert!(refused.contains("by its commit"), "{refused}");
}
