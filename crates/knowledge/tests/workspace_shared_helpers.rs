// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements executable evaluation of code changes
// against a project's own test suite, for its clients. If your team needs
// expertise in test-driven benchmarks for code-editing agents, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: a test file that other tests import (a shared base class of the
//! suite) is never withheld whole, and a fault it catches is still a task.
//! The solver's tree keeps the file without the failing cases, the hidden
//! overlay restores them at verification, and the tests that stay run in the
//! tree the solver is given.

#![allow(clippy::unwrap_used)]

mod common;

use std::fs;
use std::process::Command;

use common::Repo;
use splinter_eval::verifiers::workspace::WorkspaceCheck;
use splinter_knowledge::tasks::workspace::WorkspaceTask;

/// The module every test of the suite imports its base class from, with the
/// pricing tests in it.
const BASKET_TESTS: &str = "\
import unittest

from shop.pricing import discount, total


class Validator(unittest.TestCase):
    def check(self, got, want):
        self.assertEqual(got, want)


class BasketTest(Validator):
    def test_empty_basket_costs_nothing(self):
        self.check(total([]), 0)

    def test_prices_are_added_up(self):
        self.check(total([2, 3]), 5)

    def test_ten_percent_off(self):
        self.check(discount(200, 10), 180)
";

const LABELS: &str = "\
def shout(text):
    return text.upper()
";

const LABEL_TESTS: &str = "\
from shop.labels import shout
from tests.test_basket import Validator


class LabelTest(Validator):
    def test_a_label_is_upper_cased(self):
        self.check(shout(\"a\"), \"A\")
";

fn check_of(task: &WorkspaceTask) -> WorkspaceCheck {
    let item = task
        .task
        .privileged
        .iter()
        .find(|p| format!("{:?}", p.kind).contains("workspace-check"))
        .unwrap();
    serde_json::from_str(&item.content).unwrap()
}

#[test]
fn a_fault_caught_in_a_module_the_tests_import_leaves_the_module_with_the_failing_cases_removed() {
    let repo = Repo::of_files(
        "shared-helper",
        &[
            (
                "pyproject.toml",
                "[project]\nname = \"shop\"\nversion = \"1\"\n",
            ),
            ("shop/__init__.py", ""),
            ("shop/pricing.py", common::PRICING),
            ("shop/labels.py", LABELS),
            ("tests/__init__.py", ""),
            ("tests/test_basket.py", BASKET_TESTS),
            ("tests/test_labels.py", LABEL_TESTS),
        ],
    );
    let generation = repo.generate(6);
    let pricing: Vec<&WorkspaceTask> = generation
        .admitted
        .iter()
        .filter(|t| t.family == "shop/pricing.py")
        .collect();
    assert!(
        !pricing.is_empty(),
        "refused {:?}, admitted {}",
        generation.refused,
        generation.admitted.len()
    );
    for (n, task) in pricing.into_iter().enumerate() {
        let check = check_of(task);
        // The module stays in the tree and is protected, as the tests that
        // stay are; it is not a hidden path, so reading it is no exposure.
        assert!(!check.hidden_paths.contains(&"tests/test_basket.py".into()));
        assert!(check.protected.contains(&"tests/test_basket.py".into()));
        assert!(check.hidden.is_some());

        let tree = repo.scratch.0.join(format!("solver-{n}"));
        repo.solver_tree(task, &tree);
        let kept = fs::read_to_string(tree.join("tests/test_basket.py")).unwrap();
        assert!(kept.contains("class Validator"), "{kept}");
        // The failing cases are not in the solver's tree, by name or body.
        let failing: Vec<&str> = check.expectations[0]
            .tests
            .iter()
            .map(|id| id.rsplit('.').next().unwrap())
            .collect();
        assert!(!failing.is_empty());
        for name in &failing {
            assert!(!kept.contains(name), "{name} in\n{kept}");
        }
        assert!(kept.lines().count() < BASKET_TESTS.lines().count());

        // What stays runs and passes there.
        let run = Command::new("python3")
            .args(["-m", "unittest", "tests.test_basket", "tests.test_labels"])
            .current_dir(&tree)
            .output()
            .unwrap();
        assert!(
            run.status.success(),
            "{}",
            String::from_utf8_lossy(&run.stderr)
        );

        // The overlay restores the whole module at verification.
        let overlay = repo.store.path(check.hidden.as_ref().unwrap()).unwrap();
        let listing = Command::new("tar")
            .arg("-tf")
            .arg(&overlay)
            .output()
            .unwrap();
        assert!(String::from_utf8_lossy(&listing.stdout).contains("tests/test_basket.py"));
    }
}
