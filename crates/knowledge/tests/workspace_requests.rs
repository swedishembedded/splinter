// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements executable evaluation of code changes
// against a project's own test suite, for its clients. If your team needs
// expertise in test-driven benchmarks for code-editing agents, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: what the request of a task made from a seeded fault may say and
//! name. A test module that tests in the tree import is never withheld
//! whole; a failing test whose name is too short to act on
//! is listed with what its assertion observed, never with the test's code,
//! path or call, and a candidate whose assertion cannot be rendered is
//! refused with a counted reason.

#![allow(clippy::unwrap_used)]

mod common;

use common::Repo;
use splinter_eval::verifiers::workspace::WorkspaceCheck;
use splinter_knowledge::tasks::workspace::{Reason, WorkspaceGeneration, WorkspaceTask};

/// Tasks generated from the project `extra` is added to.
fn generate(name: &str, extra: &[(&str, &str)]) -> WorkspaceGeneration {
    Repo::new(name, extra).generate(6)
}

fn check_of(task: &WorkspaceTask) -> WorkspaceCheck {
    let item = task
        .task
        .privileged
        .iter()
        .find(|p| format!("{:?}", p.kind).contains("workspace-check"))
        .unwrap();
    serde_json::from_str(&item.content).unwrap()
}

/// An in-tree test that imports the module the pricing faults are caught in.
const USES_PRICING_TESTS: &str = "\
import unittest

from tests import test_pricing


class UsesPricingTests(unittest.TestCase):
    def test_the_pricing_tests_can_be_imported(self):
        self.assertTrue(hasattr(test_pricing, \"TotalTest\"))
";

#[test]
fn a_module_that_tests_in_the_tree_import_is_never_withheld_whole() {
    let generation = generate(
        "imported",
        &[("tests/test_uses_pricing.py", USES_PRICING_TESTS)],
    );
    // Faults in pricing are caught by tests/test_pricing.py, which the kept
    // test imports: taking it out whole would break the tree, leaving it in
    // would show the tests. Its failing cases are cut instead (see
    // workspace_shared_helpers).
    assert!(
        generation
            .admitted
            .iter()
            .any(|t| t.family == "shop/pricing.py"),
        "refused {:?}",
        generation.refused
    );
    assert_eq!(generation.count(Reason::HiddenImportedByTree), 0);
    for admitted in &generation.admitted {
        let check = check_of(admitted);
        assert!(
            !check
                .hidden_paths
                .contains(&"tests/test_pricing.py".to_string()),
            "{:?}",
            check.hidden_paths
        );
    }
}

/// A function whose only test has a one-word name.
const ROUNDING: &str = "\
def round_half_down(value):
    if value - int(value) > 0.5:
        return int(value) + 1
    return int(value)
";

const TEST_ROUNDING: &str = "\
import unittest

from shop.rounding import round_half_down


class RoundingTest(unittest.TestCase):
    def test_rounding(self):
        self.assertEqual(round_half_down(2.5), 2)
";

#[test]
fn a_symptom_too_short_to_act_on_is_rendered_from_the_failed_assertion() {
    let generation = generate(
        "symptom",
        &[
            ("shop/rounding.py", ROUNDING),
            ("tests/test_rounding.py", TEST_ROUNDING),
        ],
    );
    let rendered: Vec<&WorkspaceTask> = generation
        .admitted
        .iter()
        .filter(|t| t.family == "shop/rounding.py")
        .collect();
    assert!(
        !rendered.is_empty(),
        "admitted {:?}, refused {:?}",
        generation
            .admitted
            .iter()
            .map(|t| &t.family)
            .collect::<Vec<_>>(),
        generation.refused
    );
    for task in rendered {
        let text = &task.task.instruction;
        assert!(
            text.contains("- rounding: two values that must be equal are `"),
            "{text}"
        );
        assert!(text.contains("`2`"), "the value the test requires: {text}");
        // Nothing of the test's code, its path or the exception's name.
        assert!(
            !text.contains("test_rounding") && !text.contains("tests/"),
            "{text}"
        );
        assert!(
            !text.contains("assertEqual") && !text.contains("AssertionError"),
            "{text}"
        );
        assert!(!text.contains("round_half_down(2.5)"), "{text}");
    }
}

const FLAG: &str = "\
def is_enabled(value):
    if value > 0:
        return True
    return False
";

const TEST_FLAG: &str = "\
import unittest

from shop.flag import is_enabled


class FlagTest(unittest.TestCase):
    def test_flag(self):
        self.assertTrue(is_enabled(1))
";

#[test]
fn a_short_symptom_whose_assertion_cannot_be_rendered_refuses_the_task_with_a_count() {
    let generation = generate(
        "unrenderable",
        &[("shop/flag.py", FLAG), ("tests/test_flag.py", TEST_FLAG)],
    );
    assert!(
        generation.count(Reason::SymptomNotActionable) >= 1,
        "refused {:?}",
        generation.refused
    );
    assert!(generation
        .admitted
        .iter()
        .all(|t| t.family != "shop/flag.py"));
}
