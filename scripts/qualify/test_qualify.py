#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements qualification of learning agent systems
# against a matrix fixed before the first run, for its clients. If your team
# needs expertise in designing a qualification matrix and ablations for an
# agent that improves itself, you can procure our services by sending an
# email to info@swedishembedded.com.

"""Spec of the qualification harness: what a pass, a failure and a block are,
what is registered before anything runs, and that the shipped plan holds every
check and every mandatory ablation of the roadmap."""

import json
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

HERE = Path(__file__).parent
QUALIFY = HERE / "qualify.py"
PLAN = HERE / "plan.toml"

# The matrix and the ablations the roadmap's final qualification names.
MATRIX = {
    "builds", "tests", "isolation", "failed-candidate-never-champion", "autonomous-curriculum",
    "intervention-choice", "three-generations", "sft-beats-frozen-control",
    "negative-control-shows-no-gain", "harness-optimisation-helps", "reproducible-conclusions",
    "final-exam-independent-of-selection",
}
ABLATIONS = {
    "ablation-no-replay", "ablation-no-harness-optimisation",
    "ablation-harness-optimisation-without-training", "ablation-decision-assistance-off",
    "ablation-search-off", "ablation-retrieval-off",
}


def printing(document):
    return ["python3", "-c", f"import json; print(json.dumps({document!r}))"]


_DIRECTORIES = []


def tearDownModule():
    for directory in _DIRECTORIES:
        shutil.rmtree(directory, ignore_errors=True)


def run(plan_text, *extra):
    """Runs the harness over `plan_text`; returns (exit code, report or None, directory)."""
    directory = Path(tempfile.mkdtemp(prefix="qualify-test-"))
    _DIRECTORIES.append(directory)
    plan = directory / "plan.toml"
    plan.write_text(plan_text)
    out = directory / "out"
    done = subprocess.run(
        [sys.executable, str(QUALIFY), str(plan), "--out", str(out), *extra],
        capture_output=True, text=True,
    )
    report = out / "report.json"
    return done, json.loads(report.read_text()) if report.exists() else None, out


def toml_command(argv):
    return json.dumps(argv)


class Semantics(unittest.TestCase):
    def test_a_check_passes_only_when_its_evidence_holds_and_fails_when_it_does_not(self):
        plan = f"""
[[check]]
id = "holds"
title = "holds"
command = {toml_command(printing({"gain": 0.4}))}
[[check.assert]]
path = "gain"
op = ">="
value = 0.3
[[check]]
id = "does-not"
title = "does not"
command = {toml_command(printing({"gain": 0.1}))}
[[check.assert]]
path = "gain"
op = ">="
value = 0.3
"""
        done, report, _ = run(plan)
        status = {r["id"]: r["status"] for r in report["results"]}
        self.assertEqual(status, {"holds": "PASS", "does-not": "FAIL"})
        self.assertEqual(done.returncode, 1)

    def test_what_is_not_measured_is_not_a_pass(self):
        plan = f"""
[[check]]
id = "missing"
title = "the path is not in the evidence"
command = {toml_command(printing({"other": 1}))}
[[check.assert]]
path = "gain"
op = ">="
value = 0
"""
        _, report, _ = run(plan)
        result = report["results"][0]
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("not measured", result["assertions"][0]["reason"])

    def test_a_check_missing_something_it_needs_is_blocked_with_the_reason_and_is_not_a_pass(self):
        plan = f"""
[vars]
card = ""
needs = ""
[[check]]
id = "needs-a-card"
title = "a GPU check"
command = {toml_command(printing({"x": 1}))}
card_gb = 40
exit_is_evidence = true
[[check]]
id = "needs-a-variable"
title = "a variable"
command = ["python3", "-c", "print(1)", "{{needs}}"]
exit_is_evidence = true
[[check]]
id = "awaits-a-result"
title = "another lane's run"
result = "{{needs}}/nothing.json"
[[check.assert]]
path = "x"
op = "exists"
"""
        done, report, _ = run(plan)
        by = {r["id"]: r for r in report["results"]}
        self.assertEqual({k: v["status"] for k, v in by.items()}, {k: "BLOCKED" for k in by})
        self.assertIn("card", by["needs-a-card"]["reason"])
        self.assertIn("needs", by["needs-a-variable"]["reason"])
        self.assertEqual(done.returncode, 2, "blocked is neither a pass nor a failure")

    def test_an_ablation_states_its_effect_and_fails_when_a_side_is_unmeasured(self):
        plan = f"""
[[check]]
id = "full"
title = "the full system"
command = {toml_command(printing({"quality": 0.75}))}
metric = "quality"
[[check.assert]]
path = "quality"
op = ">"
value = 0
[[ablation]]
id = "without"
title = "without one thing"
of = "full"
command = {toml_command(printing({"quality": 0.5}))}
metric = "quality"
[[ablation]]
id = "unmeasured"
title = "reports no metric"
of = "full"
command = {toml_command(printing({"other": 1}))}
metric = "quality"
"""
        _, report, _ = run(plan)
        by = {r["id"]: r for r in report["results"]}
        self.assertEqual(by["without"]["status"], "PASS")
        self.assertAlmostEqual(by["without"]["effect"], 0.25)
        self.assertEqual(by["unmeasured"]["status"], "FAIL")
        self.assertIsNone(by["unmeasured"]["effect"])

    def test_the_plan_is_registered_before_anything_runs_and_another_plan_is_refused(self):
        plan = f"""
[[check]]
id = "one"
title = "one"
command = {toml_command(printing({"x": 1}))}
[[check.assert]]
path = "x"
op = "=="
value = 1
"""
        done, report, out = run(plan)
        lock = json.loads((out / "plan.lock.json").read_text())
        self.assertEqual(lock["plan_sha256"], report["plan_sha256"])
        refused, _, _ = run(plan, "--expect-plan", "0" * 64)
        self.assertEqual(refused.returncode, 3)
        self.assertIn("not the registered", refused.stderr)

    def test_a_plan_that_proves_nothing_is_refused(self):
        refused, _, _ = run(
            f'[[check]]\nid = "x"\ntitle = "x"\ncommand = {toml_command(printing({}))}\n'
        )
        self.assertEqual(refused.returncode, 3)
        self.assertIn("asserts something", refused.stderr)


class ShippedPlan(unittest.TestCase):
    def test_it_holds_every_check_of_the_matrix_and_every_mandatory_ablation(self):
        plan = tomllib.loads(PLAN.read_text())
        self.assertEqual({c["id"] for c in plan["check"]}, MATRIX)
        self.assertEqual({a["id"] for a in plan["ablation"]}, ABLATIONS)

    def test_it_validates_and_lists_its_entries_without_running_anything(self):
        done = subprocess.run(
            [sys.executable, str(QUALIFY), str(PLAN), "--out", "unused", "--list"],
            capture_output=True, text=True,
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        listed = {line.split("\t")[0] for line in done.stdout.splitlines()}
        self.assertEqual(listed, MATRIX | ABLATIONS)

    def test_without_its_variables_every_entry_is_blocked_and_nothing_passes(self):
        done, report, _ = run(PLAN.read_text())
        statuses = {r["status"] for r in report["results"]}
        self.assertNotIn("PASS", statuses)
        self.assertEqual(done.returncode, 2, done.stderr)


if __name__ == "__main__":
    unittest.main()
