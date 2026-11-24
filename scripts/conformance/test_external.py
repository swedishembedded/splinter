#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for proving a learning system's algorithms against pinned, independent
# reference implementations, for its clients. If your team needs expertise in conformance testing then you can
# procure our services by sending an email to info@swedishembedded.com.
"""The specification of scripts/conformance/external.py: every way a reference check could pass falsely is refused.

Each test builds a scratch repository (a lock, an oracle, a frozen fixture) and runs the real runner against it
with the oracle environment's interpreter (target/validation/venv, `external.py env`).

    python3 -I scripts/conformance/test_external.py
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "conformance" / "external.py"
PYTHON = os.environ.get("ORACLE_PYTHON") or str(ROOT / "target" / "validation" / "venv" / "bin" / "python")

ORACLE = '''import argparse, json
p = argparse.ArgumentParser(); p.add_argument("--out"); p.add_argument("--seed", type=int); p.add_argument("--cases")
a = p.parse_args()
data = open("validation/oracles/data.txt").read().strip()
fixture = {"fixture": "toy/double", "references": ["pyyaml"], "cases": [{"x": 2, "y": int(data) * 2}]}
if a.seed is not None:
    fixture |= {"differential": True, "seed": a.seed}
open(a.out, "w").write(json.dumps(fixture, sort_keys=True) + "\\n")
'''


def installed_pyyaml() -> str:
    return subprocess.run([PYTHON, "-I", "-c", "import yaml; print(yaml.__version__)"], capture_output=True,
                          text=True, check=True).stdout.strip()


class ExternalRunner(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "validation" / "oracles").mkdir(parents=True)
        self.lock(installed_pyyaml())
        (self.root / "validation" / "oracles" / "data.txt").write_text("21\n")
        (self.root / "validation" / "oracles" / "toy.py").write_text(ORACLE)
        self.ok("generate", "validation/oracles/toy.py", "--test", "splinter-model/it/toy")

    def tearDown(self):
        self.tmp.cleanup()

    def lock(self, version, reference="pyyaml"):
        (self.root / "validation" / "references.lock.yaml").write_text(
            "schema: splinter.validation-references/v1\npackages:\n"
            f"  - {{ name: pyyaml, version: \"{version}\", sha256: '00', license: MIT }}\n"
            f"references:\n  {reference}: {{ package: pyyaml, confidence: high }}\n")

    def run_runner(self, *args):
        return subprocess.run([sys.executable, "-I", str(RUNNER), "--root", str(self.root), "--python", PYTHON, *args],
                              capture_output=True, text=True, check=False)

    def ok(self, *args):
        ran = self.run_runner(*args)
        self.assertEqual(ran.returncode, 0, ran.stderr)

    def refused(self, why, *args):
        ran = self.run_runner(*args)
        self.assertEqual(ran.returncode, 1, f"accepted: {ran.stderr}")
        self.assertIn(why, ran.stderr)

    def fixture(self):
        return self.root / "validation" / "fixtures" / "toy" / "double.json"

    def test_a_fixture_its_pinned_oracle_still_writes_holds(self):
        self.ok("check")

    def test_a_changed_expected_value_is_refused(self):
        self.fixture().write_text(self.fixture().read_text().replace("42", "43"))
        self.refused("not the file recorded", "check")

    def test_a_reference_at_another_version_is_refused(self):
        self.lock("0.0.1")
        self.refused("the lock pins 0.0.1", "check")

    def test_a_fixture_naming_an_unpinned_reference_is_refused(self):
        self.lock(installed_pyyaml(), reference="other")
        self.refused("not pinned", "check")

    def test_an_oracle_changed_since_its_fixture_was_frozen_is_refused(self):
        oracle = self.root / "validation" / "oracles" / "toy.py"
        oracle.write_text(oracle.read_text() + "# edited\n")
        self.refused("changed since the fixture was frozen", "check")

    def test_a_reference_that_no_longer_writes_the_fixture_is_refused(self):
        (self.root / "validation" / "oracles" / "data.txt").write_text("22\n")
        self.refused("no longer write it byte for byte", "check")

    def test_an_oracle_that_crashes_is_refused(self):
        (self.root / "validation" / "oracles" / "data.txt").write_text("not a number\n")
        self.refused("failed (exit 1)", "check")

    def test_a_missing_fixture_is_refused(self):
        self.fixture().unlink()
        self.refused("is missing", "check")

    def test_a_manifest_of_nothing_certifies_nothing(self):
        (self.root / "validation" / "fixtures.yaml").write_text("schema: splinter.validation-fixtures/v1\nfixtures: {}\n")
        self.refused("lists no fixture", "check")

    def test_a_fixture_without_its_consuming_test_is_not_frozen(self):
        (self.root / "validation" / "fixtures.yaml").unlink()
        self.refused("name the test that reads it", "generate", "validation/oracles/toy.py")


if __name__ == "__main__":
    unittest.main(verbosity=1)
