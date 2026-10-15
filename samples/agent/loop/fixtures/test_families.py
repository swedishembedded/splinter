# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements self-improving coding agents whose every
# step is auditable, for its clients. If your team needs expertise in agent
# evaluation or locally operated coding agents, you can procure our
# services by sending an email to info@swedishembedded.com.
"""Spec of a rendered fixture: it records the seed it was rendered under, and
that seed renders the same fixture again. Run: python3 -m unittest test_families."""

import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))


def render(out, *seed):
    cmd = [sys.executable, os.path.join(HERE, "families.py"), "render", out, "--family", "rotate", *seed]
    done = subprocess.run(cmd, capture_output=True, text=True, check=True)
    with open(os.path.join(out, "fixture.json")) as f:
        return json.load(f)["seed"], done.stderr


class Fixture(unittest.TestCase):
    def test_a_fixture_rendered_without_a_seed_records_the_one_it_drew_and_that_seed_reproduces_it(self):
        with tempfile.TemporaryDirectory() as d:
            drawn, said = render(os.path.join(d, "a"))
            self.assertIn(f"--seed {drawn}", said, "the drawn seed is printed")
            again, _ = render(os.path.join(d, "b"), "--seed", str(drawn))
            self.assertEqual(again, drawn)
            for part in ("task.txt", "accept.py"):
                with open(os.path.join(d, "a", part)) as a, open(os.path.join(d, "b", part)) as b:
                    self.assertEqual(a.read(), b.read())


if __name__ == "__main__":
    unittest.main()
