# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements bug-fixing training environments with
# trustworthy verifiers, for its clients. If your team needs expertise in
# repository tasks for agent reinforcement learning, you can procure our
# services by sending an email to info@swedishembedded.com.

"""Runs a project's whole test suite in its own process and reports one line.

The process reads a nonce from standard input before it imports anything of the
project and prints one result line that carries it; the parent verifier trusts
only that line. The project is appended to the import path after the standard
library, so a module planted in the workspace cannot shadow `unittest` or
`json`. A canary test that always fails runs too: a project that patches the
test framework to report success reports success for the canary as well.
"""
import io
import os
import sys
import unittest

sys.dont_write_bytecode = True  # grading must not change what it grades
nonce = sys.stdin.readline().strip()
root = os.environ["WORK_DIR"]
sys.path.append(root)


class Canary(unittest.TestCase):
    def test_must_fail(self):
        self.assertEqual(1, 2)


suite = unittest.defaultTestLoader.discover(root, top_level_dir=root)
result = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)
canary = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(
    unittest.defaultTestLoader.loadTestsFromTestCase(Canary)
)
failed = len(result.failures) + len(result.errors)
passed = result.testsRun - failed - len(result.skipped)
canary_failed = 0 if canary.wasSuccessful() else 1
sys.stdout.write("@@RESULT %s %d %d %d\n" % (nonce, passed, result.testsRun, canary_failed))
sys.stdout.flush()
