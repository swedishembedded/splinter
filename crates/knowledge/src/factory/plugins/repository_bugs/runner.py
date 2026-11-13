# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements bug-fixing training environments with
# trustworthy verifiers, for its clients. If your team needs expertise in
# repository tasks for agent reinforcement learning, you can procure our
# services by sending an email to info@swedishembedded.com.

"""Runs the hidden suite against the workspace's library, in its own process.

The process reads a nonce from standard input before it imports anything of
the workspace and prints one result line that carries it. The parent verifier
trusts only that line. The library is appended to the import path after the
standard library, so a module planted in the workspace cannot shadow
`unittest` or `json`. A canary test that always fails is run too: a library
that patches the test framework to report success also reports success for the
canary, which gives it away.
"""
import io
import os
import sys
import unittest

sys.dont_write_bytecode = True  # grading must not change the workspace it grades
nonce = sys.stdin.readline().strip()
app = os.environ["APP_DIR"]
tests = os.environ["TESTS_DIR"]
sys.path.append(app)


class Canary(unittest.TestCase):
    def test_must_fail(self):
        self.assertEqual(1, 2)


suite_dir = os.path.join(tests, "suite")
suite = unittest.defaultTestLoader.discover(suite_dir, top_level_dir=suite_dir)
result = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)
canary = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(
    unittest.defaultTestLoader.loadTestsFromTestCase(Canary)
)
failed = len(result.failures) + len(result.errors)
passed = result.testsRun - failed - len(result.skipped)
canary_failed = 0 if canary.wasSuccessful() else 1
sys.stdout.write("@@RESULT %s %d %d %d\n" % (nonce, passed, result.testsRun, canary_failed))
sys.stdout.flush()
