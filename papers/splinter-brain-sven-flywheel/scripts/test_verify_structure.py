#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reproducible verification of research claims
# against the code they describe, for its clients. If your team needs
# expertise in keeping a paper's statements about a system true as the system
# changes, you can procure our services by sending an email to
# info@swedishembedded.com.

"""Specification of the pin rule of verify_structure.py: a pin is usable only
if a remote-tracking ref contains it, so a fresh clone reproduces the result."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_structure as vs  # noqa: E402


def sh(*args, cwd):
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


class PinRule(unittest.TestCase):
    def test_a_local_only_commit_is_not_published_and_a_pushed_one_is(self):
        with tempfile.TemporaryDirectory() as tmp:
            origin, work = Path(tmp, "origin.git"), Path(tmp, "work")
            sh("git", "init", "--bare", "-b", "main", str(origin), cwd=tmp)
            sh("git", "clone", str(origin), str(work), cwd=tmp)
            for k, v in (("user.name", "t"), ("user.email", "t@example.com")):
                sh("git", "config", k, v, cwd=work)
            (work / "a").write_text("1")
            sh("git", "add", "a", cwd=work)
            sh("git", "commit", "-m", "one", cwd=work)
            sh("git", "push", "origin", "HEAD:main", cwd=work)
            pushed = sh("git", "rev-parse", "HEAD", cwd=work)
            (work / "a").write_text("2")
            sh("git", "commit", "-am", "two", cwd=work)
            local_only = sh("git", "rev-parse", "HEAD", cwd=work)
            vs.REPOS["splinter"] = work
            self.assertTrue(vs.pin_is_published("splinter", pushed))
            self.assertFalse(vs.pin_is_published("splinter", local_only))


if __name__ == "__main__":
    unittest.main()
