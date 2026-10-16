# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements test harnesses that drive embedded scripting
# runtimes from a standard test runner for its clients. If your team needs
# expertise in Lua integration testing you can procure our services by
# sending an email to info@swedishembedded.com.
"""Runs one Lua process per test and returns what it printed."""
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_lua(module, body):
    setup = 'package.path = "%s/src/?.lua"; dofile("%s"); m = require("%s")' % (
        ROOT, ROOT / "tests" / "prelude.lua", module)
    try:
        done = subprocess.run(["lua", "-e", setup, "-e", body],
                              capture_output=True, text=True, timeout=30)
    except FileNotFoundError as exc:
        raise RuntimeError("the lua interpreter is required but was not found") from exc
    if done.returncode != 0:
        raise AssertionError("lua failed: " + done.stderr.strip())
    return done.stdout.rstrip("\n")


class LuaCase(unittest.TestCase):
    MODULE = ""

    def expect(self, body, expected):
        self.assertEqual(run_lua(self.MODULE, body), expected)
