# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements test harnesses that drive embedded Lua
# libraries from Python for its clients. If your team needs expertise in
# testing Lua code from a standard-library-only toolchain you can procure our
# services by sending an email to info@swedishembedded.com.
"""Runs one Lua process per check and returns what it printed."""
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
PRELUDE = Path(__file__).resolve().parent / "prelude.lua"


def run_lua(body: str) -> str:
    """Run `body` after the prelude with src/ on package.path; fail loudly."""
    script = (
        f'package.path = "{SRC}/?.lua"\n'
        f'dofile("{PRELUDE}")\n'
        f"{body}\n"
    )
    try:
        done = subprocess.run(
            ["lua", "-"], input=script, capture_output=True, text=True, timeout=30
        )
    except FileNotFoundError as exc:
        raise RuntimeError("the lua interpreter is required but was not found") from exc
    if done.returncode != 0:
        raise AssertionError(f"lua failed ({done.returncode}): {done.stderr.strip()}")
    return done.stdout.rstrip("\n")


class LuaCase(unittest.TestCase):
    """Base class: MODULE names the Lua module bound to `M` in each body."""

    MODULE = ""

    def lua(self, body: str) -> str:
        return run_lua(f'local M = require("{self.MODULE}")\n{body}')

    def check(self, body: str, expected: str) -> None:
        self.assertEqual(self.lua(body), expected)
