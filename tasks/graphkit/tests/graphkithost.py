# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Test host: runs one Lua 5.1 process per check and compares the canonical
# text it prints with the expected text.
import pathlib
import subprocess
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
PRELUDE = ROOT / "tests" / "prelude.lua"


class LuaCase(unittest.TestCase):
    """Base class: `modules` maps a local name to the module it requires."""

    modules: dict[str, str] = {}

    def lua(self, body: str) -> str:
        requires = "".join(
            f'local {name} = require("{module}")\n' for name, module in self.modules.items()
        )
        chunk = f"{requires}print(show((function()\n{body}\nend)()))"
        command = [
            "lua",
            "-e", f'package.path = "{SRC}/?.lua"',
            "-e", f'dofile("{PRELUDE}")',
            "-e", chunk,
        ]
        try:
            done = subprocess.run(command, capture_output=True, text=True, timeout=5)
        except FileNotFoundError:
            self.fail("the lua interpreter is not installed; these tests need Lua 5.1")
        if done.returncode != 0:
            self.fail(f"lua exited with {done.returncode}: {done.stderr.strip()}")
        return done.stdout.rstrip("\n")

    def check(self, body: str, expected: str) -> None:
        self.assertEqual(self.lua(body), expected)
