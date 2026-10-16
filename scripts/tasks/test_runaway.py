#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements benchmark suites for code-editing agents that
# stay trustworthy as they grow, for its clients. If your team needs expertise
# in executable task catalogs or supervised test runs, you can procure our
# services by sending an email to info@swedishembedded.com.

"""Spec: a mutated family that loops forever leaves no process behind.

A mutated Lua program can loop forever. The Lua families are driven from
Python `unittest` files that start one `lua` process per test, so the looping
program is a grandchild of whatever runs the suite. Two things must hold:

  * the family pre-flight (`families.run_suite`) stops the whole process tree
    of a suite that outlives its timeout, not only the `unittest` process;
  * every Lua test host gives its `lua` process a kernel CPU limit, so it ends
    by itself even when the process that started it was killed first.
"""
import importlib.util
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASKS = HERE.parents[1] / "tasks"
SETTLE_SECONDS = 5.0

# How each Lua family's host is asked to run a program that never returns.
HOSTS = {
    "caldate": ("caldatehost", 'caldatehost.run_lua("while true do end")'),
    "graphkit": ("graphkithost", 'graphkithost.LuaCase("lua").lua("while true do end")'),
    "tokenbucket": ("luahost", 'luahost.run_lua("tokenbucket", "while true do end")'),
}

spec = importlib.util.spec_from_file_location("families", HERE / "families.py")
families = importlib.util.module_from_spec(spec)
sys.modules["families"] = families
spec.loader.exec_module(families)


def is_running(pid: int) -> bool:
    """Whether `pid` is alive and not a zombie."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return not stat.rsplit(") ", 1)[-1].startswith("Z")


def wait_gone(pid: int) -> bool:
    deadline = time.monotonic() + SETTLE_SECONDS
    while is_running(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    return not is_running(pid)


def wait_for_file(path: Path) -> str:
    deadline = time.monotonic() + SETTLE_SECONDS
    while time.monotonic() < deadline:
        if path.exists() and path.read_text().strip():
            return path.read_text().strip()
        time.sleep(0.05)
    raise AssertionError(f"{path} was not written")


class RunSuiteStopsTheProcessTree(unittest.TestCase):
    def test_a_looping_grandchild_is_gone_after_the_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tests").mkdir()
            (root / "tests" / "__init__.py").write_text("")
            (root / "tests" / "test_loop.py").write_text(textwrap.dedent("""
                import subprocess, unittest

                class Loop(unittest.TestCase):
                    def test_loop(self):
                        subprocess.run(
                            ["sh", "-c", "echo $$ > lua.pid; exec lua -e 'while true do end'"],
                            timeout=60,
                        )
            """))
            code, _, _ = families.run_suite(root, timeout=1.5)
            self.assertEqual(code, 124)
            pid = int(wait_for_file(root / "lua.pid"))
            self.assertTrue(wait_gone(pid), "the looping grandchild outlived the suite's timeout")


class LuaHostsBoundTheirInterpreter(unittest.TestCase):
    """The host's `lua` has a CPU limit that does not depend on its parent."""

    def cpu_limit_line(self, family: str) -> str:
        module, call = HOSTS[family]
        driver = f"import {module}\n{call}\n"
        proc = subprocess.Popen(
            [sys.executable, "-c", driver],
            cwd=TASKS / family / "tests",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + SETTLE_SECONDS
            while time.monotonic() < deadline:
                found = subprocess.run(
                    ["pgrep", "-P", str(proc.pid), "-x", "lua"], capture_output=True, text=True,
                ).stdout.split()
                if found:
                    limits = Path(f"/proc/{found[0]}/limits").read_text()
                    return next(
                        line for line in limits.splitlines() if line.startswith("Max cpu time")
                    )
                time.sleep(0.05)
            self.fail(f"{family}: the host started no lua process")
        finally:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()

    def test_every_lua_host_sets_a_cpu_limit(self):
        for family in HOSTS:
            with self.subTest(family=family):
                line = self.cpu_limit_line(family)
                self.assertNotIn("unlimited", line, f"{family}: {line}")


if __name__ == "__main__":
    unittest.main()
