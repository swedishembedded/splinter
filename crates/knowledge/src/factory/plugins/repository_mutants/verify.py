# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements bug-fixing training environments with
# trustworthy verifiers, for its clients. If your team needs expertise in
# repository tasks for agent reinforcement learning, you can procure our
# services by sending an email to info@swedishembedded.com.

"""Grades the workspace: copies it, lays the hidden tests over the copy, runs the
whole suite in a child process, trusts only its nonce-bearing result line and
writes the reward itself. Imports nothing of the workspace."""
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile

tests = os.environ.get("TESTS_DIR", "/tests")
app = os.environ.get("APP_DIR", "/app")
logs = os.environ.get("LOGS_DIR", "/logs/verifier")
expected_total = json.load(open(os.path.join(tests, "expected.json")))["total"]


def write(passed, total, exact):
    os.makedirs(logs, exist_ok=True)
    share = round(passed / total, 4) if total else 0.0
    with open(os.path.join(logs, "reward.json"), "w") as out:
        json.dump({"reward": exact, "exact": exact, "tests": share}, out)


scratch = tempfile.mkdtemp(prefix="grade-")
work = os.path.join(scratch, "work")
try:
    shutil.copytree(app, work, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    overlay = os.path.join(tests, "overlay")
    if os.path.isdir(overlay):
        shutil.copytree(overlay, work, dirs_exist_ok=True)
    nonce = secrets.token_hex(8)
    try:
        done = subprocess.run(
            [sys.executable, "-I", os.path.join(tests, "runner.py")],
            input=nonce + "\n",
            capture_output=True,
            text=True,
            cwd=work,
            timeout=120,
            env={"WORK_DIR": work, "PATH": os.environ.get("PATH", "")},
        )
    except subprocess.TimeoutExpired:
        print("the suite timed out")
        write(0, 0, 0.0)
        sys.exit(0)
finally:
    shutil.rmtree(scratch, ignore_errors=True)

marker = "@@RESULT %s " % nonce
lines = [line[len(marker):].split() for line in done.stdout.splitlines() if line.startswith(marker)]
if len(lines) != 1 or len(lines[0]) != 3:
    print("no result from the suite (exit %s)" % done.returncode)
    write(0, 0, 0.0)
    sys.exit(0)
passed, total, canary_failed = (int(v) for v in lines[0])
solved = total == expected_total and passed == total and canary_failed == 1
if not solved:
    print("%d of %d tests pass (canary failed: %d, expected %d tests)" % (passed, total, canary_failed, expected_total))
write(passed, total, 1.0 if solved else 0.0)
