# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements emulator-verified firmware training
# environments, for its clients. If your team needs expertise in testing
# firmware against a reference model in simulation, you can procure our
# services by sending an email to info@swedishembedded.com.

"""Grades the workspace: for each hidden seed, builds a private copy of the
workspace with that seed's inputs, runs it in the emulator and compares the
console output with the reference model. Writes the reward itself; imports
nothing of the workspace."""
import json
import os
import shutil
import sys
import tempfile

tests = os.environ.get("TESTS_DIR", "/tests")
app = os.environ.get("APP_DIR", "/app")
logs = os.environ.get("LOGS_DIR", "/logs/verifier")
sys.path.insert(0, tests)
sys.dont_write_bytecode = True
import model  # noqa: E402  the grader's own reference model

hidden = json.load(open(os.path.join(tests, "hidden_inputs.json")))
passed = 0
for number in hidden:
    work = tempfile.mkdtemp(prefix="grade-")
    try:
        shutil.copytree(app, os.path.join(work, "w"), ignore=shutil.ignore_patterns("build", "__pycache__"))
        status, detail = model.check(os.path.join(work, "w"), int(number), os.environ.get("VEMU", "vemu"), os.environ.get("CLANG", "clang"))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if status == "ok":
        passed += 1
    else:
        print("hidden run %s: %s %s" % (number, status, detail if status != "differs" else detail[:2]))
exact = 1.0 if passed == len(hidden) else 0.0
os.makedirs(logs, exist_ok=True)
with open(os.path.join(logs, "reward.json"), "w") as out:
    json.dump({"reward": exact, "exact": exact, "runs": round(passed / len(hidden), 4)}, out)
