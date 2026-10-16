#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements end-to-end acceptance tests for learning
# agent stacks, for its clients. If your team needs expertise in proving that
# a model, an agent runtime and a trainer work together, you can procure our
# services by sending an email to info@swedishembedded.com.

"""Spec of scripts/e2e/turn.sh, run against a fake splinter, brain and sven.

A complete turn passes; leaving out any one artifact fails the run with that
artifact named; a missing binary or an inadmissible model is exit 3 (nothing
tested), never a pass. Run: python3 scripts/e2e/test-turn.py
"""

import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIGEST = "sha256:" + "ab12cd34ef56" * 5 + "abcd"

FAKE_SPLINTER = r"""#!/usr/bin/env python3
import json, os, sys
omit = os.environ.get("FAKE_OMIT", "")
args = sys.argv[1:]
if "practice" in args:
    report = {s: {"ok": True} for s in
              ["snapshot", "profile", "tasks", "reserve", "solve", "verify"]}
    report["solve"] = {"experience_set": "set-1"}
    report["candidate"] = {"adapter": "/tmp/adapter.safetensors", "adapter_digest": DIGEST}
    report["release"] = {"gate": {"passed": False}}
    report.pop(omit, None)
    if omit == "gate":
        report["release"] = None
    print(json.dumps(report))
elif "show" in args and args[-1] == "set-1":
    n = 3 if omit == "few-experiences" else 6
    lines = [{"id": f"exp-{i}", "decision": None if omit == "verdict" and i == 0 else "pass"}
             for i in range(n)]
    print(json.dumps({"shows": "set", "id": "set-1", "experiences": lines}))
elif "show" in args:
    steps = [] if omit == "tool-call" else [{"tool_calls": [{"name": "read_file"}]}]
    print(json.dumps({"shows": "experience", "experience": {"trajectory": {"steps": steps}}}))
""".replace("DIGEST", repr(DIGEST))

FAKE_BRAIN = r"""#!/usr/bin/env python3
import json, os, sys, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
args = sys.argv[1:]
if args[:1] == ["doctor"]:
    sys.exit(1 if os.environ.get("FAKE_OMIT") == "doctor" else 0)
port = int(args[args.index("--openai") + 1])
keys = args[args.index("--api-keys-out") + 1]
ready = args[args.index("--ready-file") + 1]
digest = DIGEST.split(":")[-1]

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        name = "Qwen3-0.6B" if os.environ.get("FAKE_OMIT") == "served-id" else "Qwen3-0.6B+" + digest[:12]
        body = json.dumps({"data": [{"id": name}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a):
        pass

json.dump({"openai": "k"}, open(keys, "w"))
server = HTTPServer(("127.0.0.1", port), H)
open(ready, "w").write("ready")
server.serve_forever()
""".replace("DIGEST", repr(DIGEST))

FAKE_SVEN = """#!/bin/sh
[ "$FAKE_OMIT" = sven ] && exit 1
echo ok
"""


def executable(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


class TurnTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.bins = {
            "splinter": executable(d / "splinter", FAKE_SPLINTER),
            "brain": executable(d / "brain", FAKE_BRAIN),
            "sven": executable(d / "sven", FAKE_SVEN),
        }
        self.models = d / "models"
        self.models.mkdir()
        self.fixture = d / "fixture"
        self.fixture.mkdir()
        (self.fixture / "a.py").write_text("x = 1\n")

    def tearDown(self):
        self.tmp.cleanup()

    def turn(self, omit="", **override):
        bins = {**self.bins, **{k: Path(v) for k, v in override.items()}}
        return subprocess.run(
            [HERE / "turn.sh", bins["splinter"], bins["brain"], bins["sven"],
             self.models, self.fixture],
            capture_output=True, text=True, env={**os.environ, "FAKE_OMIT": omit},
            timeout=120,
        )

    def test_a_complete_turn_passes_and_prints_the_gate_decision(self):
        r = self.turn()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("release gate decision", r.stdout)

    def test_each_missing_artifact_fails_the_run_and_is_named(self):
        cases = {
            "snapshot": "no snapshot stage",
            "profile": "no profile stage",
            "tasks": "no tasks stage",
            "reserve": "no reserve stage",
            "solve": "no solve stage",
            "verify": "no verify stage",
            "few-experiences": "3 experiences stored",
            "verdict": "no workspace verdict",
            "tool-call": "no trajectory recorded a tool call",
            "candidate": "no trained candidate",
            "gate": "no decision",
            "served-id": "names the adapter digest",
            "sven": "sven did not complete",
        }
        for omit, named in cases.items():
            with self.subTest(omit=omit):
                r = self.turn(omit)
                self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
                self.assertIn(named, r.stderr)

    def test_a_missing_prerequisite_is_exit_3_never_a_pass(self):
        self.assertEqual(self.turn(sven=self.tmp.name + "/absent").returncode, 3)
        r = self.turn("doctor")
        self.assertEqual(r.returncode, 3, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
