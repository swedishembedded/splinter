#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements evaluation of synthetic speech that rests on
# blind human judgement, for its clients. If your team needs expertise in
# designing listening tests whose results can be trusted, you can procure
# our services by sending an email to info@swedishembedded.com.

"""Specification of make_rating_set.py: a session built without --seed draws a
seed, prints it and records it, and passing it back rebuilds the same session.

Run: python3 -m unittest test_make_rating_set (from this directory).
"""
import array
import json
import re
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "make_rating_set.py"
SENTENCES = 12


def takes(directory):
    """Two agreed takes of each of a dozen sentences, as `voice-loop takes` writes them."""
    directory.mkdir()
    rows = []
    for s in range(SENTENCES):
        for t in range(2):
            take = f"s{s:02d}-t{t}"
            with wave.open(str(directory / f"{take}.wav"), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(8000)
                w.writeframes(array.array("h", [100 * (t + 1) * (i % 7) for i in range(400)]).tobytes())
            rows.append({"id": take, "sentence": f"s{s:02d}", "agreed": True, "text": f"Sentence {s}."})
    (directory / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def build(takes_dir, out, *extra):
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(takes_dir), str(out), "--pairs", "4", "--repeats", "1", "--controls", "1", *extra],
        capture_output=True,
        text=True,
    )


def session(out):
    """Everything a session consists of, by relative path."""
    return {str(p.relative_to(out)): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()}


class RatingSet(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        takes(self.root / "takes")

    def tearDown(self):
        self.tmp.cleanup()

    def drawn(self, name):
        done = build(self.root / "takes", self.root / name)
        self.assertEqual(done.returncode, 0, done.stderr)
        printed = re.search(r"no --seed given, using seed (\d+)", done.stderr)
        self.assertIsNotNone(printed, done.stderr)
        recorded = json.loads((self.root / name / "seed.json").read_text())["seed"]
        self.assertEqual(int(printed.group(1)), recorded, "the printed seed is the recorded one")
        return recorded

    def test_a_session_without_a_seed_draws_prints_and_records_one(self):
        self.assertNotEqual(self.drawn("a"), self.drawn("b"), "an unseeded session is not a constant")

    def test_the_printed_seed_rebuilds_the_same_session(self):
        seed = self.drawn("first")
        again = build(self.root / "takes", self.root / "again", "--seed", str(seed))
        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertNotIn("no --seed given", again.stderr)
        self.assertEqual(session(self.root / "first"), session(self.root / "again"))


if __name__ == "__main__":
    unittest.main()
