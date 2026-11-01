# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
"""loop.py: a candidate replaces the champion only on a paired exam gain, and
the loop resumes from what its generations recorded."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import loop  # noqa: E402


def a_candidate_is_promoted_only_on_a_positive_paired_gain():
    rule = {"min_mean_delta": 0.0}
    assert loop.decide({"mean_delta": 0.1, "interval95": [-0.05, 0.25]}, rule)["promoted"]
    assert not loop.decide({"mean_delta": 0.0, "interval95": [-0.1, 0.1]}, rule)["promoted"]
    assert not loop.decide({"mean_delta": None}, rule)["promoted"]
    strict = {"require_interval_above_zero": True}
    assert not loop.decide({"mean_delta": 0.1, "interval95": [-0.05, 0.25]}, strict)["promoted"]
    assert loop.decide({"mean_delta": 0.2, "interval95": [0.05, 0.35]}, strict)["promoted"]


def _gen(work: Path, n: int, state: dict) -> None:
    d = work / "generations" / f"{n:03d}"
    d.mkdir(parents=True)
    (d / "state.json").write_text(json.dumps({"generation": n, **state}))


def the_champion_is_the_last_promoted_adapter_with_its_exam(tmp_path):
    campaign = loop.Campaign({"paths": {"work": str(tmp_path)}})
    assert loop.champion(campaign)["adapter"] is None
    base = {"name": "base", "adapter": None, "exam_job": None}
    _gen(tmp_path, 0, {"champion_in": base, "exam_job": "j0", "candidate": {"name": "gen001", "adapter": "a1",
         "exam_job": "j1"}, "decision": {"promoted": False}})
    kept = loop.champion(campaign)
    assert kept["adapter"] is None and kept["exam_job"] == "j0", "a kept champion is not measured again"
    _gen(tmp_path, 1, {"champion_in": kept, "exam_job": "j0", "candidate": {"name": "gen002", "adapter": "a2",
         "exam_job": "j2"}, "decision": {"promoted": True}})
    assert loop.champion(campaign) == {"name": "gen002", "adapter": "a2", "exam_job": "j2"}


def a_generation_resumes_after_its_last_finished_step(tmp_path):
    campaign = loop.Campaign({"paths": {"work": str(tmp_path)}})
    gen = loop.Generation(campaign, 0)
    gen.record("explore", explore_job="j")
    again = loop.Generation(campaign, 0)
    assert again.done("explore") and not again.done("exam") and again.state["explore_job"] == "j"


class Loop(unittest.TestCase):
    def test_a_candidate_is_promoted_only_on_a_positive_paired_gain(self):
        a_candidate_is_promoted_only_on_a_positive_paired_gain()

    def test_the_champion_is_the_last_promoted_adapter_with_its_exam(self):
        with tempfile.TemporaryDirectory() as tmp:
            the_champion_is_the_last_promoted_adapter_with_its_exam(Path(tmp))

    def test_a_generation_resumes_after_its_last_finished_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            a_generation_resumes_after_its_last_finished_step(Path(tmp))


if __name__ == "__main__":
    unittest.main()
