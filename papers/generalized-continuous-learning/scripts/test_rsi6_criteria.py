#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements registered, examiner-judged learning experiments
# for its clients. If your team needs expertise in measuring whether a model
# keeps acquiring new skills without losing old ones, you can procure our
# services by sending an email to info@swedishembedded.com.

"""RSI6's registered criteria (prereg/rsi6.md) read from the examiner's group verdict (plan rsi6-H)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rsi6_criteria  # noqa: E402

LANGS = ("lua51", "python3", "lua51-dialect-a", "lua51-dialect-b")


def comp(low, high, strata=None):
    return {"interval": [low, high], "difference": (low + high) / 2,
            "strata": {f"family_set:{k}": {"interval": list(v), "difference": sum(v) / 2} for k, v in (strata or {}).items()}}


def verdict(steps=((0.05, 0.1),) * 4, vol=(0.02, 0.06), retention=((-0.01, 0.02),) * 3):
    c = {f"step{i + 1}": comp(*s) for i, s in enumerate(steps)}
    c["acquisition-over-volume"] = comp(*vol)
    for j, r in enumerate(retention, start=1):
        c[f"retention{j}"] = comp(-0.1, 0.1, {LANGS[j - 1]: r})
    return {"name": "rsi6-H", "comparisons": c}


class Criteria(unittest.TestCase):
    def test_all_met(self):
        r = rsi6_criteria.evaluate(verdict())
        self.assertEqual((r["H1"]["met"], r["H2"]["met"], r["H3"]["met"]), (True, True, True))

    def test_a_flat_step_fails_h1_and_is_named(self):
        r = rsi6_criteria.evaluate(verdict(steps=((0.05, 0.1), (0.05, 0.1), (-0.01, 0.03), (0.05, 0.1))))
        self.assertEqual((r["H1"]["met"], r["H1"]["failed"]), (False, ["step3"]))

    def test_volume_control_must_be_beaten_with_the_lower_bound_above_zero(self):
        self.assertFalse(rsi6_criteria.evaluate(verdict(vol=(0.0, 0.05)))["H2"]["met"])

    def test_retention_reads_the_earlier_familys_own_stratum_and_allows_two_points(self):
        ok = rsi6_criteria.evaluate(verdict(retention=((-0.019, 0.01), (-0.01, 0.0), (0.0, 0.02))))
        self.assertTrue(ok["H3"]["met"])
        bad = rsi6_criteria.evaluate(verdict(retention=((-0.01, 0.01), (-0.03, 0.0), (0.0, 0.02))))
        self.assertEqual((bad["H3"]["met"], bad["H3"]["failed"]), (False, ["python3"]))

    def test_missing_pieces_are_incomplete_not_failed(self):
        doc = verdict()
        del doc["comparisons"]["retention2"]["strata"]["family_set:python3"]
        with self.assertRaises(rsi6_criteria.Incomplete):
            rsi6_criteria.evaluate(doc)


if __name__ == "__main__":
    unittest.main()
