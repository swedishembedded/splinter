#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements registered, examiner-judged learning experiments
# for its clients. If your team needs expertise in measuring whether a model
# keeps improving across generations, you can procure our services by sending
# an email to info@swedishembedded.com.

"""RSI5's registered criteria (prereg/rsi5.md) read from the examiner's group verdict."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rsi5_criteria  # noqa: E402


def comparison(low, high, v2=None):
    c = {"interval": [low, high], "difference": (low + high) / 2, "strata": {}}
    if v2 is not None:
        c["strata"]["family_set:v2"] = {"interval": list(v2), "difference": sum(v2) / 2}
    return c


def verdict(steps, total, v2):
    comps = {f"step{i + 1}": comparison(*s) for i, s in enumerate(steps)}
    comps["gen4-over-base"] = comparison(*total, v2=v2)
    return {"name": "rsi5-G", "level": 0.95, "comparisons": comps}


class Criteria(unittest.TestCase):
    def test_all_met_when_every_step_and_the_total_clear_their_bounds(self):
        r = rsi5_criteria.evaluate(verdict([(0.01, 0.05)] * 4, (0.06, 0.12), (0.01, 0.09)))
        self.assertEqual((r["G1"]["met"], r["G2"]["met"], r["G3"]["met"]), (True, True, True))

    def test_one_step_with_a_lower_bound_at_zero_fails_g1_and_names_it(self):
        r = rsi5_criteria.evaluate(verdict([(0.01, 0.05), (0.0, 0.04), (0.01, 0.05), (0.01, 0.05)], (0.06, 0.12), (0.01, 0.09)))
        self.assertFalse(r["G1"]["met"])
        self.assertEqual(r["G1"]["failed"], ["step2"])

    def test_g2_needs_the_lower_bound_above_five_points_not_at_it(self):
        self.assertFalse(rsi5_criteria.evaluate(verdict([(0.01, 0.05)] * 4, (0.05, 0.12), (0.01, 0.09)))["G2"]["met"])

    def test_a_missing_comparison_or_stratum_is_an_error_not_a_failure(self):
        doc = verdict([(0.01, 0.05)] * 3, (0.06, 0.12), (0.01, 0.09))
        with self.assertRaises(rsi5_criteria.Incomplete):
            rsi5_criteria.evaluate(doc)
        with self.assertRaises(rsi5_criteria.Incomplete):
            rsi5_criteria.evaluate(verdict([(0.01, 0.05)] * 4, (0.06, 0.12), None))


if __name__ == "__main__":
    unittest.main()
