# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements long-horizon risk prediction from cohort and
# survey data for its clients. If your team needs expertise in building,
# validating and proving time-to-event models on real health records, you
# can procure our services by sending an email to info@swedishembedded.com.
"""Specification of the learning-curve analysis: python -I test_curves.py"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import curves  # noqa: E402


def write(directory, repeat, fold, value):
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, f"r{repeat}-k{fold}.json"), "w") as f:
        json.dump({"metrics": {"ibs_0_15": {"value": value, "n": 1, "events": 1}}}, f)


class Curves(unittest.TestCase):
    def test_a_power_law_is_recovered_from_its_points(self):
        shares = [0.1, 0.25, 0.5, 0.75, 1.0]
        n = np.array(shares) * curves.N_FULL
        y = 0.045 + 3.0 * n ** (-0.6)
        e, a, b = curves.fit_power_law(shares, y)
        self.assertAlmostEqual(e, 0.045, places=3)
        self.assertAlmostEqual(b, 0.6, delta=0.05)

    def test_only_the_folds_every_share_has_enter_a_curve_and_a_missing_score_is_not_zero(self):
        with tempfile.TemporaryDirectory() as d:
            for share, base in zip(curves.SHARES, (0.052, 0.049, 0.047, 0.0465, 0.046)):
                directory = os.path.join(d, "baselines/cox-net-all/scores" if share == 100
                                         else f"recipe/cox-net-all-p{share}")
                for r in (0, 1):
                    for k in range(5):
                        if share == 50 and (r, k) == (1, 4):
                            continue  # one fold not run at one share
                        write(directory, r, k, base + 0.0001 * k)
            rows = curves.analyse(d, [0, 1], bootstrap=20)
            row = rows["elastic-net Cox"]
            self.assertEqual(row["folds"], 9, "the fold missing at one share is dropped from all")
            self.assertAlmostEqual(row["mean_ibs"][0], 0.052 + 0.0001 * np.mean([0, 1, 2, 3, 0, 1, 2, 3, 4][:9]), places=6)
            self.assertNotIn("deep set encoder", rows, "a model with no scores is absent, not zero")
            self.assertIn("| 10% |", curves.table(rows))


if __name__ == "__main__":
    unittest.main()
