# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements long-horizon risk prediction from cohort and
# survey data for its clients. If your team needs expertise in building,
# validating and proving time-to-event models on real health records, you
# can procure our services by sending an email to info@swedishembedded.com.
"""Specification of the variable-importance analysis: python -I test_drivers.py"""
import json
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drivers  # noqa: E402
import models  # noqa: E402


def cohort(n=3000, seed=0):
    """Hazard rises with `harm` and the 'yes' level of `smoker`, falls with `protect`; `noise` does nothing."""
    rng = np.random.RandomState(seed)
    num = rng.normal(size=(n, 3)).astype(np.float32)  # harm, noise, protect
    smoker = (rng.uniform(size=n) < 0.3).astype(np.int8)
    rate = 0.03 * np.exp(0.8 * num[:, 0] - 0.6 * num[:, 2] + 0.7 * smoker)
    t = rng.exponential(1.0 / rate)
    died = t < 15.0
    return dict(
        ids=np.array([str(i) for i in range(n)]), weight=np.ones(n), num=num,
        num_names=np.array(["harm", "noise", "protect"]), cat=smoker[:, None],
        cat_names=np.array(["smoker"]), cat_levels=np.array(json.dumps({"smoker": ["no", "yes"]})),
        time=np.minimum(t, 15.0), cause=np.where(died, 0, -1).astype(np.int8))


class Drivers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = cohort()
        rows = np.arange(len(data["time"]))
        train, test = rows[: 2200], rows[2200:]
        ctx = models.Context(data, train, test, seed=1, horizons={})
        cls.result = drivers.analyse(data, ctx, n_permutations=5, seed=3)

    def test_variables_that_move_the_hazard_outrank_noise(self):
        drop = self.result["importance"]
        self.assertGreater(drop["harm"], 0.02)
        self.assertGreater(drop["protect"], 0.01)
        self.assertGreater(drop["smoker"], 0.005)
        self.assertLess(abs(drop["noise"]), 0.2 * drop["smoker"])

    def test_contrasts_have_the_direction_of_the_true_effect(self):
        c = self.result["contrast"]
        self.assertGreater(c["harm"]["p90 vs p10"], 3.0)
        self.assertLess(c["protect"]["p90 vs p10"], 0.5)
        self.assertGreater(c["smoker"]["yes vs no"], 1.5)
        self.assertAlmostEqual(c["noise"]["p90 vs p10"], 1.0, delta=0.25)

    def test_a_mostly_missing_column_has_no_contrast(self):
        data = cohort()
        data["num"][:, 1] = np.where(np.arange(len(data["num"])) % 5 == 0, data["num"][:, 1], np.nan)
        test = np.arange(2200, 3000)
        self.assertEqual(drivers.contrast_numeric(lambda d: np.zeros(len(test)), data, test, 1), {})

    def test_early_deaths_are_excluded_and_survivors_kept(self):
        time = np.array([0.5, 1.9, 2.0, 3.0, 1.0])
        cause = np.array([0, 1, 0, -1, -1])
        self.assertEqual(drivers.late_rows(time, cause, 2.0).tolist(), [False, False, True, True, True])


if __name__ == "__main__":
    unittest.main()
