# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements long-horizon risk prediction from cohort and
# survey data for its clients. If your team needs expertise in building,
# validating and proving time-to-event models on real health records, you
# can procure our services by sending an email to info@swedishembedded.com.
"""Specification of the acceleration association: python -I test_assoc.py"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import assoc  # noqa: E402


def cohort(effect, n=6000, seed=2):
    rng = np.random.RandomState(seed)
    age = rng.uniform(30, 80, n)
    accel5 = rng.normal(0, 1, n)
    t = rng.exponential(1.0 / (0.01 * np.exp(0.04 * (age - 50) + effect * accel5)))
    died = t < 12
    flagged = rng.uniform(size=n) < 0.3
    return pd.DataFrame(dict(
        subject_id=[f"s{i}" for i in range(n)], age=age, female=rng.randint(0, 2, n).astype(float),
        accel5=accel5, time=np.minimum(t, 12.0), cause=np.where(died, rng.randint(0, 3, n), -1),
        weight=np.ones(n), cluster=np.arange(n) // 10, diabetes=flagged, hypertension=~flagged,
        available=np.ones(n, dtype=bool)))


class Association(unittest.TestCase):
    def test_a_real_effect_is_recovered_and_no_effect_gives_a_ratio_of_one(self):
        hr, lo, hi, events = assoc.hazard_ratio(cohort(0.4), "all-cause")
        self.assertTrue(lo < np.exp(0.4) < hi and events > 300, (hr, lo, hi))
        hr, lo, hi, _ = assoc.hazard_ratio(cohort(0.0), "all-cause")
        self.assertTrue(lo < 1.0 < hi, (hr, lo, hi))

    def test_a_death_without_multiple_cause_data_is_not_called_flag_free(self):
        df = cohort(0.0, n=200)
        df.loc[df["cause"] >= 0, "available"] = False
        self.assertEqual(int(assoc.event_of(df, "diabetes mention").sum()), 0)
        self.assertGreater(int(assoc.event_of(df, "all-cause").sum()), 0)


if __name__ == "__main__":
    unittest.main()
