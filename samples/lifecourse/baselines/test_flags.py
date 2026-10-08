# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements long-horizon risk prediction from cohort and
# survey data for its clients. If your team needs expertise in building,
# validating and proving time-to-event models on real health records, you
# can procure our services by sending an email to info@swedishembedded.com.
"""Specification of the certificate-flag outcome: python -I test_flags.py"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flags  # noqa: E402
import models  # noqa: E402


def cohort(n=3000, seed=5):
    """Rows whose risk of dying WITH the flag rises with input `a`; deaths of
    low-`a` subjects mostly carry no flag."""
    rng = np.random.RandomState(seed)
    x = rng.normal(size=(n, 2)).astype(np.float32)
    t = rng.exponential(1.0 / (0.04 * np.exp(0.8 * x[:, 0])))
    died = t < 15
    time = np.where(died, t, 15.0)
    carries = rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-2.0 * x[:, 0]))
    data = dict(num=x, cat=np.zeros((n, 1), dtype=np.int8), num_names=np.array(["a", "b"]),
                cat_names=np.array(["c"]), cat_levels=np.array(json.dumps({"c": ["u", "v"]})),
                time=time, cause=np.where(died, 0, -1).astype(np.int8),
                ids=np.array([f"s{i}" for i in range(n)]))
    flag = np.where(died, carries.astype(np.int8), -1).astype(np.int8)
    data["flag_diabetes"], data["flag_hypertension"] = flag, flag.copy()
    return data


class Flags(unittest.TestCase):
    def test_death_with_the_flag_competes_with_death_without_it(self):
        time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cause = np.array([0, 1, 2, 0, -1], dtype=np.int8)
        flag = np.array([1, 0, -1, -1, -1], dtype=np.int8)
        _, out = flags.flag_outcome(time, cause, flag)
        # With the flag, without it, a death with no multiple-cause data (censored,
        # not called flag-free), another such death, and a survivor.
        self.assertEqual(out.tolist(), [0, 1, -1, -1, -1])

    def test_the_two_outcomes_add_up_to_the_all_cause_incidence(self):
        dh = np.full((1, len(models.GRID) - 1, 2), 0.01 / 12)
        dh[:, :, 0] *= 3
        all_cause, per = models.cumulative_incidence(dh, names=("flag", "other"))
        np.testing.assert_allclose(per["flag"] + per["other"], all_cause, atol=1e-12)
        np.testing.assert_allclose(per["flag"][0], 3 * per["other"][0], atol=1e-12)

    def test_a_subject_who_dies_of_what_the_flag_follows_gets_a_higher_flag_incidence(self):
        data = cohort()
        train, test = np.arange(0, 2200), np.arange(2200, 3000)
        ctx = models.Context(data=data, train=train, test=test, seed=3, horizons={})
        curves = flags.FlagCox("all").fit_predict(ctx)["diabetes"]
        self.assertTrue(np.all(np.diff(curves, axis=1) >= -1e-12), "an incidence never falls")
        high = data["num"][test, 0] > 0.5
        low = data["num"][test, 0] < -0.5
        self.assertGreater(curves[high, 9].mean(), 2 * curves[low, 9].mean())

    def test_the_share_model_never_exceeds_the_all_cause_incidence(self):
        data = cohort()
        train, test = np.arange(0, 2200), np.arange(2200, 3000)
        ctx = models.Context(data=data, train=train, test=test, seed=3, horizons={})
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "cs-cox-all"))
            base = np.cumsum(np.full(15, 0.02))
            with open(os.path.join(d, "cs-cox-all", "r0-k0.jsonl"), "w") as f:
                for s in data["ids"][test]:
                    f.write(json.dumps({"subject_id": str(s), "cif": base.tolist()}) + "\n")
            curves = flags.FlagShare("all", d).fit_predict(ctx, 0, 0)["diabetes"]
        self.assertTrue(np.all(curves <= base[None, :] + 1e-12))
        self.assertGreater(curves[data["num"][test, 0] > 0.5, 9].mean(),
                           curves[data["num"][test, 0] < -0.5, 9].mean())


if __name__ == "__main__":
    unittest.main()
