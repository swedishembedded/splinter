# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements long-horizon risk prediction from cohort and
# survey data for its clients. If your team needs expertise in building,
# validating and proving time-to-event models on real health records, you
# can procure our services by sending an email to info@swedishembedded.com.
"""Specification of the baseline harness: python -I test_baselines.py"""
import json
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features  # noqa: E402
import models  # noqa: E402


def toy(n=400, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.normal(size=(n, 2)).astype(np.float32)
    x[::7, 0] = np.nan
    cat = rng.randint(-1, 2, size=(n, 1)).astype(np.int8)
    return dict(num=x, cat=cat, num_names=np.array(["a", "b"]), cat_names=np.array(["c"]),
                cat_levels=np.array(json.dumps({"c": ["u", "v"]})))


class Harness(unittest.TestCase):
    def test_preprocessing_uses_only_the_rows_it_is_fitted_on(self):
        data = toy()
        rows = np.arange(0, 200)
        a = features.Preprocessor(data, "all").fit(data, rows)
        changed = dict(data, num=data["num"].copy())
        changed["num"][200:] = 1e6  # rows outside the fit, such as a test fold
        b = features.Preprocessor(changed, "all").fit(changed, rows)
        np.testing.assert_array_equal(a.transform(data, rows), b.transform(changed, rows))
        np.testing.assert_array_equal(a.median, b.median)
        np.testing.assert_array_equal(a.scale, b.scale)

    def test_a_training_subsample_must_stay_inside_the_fold(self):
        import tempfile
        import run
        data = dict(ids=np.array(["a", "b", "c", "d"]))
        train = np.array([0, 1, 2])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "r0-k0.ids")
            with open(path, "w") as f:
                f.write("c\na\n")
            np.testing.assert_array_equal(run.restrict(data, train, path), np.array([0, 2]))
            with open(path, "w") as f:
                f.write("a\nd\n")  # d is a test subject
            with self.assertRaises(ValueError):
                run.restrict(data, train, path)

    def test_missing_values_are_imputed_and_flagged(self):
        data = toy()
        rows = np.arange(len(data["num"]))
        prep = features.Preprocessor(data, "all").fit(data, rows)
        x = prep.transform(data, rows)
        self.assertTrue(np.all(np.isfinite(x)))
        self.assertIn("missing:a", prep.names(data))

    def test_causes_add_up_to_the_all_cause_incidence(self):
        rng = np.random.RandomState(1)
        time = rng.exponential(10.0, size=2000)
        cause = np.where(time < 14, rng.randint(0, 3, size=2000), -1).astype(np.int8)
        surv, cif = models.aalen_johansen(time, cause, models.YEARS.astype(float))
        np.testing.assert_allclose(cif.sum(axis=0), 1.0 - surv, atol=1e-12)
        dh = np.full((2, len(models.GRID) - 1, 3), 0.01 / 12)
        all_cause, per_cause = models.cumulative_incidence(dh)
        np.testing.assert_allclose(all_cause[0], 1.0 - np.exp(-0.03 * models.YEARS), atol=1e-9)
        np.testing.assert_allclose(per_cause["death:cvd"][0] * 3, all_cause[0], atol=1e-12)

    def test_without_censoring_every_label_counts_once(self):
        time = np.array([1.0, 3.0, 8.0, 12.0])
        cause = np.array([0, -1, 1, -1], dtype=np.int8)
        label, weight, known = models.ipcw_sample(time, cause, 5.0)
        self.assertEqual(label.tolist(), [1, 0, 0, 0])
        # censored at 3 years, before the horizon: no label; the rest are
        # weighted up for the censoring that removed it.
        self.assertEqual(known.tolist(), [True, False, True, True])
        self.assertTrue(np.all(weight[known] >= 1.0))
        self.assertEqual(weight[1], 0.0)

    def test_the_partial_likelihood_prefers_the_true_risk_order(self):
        rng = np.random.RandomState(2)
        x = rng.normal(size=3000)
        time = rng.exponential(1.0 / np.exp(x))
        event = np.ones(3000, dtype=bool)
        self.assertGreater(models.partial_loglik(x, time, event), models.partial_loglik(-x, time, event))


if __name__ == "__main__":
    unittest.main()
