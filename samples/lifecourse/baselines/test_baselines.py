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


def concordance(risk, time, event):
    """Fraction of comparable held-out pairs (a subject died before the
    other's death or censoring, or both died at different times) in which the
    one who died first has the higher predicted risk; ties count one half."""
    ok = tot = 0
    n = len(time)
    for i in range(n):
        for j in range(i + 1, n):
            ti, tj, ei, ej = time[i], time[j], event[i], event[j]
            if ti == tj:
                continue
            if ei and ej:
                comparable = True
            elif ei and not ej:
                comparable = ti < tj
            elif ej and not ei:
                comparable = tj < ti
            else:
                comparable = False
            if not comparable:
                continue
            tot += 1
            lo, hi = (i, j) if ti < tj else (j, i)
            if risk[lo] > risk[hi]:
                ok += 1
            elif risk[lo] == risk[hi]:
                ok += 0.5  # a tie ranks nobody: half a pair, so a constant score gives 0.5
    return ok / max(tot, 1)


def u_shaped(n, seed):
    """Synthetic rows whose log-hazard is a U-shaped function of input `a`."""
    rng = np.random.RandomState(seed)
    x = rng.normal(size=(n, 2)).astype(np.float32)
    cat = np.zeros((n, 1), dtype=np.int8)
    t = rng.weibull(1.2, size=n) * np.exp((x[:, 0] ** 2 - 2.0) / 2.0)
    died = rng.uniform(size=n) < 1.0 / (1.0 + np.exp(t))
    time = np.where(died, t, np.maximum(t, 15.0))
    cause = np.where(died, 0, -1).astype(np.int8)
    return (dict(num=x, cat=cat, num_names=np.array(["a", "b"]), cat_names=np.array(["c"]),
                 cat_levels=np.array(json.dumps({"c": ["u", "v"]}))), time, cause)


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

    def test_the_age_sex_input_set_is_age_and_sex_alone(self):
        d = dict(num_names=np.array(["sbp", "age", "bmi"]), cat_names=np.array(["smoking", "sex"]))
        num, cat = features.columns(d, "agesex")
        self.assertEqual(([d["num_names"][i] for i in num], [d["cat_names"][i] for i in cat]),
                         (["age"], ["sex"]))
        self.assertIn("cs-cox-agesex", models.REGISTRY)

    def test_dropping_an_input_removes_exactly_that_input_from_the_design(self):
        data, time, cause = u_shaped(1500, seed=3)
        data["time"], data["cause"] = time.astype(np.float64), cause
        data["cat_names"] = np.array(["told_weak_kidneys"])
        ctx = models.Context(data=data, train=np.arange(0, 1000), test=np.arange(1000, 1500), seed=1, horizons={})
        model = models.REGISTRY["spline-cox-net-without-told_weak_kidneys"]()
        cif = model.fit_predict(ctx)[0]
        self.assertEqual(model.inputs[1], ())
        self.assertEqual(model.inputs[0], ("a", "b"))
        self.assertEqual(cif.shape, (500, 15))
        data["cat_names"] = np.array(["sex"])
        with self.assertRaises(ValueError):
            models.REGISTRY["spline-cox-net-without-told_weak_kidneys"]().fit_predict(ctx)

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

    def test_the_spline_cox_beats_the_linear_cox_on_a_u_shaped_log_hazard(self):
        data, time, cause = u_shaped(4000, seed=11)
        data["time"], data["cause"] = time.astype(np.float64), cause
        train = np.arange(0, 3000)
        test = np.arange(3000, 4000)
        ctx = models.Context(data=data, train=train, test=test, seed=7, horizons={})
        t_te, c_te = ctx.outcome(test)
        event = c_te >= 0
        spline = models.REGISTRY["spline-cox-net-all"]().fit_predict(ctx)[0]
        ctx2 = models.Context(data=data, train=train, test=test, seed=7, horizons={})
        linear = models.REGISTRY["cox-net-all"]().fit_predict(ctx2)[0]
        ten = int(np.searchsorted(models.YEARS, 10))  # index of the 10-year column
        self.assertGreater(concordance(spline[:, ten], t_te, event),
                           concordance(linear[:, ten], t_te, event))

    def test_the_spline_predictions_of_a_row_ignore_the_other_test_rows(self):
        data, time, cause = u_shaped(4000, seed=11)
        data["time"], data["cause"] = time.astype(np.float64), cause
        train = np.arange(0, 3000)
        test = np.arange(3000, 4000)
        ctx = models.Context(data=data, train=train, test=test, seed=7, horizons={})
        cif = models.REGISTRY["spline-cox-net-all"]().fit_predict(ctx)[0]
        kept = int(test[0])
        changed = dict(data, num=data["num"].copy())
        changed["num"][np.delete(test, 0)] = 1e6  # extreme values on the other held-out rows
        ctx2 = models.Context(data=changed, train=train, test=test, seed=7, horizons={})
        cif2 = models.REGISTRY["spline-cox-net-all"]().fit_predict(ctx2)[0]
        np.testing.assert_allclose(cif2[0], cif[0], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
