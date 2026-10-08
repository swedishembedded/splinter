# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements long-horizon risk prediction from cohort and
# survey data for its clients. If your team needs expertise in building,
# validating and proving time-to-event models on real health records, you
# can procure our services by sending an email to info@swedishembedded.com.
"""Specification of the residual network: python -I test_residual.py"""
import json
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models  # noqa: E402
import residual  # noqa: E402
from test_baselines import concordance  # noqa: E402


def world(n, interaction, seed):
    """Rows whose log-hazard is additive in a and b, plus `interaction * a * b`."""
    rng = np.random.RandomState(seed)
    x = rng.normal(size=(n, 4)).astype(np.float32)
    log_hazard = 0.5 * x[:, 0] + 0.4 * x[:, 1] + interaction * x[:, 0] * x[:, 1]
    t = rng.exponential(1.0 / (0.03 * np.exp(log_hazard)))
    died = t < 15
    data = dict(num=x, cat=np.zeros((n, 1), dtype=np.int8), num_names=np.array(["a", "b", "c", "d"]),
                cat_names=np.array(["k"]), cat_levels=np.array(json.dumps({"k": ["u", "v"]})),
                time=np.where(died, t, 15.0), cause=np.where(died, 0, -1).astype(np.int8),
                ids=np.array([f"s{i}" for i in range(n)]))
    return data


def ten_year_concordance(cif, ctx):
    """Share of comparable pairs in which the subject who died first had the higher
    predicted incidence at ten years."""
    t, c = ctx.outcome(ctx.test)
    return concordance(cif[:, 9], t, c >= 0)


class Residual(unittest.TestCase):
    def test_breslow_ties_put_the_whole_tied_group_in_each_risk_set(self):
        import torch
        time = np.array([3.0, 3.0, 2.0, 1.0])
        event = np.array([1, 1, 0, 1])
        lp = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
        got = float(residual.PartialLikelihood(time, event)(lp))
        first = np.log(np.exp(0.1) + np.exp(0.2))
        everyone = np.log(np.exp([0.1, 0.2, 0.3, 0.4]).sum())
        want = -((0.1 - first) + (0.2 - first) + (0.4 - everyone)) / 3
        self.assertAlmostEqual(got, want, places=12)

    def test_a_network_before_training_adds_nothing(self):
        net = residual.make_net(16, 2, 7, seed=1)
        out = residual.predict_offset(net, np.random.RandomState(0).normal(size=(5, 7)))
        np.testing.assert_array_equal(out, np.zeros(5))

    def test_the_blocks_select_their_columns(self):
        names = ["age", "missing:age", "diet:energy_kcal", "missing:diet:energy_kcal",
                 "hist:weight_1y_ago", "sex=male", "smoking=never"]
        picks = {b: [n for n, k in zip(names, residual.block_mask(names, b)) if k] for b in residual.BLOCKS}
        self.assertEqual(picks["exam"], ["age", "missing:age"])
        self.assertEqual(picks["diet"], ["diet:energy_kcal", "missing:diet:energy_kcal"])
        self.assertEqual(picks["history"], ["hist:weight_1y_ago"])
        self.assertEqual(picks["questionnaire"], ["sex=male", "smoking=never"])
        self.assertTrue(residual.block_mask(names, None).all())

    def test_it_finds_an_interaction_the_additive_model_cannot(self):
        data = world(9000, interaction=0.9, seed=3)
        train, test = np.arange(0, 6500), np.arange(6500, 9000)
        spline = models.REGISTRY["spline-cox-net-all"]().fit_predict(
            models.Context(data=data, train=train, test=test, seed=5, horizons={}))[0]
        ctx = models.Context(data=data, train=train, test=test, seed=5, horizons={})
        cif = residual.ResidualCox(64, 2).fit_predict(ctx)[0]
        self.assertGreater(ctx.chosen["epochs"], 0)
        self.assertGreater(ctx.chosen["validation_gain"], 0.0)
        ctx0 = models.Context(data=data, train=train, test=test, seed=5, horizons={})
        base = ten_year_concordance(spline, ctx0)
        self.assertGreater(ten_year_concordance(cif, ctx0), base + 0.01)

    def test_it_does_not_hurt_when_the_truth_is_additive(self):
        data = world(9000, interaction=0.0, seed=4)
        train, test = np.arange(0, 6500), np.arange(6500, 9000)
        ctx = models.Context(data=data, train=train, test=test, seed=5, horizons={})
        spline = models.REGISTRY["spline-cox-net-all"]().fit_predict(
            models.Context(data=data, train=train, test=test, seed=5, horizons={}))[0]
        cif = residual.ResidualCox(64, 2).fit_predict(ctx)[0]
        self.assertGreaterEqual(ctx.chosen["validation_gain"], 0.0, "epoch 0 is always a candidate")
        self.assertGreater(ten_year_concordance(cif, ctx), ten_year_concordance(spline, ctx) - 0.01)

    def test_outcomes_shuffled_among_the_training_subjects_give_no_gain(self):
        data = world(9000, interaction=0.9, seed=3)
        train, test = np.arange(0, 6500), np.arange(6500, 9000)
        ctx = models.Context(data=data, train=train, test=test, seed=5, horizons={})
        spline = models.REGISTRY["spline-cox-net-all"]().fit_predict(
            models.Context(data=data, train=train, test=test, seed=5, horizons={}))[0]
        cif = residual.ResidualCox(64, 2, shuffled=True).fit_predict(ctx)[0]
        self.assertLess(ten_year_concordance(cif, ctx), ten_year_concordance(spline, ctx) + 0.005)

    def test_training_rows_that_are_neither_contiguous_nor_zero_based_work(self):
        # The earlier residual candidate indexed training rows by global row number and
        # crashed on real folds; folds are shuffled and offset here.
        data = world(6000, interaction=0.5, seed=8)
        rng = np.random.RandomState(0)
        order = rng.permutation(np.arange(1000, 6000))
        train, test = np.sort(order[:3500]), np.sort(order[3500:])
        ctx = models.Context(data=data, train=train, test=test, seed=2, horizons={})
        cif = residual.ResidualCox(16, 1).fit_predict(ctx)[0]
        self.assertEqual(cif.shape, (len(test), 15))
        self.assertTrue(np.all(np.diff(cif, axis=1) >= -1e-12) and np.all((cif >= 0) & (cif <= 1)))

    def test_a_series_network_finds_what_only_the_series_holds(self):
        # The hazard depends on the level of the series late in the recording, which no
        # tabular input carries.
        data = world(6000, interaction=0.0, seed=12)
        rng = np.random.RandomState(1)
        seq = rng.normal(size=(6000, 2, 96)).astype(np.float32)
        level = seq[:, 0, 60:].mean(axis=1)
        t = rng.exponential(1.0 / (0.03 * np.exp(0.5 * data["num"][:, 0] + 8.0 * level)))
        died = t < 15
        data["time"], data["cause"] = np.where(died, t, 15.0), np.where(died, 0, -1).astype(np.int8)
        data["seq"] = seq
        train, test = np.arange(0, 4200), np.arange(4200, 6000)
        ctx = models.Context(data=data, train=train, test=test, seed=3, horizons={})
        base = models.REGISTRY["spline-cox-net-all"]().fit_predict(
            models.Context(data=data, train=train, test=test, seed=3, horizons={}))[0]
        cif = residual.ResidualSeq().fit_predict(ctx)[0]
        self.assertGreater(ctx.chosen["epochs"], 0)
        self.assertGreater(ten_year_concordance(cif, ctx), ten_year_concordance(base, ctx) + 0.03)

    def test_the_prediction_of_a_row_ignores_the_other_test_rows(self):
        data = world(5000, interaction=0.5, seed=9)
        train, test = np.arange(0, 3500), np.arange(3500, 5000)
        ctx = models.Context(data=data, train=train, test=test, seed=2, horizons={})
        cif = residual.ResidualCox(16, 1).fit_predict(ctx)[0]
        changed = dict(data, num=data["num"].copy())
        changed["num"][np.delete(test, 0)] = 1e6
        ctx2 = models.Context(data=changed, train=train, test=test, seed=2, horizons={})
        cif2 = residual.ResidualCox(16, 1).fit_predict(ctx2)[0]
        np.testing.assert_allclose(cif2[0], cif[0], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
