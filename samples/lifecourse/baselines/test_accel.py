# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements long-horizon risk prediction from cohort and
# survey data for its clients. If your team needs expertise in turning
# wearable sensor streams into validated risk inputs, you can procure our
# services by sending an email to info@swedishembedded.com.
"""Specification of the accelerometer summaries: python -I test_accel.py"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import accel  # noqa: E402


def week(per_day, days=7, seed=0):
    """Counts for `days` days: the first `per_day` minutes of each day worn at random
    low-to-moderate intensity, the rest of the day not worn (zeros)."""
    rng = np.random.RandomState(seed)
    c = np.zeros(days * accel.DAY)
    for d in range(days):
        c[d * accel.DAY:d * accel.DAY + per_day] = rng.randint(1, 1500, size=per_day)
    return c, np.arange(1, len(c) + 1)


class Accelerometer(unittest.TestCase):
    def test_the_transport_files_zero_is_a_zero_minute(self):
        raw = np.array([5.397605346934028e-79, 3.0, 1391.15, 0.0])
        self.assertEqual(accel.clean_counts(raw).tolist(), [0.0, 3.0, 1391.0, 0.0])
        worn = np.concatenate([np.full(300, 5.397605346934028e-79), np.arange(1.0, 301.0)])
        self.assertTrue(accel.nonwear(accel.clean_counts(worn))[:300].all())

    def test_a_long_zero_run_is_non_wear_and_a_short_one_is_not(self):
        c = np.full(400, 50.0)
        c[100:190] = 0  # 90 minutes
        c[300:330] = 0  # 30 minutes
        w = accel.nonwear(c)
        self.assertTrue(w[100:190].all() and not w[:100].any() and not w[300:330].any())

    def test_a_brief_low_spike_does_not_break_a_non_wear_run_but_a_real_movement_does(self):
        c = np.zeros(300)
        c[150] = 60  # one minute of low counts inside the run
        self.assertTrue(accel.nonwear(c).all())
        c[150] = 500
        w = accel.nonwear(c)
        self.assertTrue(w[:150].all() and w[151:].all() and not w[150])
        # Three minutes of low counts are a real interruption.
        c = np.zeros(300)
        c[150:153] = 60
        self.assertFalse(accel.nonwear(c)[150:153].any())

    def test_days_need_enough_wear_and_subjects_enough_days(self):
        c, m = week(per_day=700, days=7)
        f = accel.summarise(c, m)
        self.assertEqual(f["valid_days"], 7.0)
        self.assertAlmostEqual(f["wear_min_day"], 700.0, delta=40)
        c, m = week(per_day=700, days=3)
        self.assertIsNone(accel.summarise(c, m), "three valid days are too few")
        c, m = week(per_day=500, days=7)
        self.assertIsNone(accel.summarise(c, m), "500 minutes of wear is not a valid day")

    def test_the_summaries_follow_the_counts(self):
        c = np.zeros(7 * accel.DAY)
        for d in range(7):
            day = np.zeros(accel.DAY)
            day[:600] = 50      # sedentary
            day[600:800] = 500  # light
            day[800:900] = 3000 # moderate to vigorous
            c[d * accel.DAY:(d + 1) * accel.DAY] = day
        f = accel.summarise(c, np.arange(1, len(c) + 1))
        self.assertEqual((f["sedentary_min_day"], f["light_min_day"], f["mvpa_min_day"]), (600.0, 200.0, 100.0))
        self.assertEqual(f["wear_min_day"], 900.0)
        self.assertAlmostEqual(f["mean_cpm"], (600 * 50 + 200 * 500 + 100 * 3000) / 900.0)
        self.assertAlmostEqual(f["peak30_cpm"], 3000.0)
        # One sedentary-to-active transition per day, out of 599 sedentary minutes with a successor.
        self.assertAlmostEqual(f["sedentary_to_active"], 1.0 / 600.0, places=6)

    def test_minutes_are_ordered_by_their_recording_index(self):
        c, m = week(per_day=700, days=7, seed=3)
        shuffled = np.random.RandomState(1).permutation(len(c))
        a, b = accel.summarise(c, m), accel.summarise(c[shuffled], m[shuffled])
        for k in a:
            self.assertAlmostEqual(a[k], b[k], places=9)


class Grid(unittest.TestCase):
    def test_the_grid_has_the_log_mean_and_the_worn_share_of_each_bin(self):
        c, m = week(per_day=700, days=7, seed=2)
        g = accel.grid(c, m)
        self.assertEqual(g.shape, (2, accel.BINS))
        # The first bin of a day is worn throughout: the mean of its counts, logged.
        first = c[:accel.BIN_MINUTES]
        self.assertAlmostEqual(float(g[0, 0]), float(np.log1p(first.mean())), places=5)
        self.assertEqual(float(g[1, 0]), 1.0)
        # The tail of the day (zeros for hours) is not worn and has no counts.
        last_bin_of_day = accel.DAY // accel.BIN_MINUTES - 1
        self.assertEqual((float(g[0, last_bin_of_day]), float(g[1, last_bin_of_day])), (0.0, 0.0))

    def test_a_shorter_recording_is_padded_with_not_worn(self):
        c, m = week(per_day=700, days=7, seed=1)
        g = accel.grid(c[:3 * accel.DAY], m[:3 * accel.DAY])
        self.assertEqual(float(g[1, 3 * accel.DAY // accel.BIN_MINUTES:].sum()), 0.0)

    def test_the_order_of_the_rows_does_not_matter(self):
        c, m = week(per_day=700, days=7, seed=4)
        p = np.random.RandomState(0).permutation(len(c))
        np.testing.assert_allclose(accel.grid(c, m), accel.grid(c[p], m[p]))


class Experiment(unittest.TestCase):
    def test_the_block_is_appended_after_the_existing_inputs_and_folds_stay_inside_themselves(self):
        import accel_experiment as ex
        data = dict(num=np.arange(12, dtype=np.float32).reshape(4, 3), num_names=np.array(["a", "b", "c"]))
        block = np.full((4, len(accel.FEATURES)), 7.0)
        d = ex.with_block(data, block)
        self.assertEqual(d["num"].shape, (4, 3 + len(accel.FEATURES)))
        self.assertEqual(list(d["num_names"][:4]), ["a", "b", "c", "accel:" + accel.FEATURES[0]])
        np.testing.assert_array_equal(d["num"][:, :3], data["num"])
        has = np.array([True, False, True, True])
        train, test = ex.restrict(np.array([0, 1, 2]), np.array([3]), has)
        self.assertEqual((train.tolist(), test.tolist()), ([0, 2], [3]))


if __name__ == "__main__":
    unittest.main()
