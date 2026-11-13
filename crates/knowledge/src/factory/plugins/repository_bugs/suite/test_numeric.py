# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements bug-fixing training environments with
# trustworthy verifiers, for its clients. If your team needs expertise in
# repository tasks for agent reinforcement learning, you can procure our
# services by sending an email to info@swedishembedded.com.

import unittest

from textkit.numeric import clamp, mean, median, percent


class ClampTests(unittest.TestCase):
    def test_inside_is_unchanged(self):
        self.assertEqual(clamp(3, 0, 5), 3)

    def test_below_gives_the_lower_bound(self):
        self.assertEqual(clamp(-4, 0, 5), 0)

    def test_above_gives_the_upper_bound(self):
        self.assertEqual(clamp(9, 0, 5), 5)

    def test_reversed_bounds_are_refused(self):
        with self.assertRaises(ValueError):
            clamp(1, 5, 0)


class MeanTests(unittest.TestCase):
    def test_exact_division(self):
        self.assertEqual(mean([2, 4]), 3)

    def test_fraction_is_kept(self):
        self.assertEqual(mean([1, 2]), 1.5)

    def test_empty_is_refused(self):
        with self.assertRaises(ValueError):
            mean([])


class MedianTests(unittest.TestCase):
    def test_odd_count(self):
        self.assertEqual(median([5, 1, 3]), 3)

    def test_even_count_averages_the_middle_pair(self):
        self.assertEqual(median([4, 1, 3, 2]), 2.5)

    def test_empty_is_refused(self):
        with self.assertRaises(ValueError):
            median([])


class PercentTests(unittest.TestCase):
    def test_one_decimal(self):
        self.assertEqual(percent(1, 3), 33.3)

    def test_whole(self):
        self.assertEqual(percent(5, 5), 100.0)

    def test_zero_whole_is_zero(self):
        self.assertEqual(percent(3, 0), 0.0)
