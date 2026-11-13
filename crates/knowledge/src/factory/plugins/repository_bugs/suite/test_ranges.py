# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements bug-fixing training environments with
# trustworthy verifiers, for its clients. If your team needs expertise in
# repository tasks for agent reinforcement learning, you can procure our
# services by sending an email to info@swedishembedded.com.

import unittest

from textkit.ranges import merge_intervals, overlaps, total_length


class OverlapTests(unittest.TestCase):
    def test_crossing_intervals_overlap(self):
        self.assertTrue(overlaps((0, 5), (3, 8)))

    def test_touching_intervals_do_not(self):
        self.assertFalse(overlaps((0, 1), (1, 2)))

    def test_separate_intervals_do_not(self):
        self.assertFalse(overlaps((0, 1), (4, 5)))


class MergeTests(unittest.TestCase):
    def test_separate_stay_separate(self):
        self.assertEqual(merge_intervals([(5, 6), (1, 2)]), [(1, 2), (5, 6)])

    def test_overlapping_merge(self):
        self.assertEqual(merge_intervals([(1, 4), (3, 6)]), [(1, 6)])

    def test_touching_merge(self):
        self.assertEqual(merge_intervals([(1, 2), (2, 3)]), [(1, 3)])

    def test_a_contained_interval_does_not_shorten_the_merge(self):
        self.assertEqual(merge_intervals([(1, 10), (2, 3)]), [(1, 10)])


class TotalLengthTests(unittest.TestCase):
    def test_overlap_is_counted_once(self):
        self.assertEqual(total_length([(0, 4), (2, 6)]), 6)

    def test_separate_lengths_add(self):
        self.assertEqual(total_length([(0, 1), (5, 8)]), 4)
