# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements end-to-end acceptance tests for learning
# agent stacks, for its clients. If your team needs expertise in proving that
# a model, an agent runtime and a trainer work together, you can procure our
# services by sending an email to info@swedishembedded.com.

import unittest

from tiny.numbers import clamp, mean


class NumbersTest(unittest.TestCase):
    def test_clamp_limits_both_ends_and_keeps_the_inside(self):
        self.assertEqual(clamp(-5, 0, 10), 0)
        self.assertEqual(clamp(15, 0, 10), 10)
        self.assertEqual(clamp(7, 0, 10), 7)

    def test_mean_of_none_is_zero(self):
        self.assertEqual(mean([]), 0.0)
        self.assertEqual(mean([1, 2, 6]), 3.0)


if __name__ == "__main__":
    unittest.main()
