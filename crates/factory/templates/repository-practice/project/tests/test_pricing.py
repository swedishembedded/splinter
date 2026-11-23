# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
import unittest

from shop.pricing import discount, total


class TotalTest(unittest.TestCase):
    def test_empty_basket_costs_nothing(self):
        self.assertEqual(total([]), 0)

    def test_prices_are_added_up(self):
        self.assertEqual(total([2, 3]), 5)


class DiscountTest(unittest.TestCase):
    def test_ten_percent_off(self):
        self.assertEqual(discount(200, 10), 180)

    def test_over_hundred_percent_is_refused(self):
        with self.assertRaises(ValueError):
            discount(10, 101)
