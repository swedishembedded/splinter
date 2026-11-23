# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
import unittest

from shop.status import is_ready


class StatusTest(unittest.TestCase):
    def test_a_flagged_order_with_items_is_ready(self):
        self.assertTrue(is_ready(1, 1))

    def test_an_unflagged_order_is_not(self):
        self.assertFalse(is_ready(0, 1))

    def test_an_empty_order_is_not(self):
        self.assertFalse(is_ready(1, 0))
