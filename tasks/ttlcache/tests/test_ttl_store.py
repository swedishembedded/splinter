# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
import unittest

from ttlcache.ttl_store import TtlStore, ttl_check_lifetime


class FakeClock:
    def __init__(self, now=0):
        self.now = now

    def __call__(self):
        return self.now


class TtlStoreTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock(100)
        self.store = TtlStore(self.clock)

    def test_lifetime_check_accepts_positive_number(self):
        self.assertEqual(ttl_check_lifetime(2.5), 2.5)

    def test_lifetime_check_accepts_none(self):
        self.assertIsNone(ttl_check_lifetime(None))

    def test_lifetime_check_rejects_zero(self):
        with self.assertRaises(ValueError) as caught:
            ttl_check_lifetime(0)
        self.assertEqual(str(caught.exception), "ttl must be positive")

    def test_lifetime_check_rejects_negative(self):
        with self.assertRaises(ValueError):
            ttl_check_lifetime(-1)

    def test_smallest_positive_lifetime_is_accepted(self):
        self.assertEqual(ttl_check_lifetime(1), 1)

    def test_non_callable_clock_is_rejected(self):
        with self.assertRaises(TypeError) as caught:
            TtlStore(42)
        self.assertEqual(str(caught.exception), "clock must be callable")

    def test_invalid_default_lifetime_is_rejected(self):
        with self.assertRaises(ValueError):
            TtlStore(self.clock, default_ttl=0)

    def test_entry_without_ttl_never_expires(self):
        self.store.put("a", 1)
        self.clock.now = 10**9
        self.assertEqual(self.store.get("a"), 1)

    def test_entry_is_readable_before_deadline(self):
        self.store.put("a", 1, ttl=10)
        self.clock.now = 109
        self.assertEqual(self.store.get("a"), 1)

    def test_entry_expires_exactly_at_deadline(self):
        self.store.put("a", 1, ttl=10)
        self.clock.now = 110
        self.assertIsNone(self.store.get("a"))

    def test_entry_stays_gone_after_deadline(self):
        self.store.put("a", 1, ttl=10)
        self.clock.now = 500
        self.assertEqual(self.store.get("a", "gone"), "gone")

    def test_missing_key_returns_default(self):
        self.assertEqual(self.store.get("nope", 5), 5)
        self.assertIsNone(self.store.get("nope"))

    def test_default_ttl_applies_to_plain_puts(self):
        store = TtlStore(self.clock, default_ttl=5)
        store.put("a", 1)
        self.clock.now = 105
        self.assertNotIn("a", store)

    def test_explicit_ttl_overrides_default(self):
        store = TtlStore(self.clock, default_ttl=5)
        store.put("a", 1, ttl=50)
        self.clock.now = 120
        self.assertEqual(store.get("a"), 1)

    def test_put_rejects_non_positive_ttl(self):
        with self.assertRaises(ValueError):
            self.store.put("a", 1, ttl=-3)
        self.assertNotIn("a", self.store)

    def test_overwrite_restarts_the_lifetime(self):
        self.store.put("a", 1, ttl=10)
        self.clock.now = 108
        self.store.put("a", 2, ttl=10)
        self.clock.now = 115
        self.assertEqual(self.store.get("a"), 2)

    def test_membership_is_false_once_expired(self):
        self.store.put("a", 1, ttl=1)
        self.assertIn("a", self.store)
        self.clock.now = 101
        self.assertNotIn("a", self.store)

    def test_length_excludes_expired_entries(self):
        self.store.put("a", 1, ttl=5)
        self.store.put("b", 2, ttl=50)
        self.store.put("c", 3)
        self.clock.now = 110
        self.assertEqual(len(self.store), 2)

    def test_delete_live_entry_reports_true(self):
        self.store.put("a", 1)
        self.assertIs(self.store.delete("a"), True)
        self.assertNotIn("a", self.store)

    def test_delete_missing_entry_reports_false(self):
        self.assertIs(self.store.delete("a"), False)

    def test_delete_expired_entry_reports_false(self):
        self.store.put("a", 1, ttl=2)
        self.clock.now = 102
        self.assertIs(self.store.delete("a"), False)

    def test_time_to_live_counts_down(self):
        self.store.put("a", 1, ttl=30)
        self.clock.now = 112
        self.assertEqual(self.store.time_to_live("a"), 18)

    def test_time_to_live_is_none_without_expiry(self):
        self.store.put("a", 1)
        self.assertIsNone(self.store.time_to_live("a"))

    def test_time_to_live_of_missing_key_raises(self):
        with self.assertRaises(KeyError) as caught:
            self.store.time_to_live("a")
        self.assertEqual(caught.exception.args[0], "a")

    def test_time_to_live_of_expired_key_raises(self):
        self.store.put("a", 1, ttl=1)
        self.clock.now = 101
        with self.assertRaises(KeyError):
            self.store.time_to_live("a")

    def test_touch_extends_a_live_entry(self):
        self.store.put("a", 1, ttl=10)
        self.clock.now = 105
        self.assertIs(self.store.touch("a", ttl=10), True)
        self.assertEqual(self.store.time_to_live("a"), 10)
        self.assertEqual(self.store.get("a"), 1)

    def test_touch_without_ttl_uses_default(self):
        store = TtlStore(self.clock, default_ttl=7)
        store.put("a", 1, ttl=100)
        self.assertTrue(store.touch("a"))
        self.assertEqual(store.time_to_live("a"), 7)

    def test_touch_of_missing_key_reports_false(self):
        self.assertIs(self.store.touch("a", ttl=5), False)

    def test_touch_cannot_revive_expired_entry(self):
        self.store.put("a", 1, ttl=1)
        self.clock.now = 101
        self.assertFalse(self.store.touch("a", ttl=10))
        self.assertNotIn("a", self.store)

    def test_touch_rejects_non_positive_ttl(self):
        self.store.put("a", 1, ttl=10)
        with self.assertRaises(ValueError):
            self.store.touch("a", ttl=0)

    def test_sweep_removes_only_expired_entries(self):
        self.store.put("a", 1, ttl=5)
        self.store.put("b", 2, ttl=50)
        self.store.put("c", 3)
        self.clock.now = 105
        self.assertEqual(self.store.sweep(), 1)
        self.assertEqual(self.store.get("b"), 2)

    def test_sweep_on_fresh_store_removes_nothing(self):
        self.store.put("a", 1, ttl=5)
        self.assertEqual(self.store.sweep(), 0)

    def test_sweep_limit_caps_removals(self):
        for key in "abc":
            self.store.put(key, 0, ttl=1)
        self.clock.now = 200
        self.assertEqual(self.store.sweep(limit=2), 2)
        self.assertEqual(self.store.sweep(), 1)

    def test_sweep_limit_of_one_is_allowed(self):
        self.store.put("a", 0, ttl=1)
        self.store.put("b", 0, ttl=1)
        self.clock.now = 200
        self.assertEqual(self.store.sweep(limit=1), 1)

    def test_sweep_rejects_zero_limit(self):
        with self.assertRaises(ValueError) as caught:
            self.store.sweep(limit=0)
        self.assertEqual(str(caught.exception), "limit must be positive")

    def test_earliest_deadline_finds_the_minimum(self):
        self.store.put("a", 1, ttl=30)
        self.store.put("b", 2, ttl=10)
        self.store.put("c", 3)
        self.assertEqual(self.store.earliest_deadline(), 110)

    def test_earliest_deadline_is_none_without_expiring_entries(self):
        self.assertIsNone(self.store.earliest_deadline())
        self.store.put("a", 1)
        self.assertIsNone(self.store.earliest_deadline())

    def test_live_entries_lists_remaining_seconds(self):
        self.store.put("a", 1, ttl=20)
        self.store.put("b", 2)
        self.clock.now = 105
        self.assertEqual(self.store.live_entries(), [("a", 1, 15), ("b", 2, None)])

    def test_live_entries_omit_expired_ones(self):
        self.store.put("a", 1, ttl=2)
        self.store.put("b", 2, ttl=20)
        self.clock.now = 102
        self.assertEqual(self.store.live_entries(), [("b", 2, 18)])


if __name__ == "__main__":
    unittest.main()
