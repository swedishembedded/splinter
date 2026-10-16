# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
import unittest

from ttlcache.lru_store import LruStore, lru_check_capacity


def filled(capacity, *keys):
    store = LruStore(capacity)
    for key in keys:
        store.put(key, key.upper())
    return store


class LruStoreTests(unittest.TestCase):
    def test_capacity_check_returns_valid_value(self):
        self.assertEqual(lru_check_capacity(3), 3)

    def test_capacity_one_is_accepted(self):
        self.assertEqual(lru_check_capacity(1), 1)

    def test_capacity_zero_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            lru_check_capacity(0)
        self.assertEqual(str(caught.exception), "capacity must be at least 1")

    def test_negative_capacity_is_rejected(self):
        with self.assertRaises(ValueError):
            LruStore(-4)

    def test_float_capacity_is_rejected(self):
        with self.assertRaises(TypeError) as caught:
            LruStore(2.0)
        self.assertEqual(str(caught.exception), "capacity must be an integer")

    def test_capacity_property_reports_limit(self):
        self.assertEqual(LruStore(7).capacity, 7)

    def test_empty_store_has_no_entries(self):
        store = LruStore(2)
        self.assertEqual(len(store), 0)
        self.assertEqual(store.keys_most_recent_first(), [])

    def test_stored_value_is_returned(self):
        self.assertEqual(filled(2, "a").get("a"), "A")

    def test_missing_key_returns_default(self):
        self.assertEqual(LruStore(2).get("x", "fallback"), "fallback")

    def test_missing_key_default_is_none(self):
        self.assertIsNone(LruStore(2).get("x"))

    def test_stored_none_is_distinct_from_miss(self):
        store = LruStore(2)
        store.put("n", None)
        self.assertIn("n", store)
        self.assertEqual(store.get("n", "d"), None)

    def test_get_promotes_entry_to_front(self):
        store = filled(3, "a", "b", "c")
        store.get("a")
        self.assertEqual(store.keys_most_recent_first(), ["a", "c", "b"])

    def test_peek_leaves_order_untouched(self):
        store = filled(3, "a", "b", "c")
        self.assertEqual(store.peek("a"), "A")
        self.assertEqual(store.keys_most_recent_first(), ["c", "b", "a"])

    def test_peek_missing_key_gives_default(self):
        store = LruStore(2)
        self.assertEqual(store.peek("q", 9), 9)
        self.assertIsNone(store.peek("q"))

    def test_membership_does_not_promote(self):
        store = filled(3, "a", "b", "c")
        self.assertIn("a", store)
        self.assertNotIn("zz", store)
        self.assertEqual(store.keys_most_recent_first(), ["c", "b", "a"])

    def test_put_new_key_goes_to_front(self):
        store = filled(3, "a", "b")
        self.assertEqual(store.keys_most_recent_first(), ["b", "a"])

    def test_overflow_evicts_least_recent_entry(self):
        store = filled(2, "a", "b", "c")
        self.assertEqual(store.keys_most_recent_first(), ["c", "b"])
        self.assertNotIn("a", store)

    def test_store_at_capacity_keeps_all_entries(self):
        store = filled(3, "a", "b", "c")
        self.assertEqual(len(store), 3)

    def test_recent_read_protects_entry_from_eviction(self):
        store = filled(2, "a", "b")
        store.get("a")
        store.put("c", "C")
        self.assertIn("a", store)
        self.assertNotIn("b", store)

    def test_overwrite_updates_value_without_growth(self):
        store = filled(2, "a", "b")
        store.put("a", "new")
        self.assertEqual(store.peek("a"), "new")
        self.assertEqual(len(store), 2)

    def test_overwrite_promotes_entry_to_front(self):
        store = filled(3, "a", "b", "c")
        store.put("a", 1)
        self.assertEqual(store.keys_most_recent_first(), ["a", "c", "b"])

    def test_capacity_one_store_holds_latest_only(self):
        store = filled(1, "a", "b")
        self.assertEqual(store.keys_most_recent_first(), ["b"])

    def test_eviction_callback_receives_victim(self):
        seen = []
        store = LruStore(1, on_evict=lambda k, v: seen.append((k, v)))
        store.put("a", 1)
        store.put("b", 2)
        self.assertEqual(seen, [("a", 1)])

    def test_overwrite_does_not_fire_eviction_callback(self):
        seen = []
        store = LruStore(1, on_evict=lambda k, v: seen.append(k))
        store.put("a", 1)
        store.put("a", 2)
        self.assertEqual(seen, [])

    def test_delete_existing_key_reports_true(self):
        store = filled(3, "a", "b")
        self.assertIs(store.delete("a"), True)
        self.assertEqual(store.keys_most_recent_first(), ["b"])
        self.assertEqual(len(store), 1)

    def test_delete_missing_key_reports_false(self):
        self.assertIs(filled(3, "a").delete("zz"), False)

    def test_delete_skips_eviction_callback(self):
        seen = []
        store = LruStore(2, on_evict=lambda k, v: seen.append(k))
        store.put("a", 1)
        store.delete("a")
        self.assertEqual(seen, [])

    def test_pop_oldest_returns_pair_and_removes(self):
        store = filled(3, "a", "b", "c")
        self.assertEqual(store.pop_oldest(), ("a", "A"))
        self.assertEqual(store.keys_most_recent_first(), ["c", "b"])
        self.assertEqual(len(store), 2)

    def test_pop_oldest_on_empty_store_raises(self):
        with self.assertRaises(KeyError) as caught:
            LruStore(2).pop_oldest()
        self.assertEqual(caught.exception.args[0], "store is empty")

    def test_shrinking_capacity_evicts_oldest_entries(self):
        seen = []
        store = LruStore(4, on_evict=lambda k, v: seen.append(k))
        for key in "abcd":
            store.put(key, 0)
        store.resize(2)
        self.assertEqual(store.keys_most_recent_first(), ["d", "c"])
        self.assertEqual(seen, ["a", "b"])
        self.assertEqual(store.capacity, 2)

    def test_growing_capacity_keeps_everything(self):
        store = filled(2, "a", "b")
        store.resize(5)
        store.put("c", "C")
        self.assertEqual(len(store), 3)

    def test_resize_to_invalid_capacity_is_rejected(self):
        store = filled(2, "a")
        with self.assertRaises(ValueError):
            store.resize(0)
        self.assertEqual(store.capacity, 2)

    def test_resize_to_same_capacity_evicts_nothing(self):
        store = filled(2, "a", "b")
        store.resize(2)
        self.assertEqual(len(store), 2)

    def test_clear_removes_all_without_callbacks(self):
        seen = []
        store = LruStore(3, on_evict=lambda k, v: seen.append(k))
        store.put("a", 1)
        store.put("b", 2)
        store.clear()
        self.assertEqual(len(store), 0)
        self.assertEqual(seen, [])

    def test_store_is_reusable_after_clear(self):
        store = filled(2, "a", "b")
        store.clear()
        store.put("c", "C")
        self.assertEqual(store.keys_most_recent_first(), ["c"])


if __name__ == "__main__":
    unittest.main()
