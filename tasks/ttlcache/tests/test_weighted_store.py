# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
import unittest

from ttlcache.weighted_store import WeightedStore, weight_check


class T(unittest.TestCase):
    def test_weight_check_accepts_positive_integer(self):
        self.assertEqual(weight_check(5), 5)

    def test_weight_check_rejects_zero(self):
        with self.assertRaises(ValueError) as caught:
            weight_check(0)
        self.assertEqual(str(caught.exception), "weight must be at least 1")

    def test_weight_check_rejects_non_integer(self):
        with self.assertRaises(TypeError) as caught:
            weight_check(1.5)
        self.assertEqual(str(caught.exception), "weight must be an integer")

    def test_budget_must_be_valid_weight(self):
        with self.assertRaises(ValueError):
            WeightedStore(0)

    def test_fresh_store_reports_full_headroom(self):
        store = WeightedStore(10)
        self.assertEqual(store.max_weight, 10)
        self.assertEqual(store.total_weight, 0)
        self.assertEqual(store.remaining_weight(), 10)
        self.assertEqual(len(store), 0)

    def test_put_accumulates_total_weight(self):
        store = WeightedStore(10)
        store.put("a", 1, 3)
        store.put("b", 2, 4)
        self.assertEqual(store.total_weight, 7)
        self.assertEqual(store.remaining_weight(), 3)

    def test_missing_key_returns_default(self):
        store = WeightedStore(10)
        self.assertEqual(store.get("x", 8), 8)
        self.assertIsNone(store.get("x"))

    def test_weight_of_reports_entry_weight(self):
        store = WeightedStore(10)
        store.put("a", 1, 6)
        self.assertEqual(store.weight_of("a"), 6)

    def test_weight_of_missing_key_raises(self):
        with self.assertRaises(KeyError) as caught:
            WeightedStore(10).weight_of("zz")
        self.assertEqual(caught.exception.args[0], "zz")

    def test_exact_fit_does_not_evict(self):
        store = WeightedStore(10)
        store.put("a", 1, 4)
        store.put("b", 2, 6)
        self.assertEqual(store.keys_oldest_first(), ["a", "b"])
        self.assertEqual(store.remaining_weight(), 0)

    def test_one_unit_over_budget_evicts_oldest(self):
        store = WeightedStore(10)
        store.put("a", 1, 4)
        store.put("b", 2, 6)
        store.put("c", 3, 1)
        self.assertEqual(store.keys_oldest_first(), ["b", "c"])
        self.assertEqual(store.total_weight, 7)

    def test_large_entry_evicts_several_older_ones(self):
        store = WeightedStore(10)
        for key in "abc":
            store.put(key, 0, 3)
        store.put("d", 0, 8)
        self.assertEqual(store.keys_oldest_first(), ["d"])
        self.assertEqual(store.total_weight, 8)

    def test_entry_equal_to_budget_is_accepted(self):
        store = WeightedStore(10)
        store.put("a", 1, 2)
        store.put("b", 2, 10)
        self.assertEqual(store.keys_oldest_first(), ["b"])

    def test_entry_over_budget_is_refused(self):
        store = WeightedStore(10)
        with self.assertRaises(ValueError) as caught:
            store.put("a", 1, 11)
        self.assertEqual(str(caught.exception), "weight exceeds the cache budget")
        self.assertNotIn("a", store)

    def test_refused_overwrite_keeps_previous_entry(self):
        store = WeightedStore(10)
        store.put("a", "old", 3)
        with self.assertRaises(ValueError):
            store.put("a", "new", 99)
        self.assertEqual(store.get("a"), "old")
        self.assertEqual(store.total_weight, 3)

    def test_put_rejects_invalid_weight_types(self):
        store = WeightedStore(10)
        with self.assertRaises(TypeError):
            store.put("a", 1, "3")
        with self.assertRaises(ValueError):
            store.put("a", 1, 0)

    def test_get_protects_entry_from_eviction(self):
        store = WeightedStore(6)
        store.put("a", 1, 3)
        store.put("b", 2, 3)
        store.get("a")
        store.put("c", 3, 3)
        self.assertEqual(store.keys_oldest_first(), ["a", "c"])

    def test_overwrite_replaces_weight_not_adds(self):
        store = WeightedStore(10)
        store.put("a", 1, 6)
        store.put("a", 2, 2)
        self.assertEqual(store.total_weight, 2)
        self.assertEqual(len(store), 1)

    def test_overwrite_refreshes_entry_recency(self):
        store = WeightedStore(10)
        store.put("a", 1, 2)
        store.put("b", 2, 2)
        store.put("a", 3, 2)
        self.assertEqual(store.keys_oldest_first(), ["b", "a"])

    def test_overwrite_never_evicts_itself(self):
        store = WeightedStore(10)
        store.put("a", 1, 4)
        store.put("b", 2, 4)
        store.put("b", 3, 9)
        self.assertEqual(store.keys_oldest_first(), ["b"])
        self.assertEqual(store.get("b"), 3)

    def test_eviction_callback_sees_each_victim_in_order(self):
        seen = []
        store = WeightedStore(5, on_evict=lambda k, v: seen.append((k, v)))
        store.put("a", "A", 2)
        store.put("b", "B", 2)
        store.put("c", "C", 5)
        self.assertEqual(seen, [("a", "A"), ("b", "B")])

    def test_overwrite_does_not_notify_callback(self):
        seen = []
        store = WeightedStore(5, on_evict=lambda k, v: seen.append(k))
        store.put("a", 1, 2)
        store.put("a", 2, 3)
        self.assertEqual(seen, [])

    def test_delete_releases_entry_weight(self):
        store = WeightedStore(10)
        store.put("a", 1, 4)
        self.assertIs(store.delete("a"), True)
        self.assertEqual(store.total_weight, 0)

    def test_delete_missing_key_reports_false(self):
        self.assertIs(WeightedStore(10).delete("a"), False)

    def test_shrinking_budget_evicts_until_it_fits(self):
        seen = []
        store = WeightedStore(10, on_evict=lambda k, v: seen.append(k))
        for key in "abcd":
            store.put(key, 0, 2)
        store.resize(4)
        self.assertEqual(store.keys_oldest_first(), ["c", "d"])
        self.assertEqual(seen, ["a", "b"])
        self.assertEqual(store.max_weight, 4)

    def test_resize_rejects_invalid_budget(self):
        store = WeightedStore(10)
        with self.assertRaises(ValueError):
            store.resize(0)
        self.assertEqual(store.max_weight, 10)

    def test_heaviest_entry_is_evicted_on_request(self):
        store = WeightedStore(20)
        store.put("a", "A", 3)
        store.put("b", "B", 9)
        store.put("c", "C", 5)
        self.assertEqual(store.evict_heaviest(), ("b", "B"))
        self.assertEqual(store.total_weight, 8)

    def test_heaviest_tie_picks_least_recent(self):
        store = WeightedStore(20)
        store.put("a", "A", 4)
        store.put("b", "B", 4)
        self.assertEqual(store.evict_heaviest(), ("a", "A"))

    def test_evict_heaviest_on_empty_store_raises(self):
        with self.assertRaises(KeyError) as caught:
            WeightedStore(3).evict_heaviest()
        self.assertEqual(caught.exception.args[0], "store is empty")



if __name__ == "__main__":
    unittest.main()
