# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
import unittest

from ttlcache.partitions import (
    PartitionedCaches,
    partition_check_name,
    partition_join,
    partition_split,
)


class PartitionsTests(unittest.TestCase):
    def setUp(self):
        self.caches = PartitionedCaches(2)

    def test_name_check_returns_valid_name(self):
        self.assertEqual(partition_check_name("users"), "users")

    def test_empty_name_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            partition_check_name("")
        self.assertEqual(str(caught.exception), "namespace must be a non-empty string")

    def test_non_string_name_is_rejected(self):
        with self.assertRaises(ValueError):
            partition_check_name(7)

    def test_join_places_separator_between_parts(self):
        self.assertEqual(partition_join("users", "42"), "users:42")

    def test_join_escapes_separator_in_namespace(self):
        self.assertEqual(partition_join("a:b", "k"), "a\\:b:k")

    def test_join_escapes_backslash_in_namespace(self):
        self.assertEqual(partition_join("a\\b", "k"), "a\\\\b:k")

    def test_join_leaves_key_unescaped(self):
        self.assertEqual(partition_join("ns", "x:y\\z"), "ns:x:y\\z")

    def test_split_recovers_simple_parts(self):
        self.assertEqual(partition_split("users:42"), ("users", "42"))

    def test_split_recovers_escaped_namespace(self):
        self.assertEqual(partition_split("a\\:b:k"), ("a:b", "k"))

    def test_split_keeps_later_separators_in_key(self):
        self.assertEqual(partition_split("ns:x:y"), ("ns", "x:y"))

    def test_split_allows_empty_key(self):
        self.assertEqual(partition_split("ns:"), ("ns", ""))

    def test_split_round_trips_awkward_names(self):
        for name in ("plain", "a:b", "x\\", "\\:\\", "::"):
            self.assertEqual(partition_split(partition_join(name, "k:1")), (name, "k:1"))

    def test_split_without_separator_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            partition_split("nosep")
        self.assertEqual(str(caught.exception), "missing separator in 'nosep'")

    def test_split_with_dangling_escape_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            partition_split("abc\\")
        self.assertEqual(str(caught.exception), "dangling escape in 'abc\\\\'")

    def test_namespaces_do_not_share_keys(self):
        self.caches.put("a", "k", 1)
        self.caches.put("b", "k", 2)
        self.assertEqual(self.caches.get("a", "k"), 1)
        self.assertEqual(self.caches.get("b", "k"), 2)

    def test_unknown_namespace_lookup_returns_default(self):
        self.assertEqual(self.caches.get("zz", "k", "dflt"), "dflt")
        self.assertIsNone(self.caches.get("zz", "k"))

    def test_unknown_key_lookup_returns_default(self):
        self.caches.put("a", "k", 1)
        self.assertEqual(self.caches.get("a", "other", 5), 5)

    def test_put_rejects_empty_namespace(self):
        with self.assertRaises(ValueError):
            self.caches.put("", "k", 1)

    def test_default_capacity_bounds_each_namespace(self):
        for key in "xyz":
            self.caches.put("a", key, 0)
        self.assertEqual(self.caches.keys("a"), ["y", "z"])

    def test_capacity_pressure_in_one_namespace_spares_others(self):
        self.caches.put("b", "keep", 1)
        for key in "xyz":
            self.caches.put("a", key, 0)
        self.assertEqual(self.caches.get("b", "keep"), 1)

    def test_read_protects_entry_from_eviction(self):
        self.caches.put("a", "x", 1)
        self.caches.put("a", "y", 2)
        self.caches.get("a", "x")
        self.caches.put("a", "z", 3)
        self.assertEqual(self.caches.keys("a"), ["x", "z"])

    def test_overwrite_moves_key_to_newest_slot(self):
        self.caches.put("a", "x", 1)
        self.caches.put("a", "y", 2)
        self.caches.put("a", "x", 3)
        self.assertEqual(self.caches.keys("a"), ["y", "x"])
        self.assertEqual(self.caches.get("a", "x"), 3)

    def test_capacity_of_falls_back_to_default(self):
        self.assertEqual(self.caches.capacity_of("anything"), 2)

    def test_per_namespace_capacity_overrides_default(self):
        self.caches.set_capacity("big", 4)
        self.assertEqual(self.caches.capacity_of("big"), 4)
        for key in "abcde":
            self.caches.put("big", key, 0)
        self.assertEqual(self.caches.size("big"), 4)

    def test_shrinking_capacity_trims_existing_entries(self):
        self.caches.set_capacity("a", 3)
        for key in "xyz":
            self.caches.put("a", key, 0)
        self.caches.set_capacity("a", 1)
        self.assertEqual(self.caches.keys("a"), ["z"])

    def test_capacity_for_empty_namespace_name_is_rejected(self):
        with self.assertRaises(ValueError):
            self.caches.set_capacity("", 3)

    def test_zero_capacity_override_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            self.caches.set_capacity("a", 0)
        self.assertEqual(str(caught.exception), "capacity must be at least 1")

    def test_zero_default_capacity_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            PartitionedCaches(0)
        self.assertEqual(str(caught.exception), "capacity must be at least 1")

    def test_delete_reports_whether_key_existed(self):
        self.caches.put("a", "x", None)
        self.assertIs(self.caches.delete("a", "x"), True)
        self.assertIs(self.caches.delete("a", "x"), False)
        self.assertIs(self.caches.delete("nowhere", "x"), False)

    def test_size_counts_one_namespace(self):
        self.caches.put("a", "x", 1)
        self.caches.put("b", "y", 1)
        self.assertEqual(self.caches.size("a"), 1)
        self.assertEqual(self.caches.size("unused"), 0)

    def test_total_size_sums_all_namespaces(self):
        self.caches.put("a", "x", 1)
        self.caches.put("a", "y", 1)
        self.caches.put("b", "x", 1)
        self.assertEqual(self.caches.total_size(), 3)

    def test_namespaces_listing_is_sorted_and_nonempty_only(self):
        self.caches.put("zeta", "x", 1)
        self.caches.put("alpha", "x", 1)
        self.caches.put("mid", "x", 1)
        self.caches.delete("mid", "x")
        self.assertEqual(self.caches.namespaces(), ["alpha", "zeta"])

    def test_clear_namespace_returns_removed_count(self):
        self.caches.put("a", "x", 1)
        self.caches.put("a", "y", 1)
        self.caches.put("b", "x", 1)
        self.assertEqual(self.caches.clear_namespace("a"), 2)
        self.assertEqual(self.caches.total_size(), 1)

    def test_clear_unknown_namespace_returns_zero(self):
        self.assertEqual(self.caches.clear_namespace("ghost"), 0)

    def test_keys_of_unknown_namespace_are_empty(self):
        self.assertEqual(self.caches.keys("ghost"), [])


if __name__ == "__main__":
    unittest.main()
