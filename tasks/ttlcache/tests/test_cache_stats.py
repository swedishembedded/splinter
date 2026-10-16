# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
import unittest
from fractions import Fraction

from ttlcache.cache_stats import (
    COUNTER_FIELDS,
    CacheCounters,
    counters_delta,
    counters_lookup,
    counters_report,
)


class T(unittest.TestCase):
    def setUp(self):
        self.counters = CacheCounters()

    def test_fresh_counters_start_at_zero(self):
        self.assertEqual(self.counters.as_dict(), dict.fromkeys(COUNTER_FIELDS, 0))

    def test_field_names_are_stable(self):
        self.assertEqual(COUNTER_FIELDS, ("hits", "misses", "evictions", "expirations", "writes"))

    def test_hit_recording_accepts_a_count(self):
        self.counters.record_hit(4)
        self.assertEqual(self.counters.hits, 4)

    def test_miss_recording_tracks_its_own_field(self):
        self.counters.record_miss(3)
        self.assertEqual(self.counters.as_dict()["misses"], 3)
        self.assertEqual(self.counters.hits, 0)

    def test_eviction_recording_tracks_its_own_field(self):
        self.counters.record_eviction(2)
        self.counters.record_eviction()
        self.assertEqual(self.counters.evictions, 3)

    def test_expiration_recording_tracks_its_own_field(self):
        self.counters.record_expiration(5)
        self.assertEqual(self.counters.expirations, 5)
        self.assertEqual(self.counters.evictions, 0)

    def test_expiration_recording_adds_one_by_default(self):
        self.counters.record_expiration()
        self.assertEqual(self.counters.expirations, 1)

    def test_write_recording_tracks_its_own_field(self):
        self.counters.record_write()
        self.counters.record_write(6)
        self.assertEqual(self.counters.writes, 7)

    def test_lookups_sum_hits_and_misses(self):
        self.counters.record_hit(3)
        self.counters.record_miss(4)
        self.counters.record_write(10)
        self.assertEqual(self.counters.lookups(), 7)

    def test_ratio_is_zero_before_any_lookup(self):
        self.assertEqual(self.counters.hit_ratio(), Fraction(0))

    def test_ratio_is_exact_fraction(self):
        self.counters.record_hit(1)
        self.counters.record_miss(2)
        self.assertEqual(self.counters.hit_ratio(), Fraction(1, 3))

    def test_percent_is_zero_before_any_lookup(self):
        self.assertEqual(self.counters.hit_percent(), 0)

    def test_percent_rounds_down_to_integer(self):
        self.counters.record_hit(2)
        self.counters.record_miss(1)
        self.assertEqual(self.counters.hit_percent(), 66)

    def test_percent_reaches_one_hundred(self):
        self.counters.record_hit(7)
        self.assertEqual(self.counters.hit_percent(), 100)

    def test_percent_ignores_non_lookup_events(self):
        self.counters.record_hit(1)
        self.counters.record_miss(1)
        self.counters.record_eviction(50)
        self.assertEqual(self.counters.hit_percent(), 50)

    def test_dict_export_is_a_copy(self):
        exported = self.counters.as_dict()
        exported["hits"] = 99
        self.assertEqual(self.counters.hits, 0)

    def test_absorb_adds_every_field(self):
        other = CacheCounters()
        other.record_hit(1)
        other.record_miss(2)
        other.record_eviction(3)
        other.record_expiration(4)
        other.record_write(5)
        self.counters.record_hit(10)
        self.counters.absorb(other)
        self.assertEqual(
            self.counters.as_dict(),
            {"hits": 11, "misses": 2, "evictions": 3, "expirations": 4, "writes": 5},
        )

    def test_reset_zeroes_every_field(self):
        self.counters.record_hit(1)
        self.counters.record_write(2)
        self.counters.record_expiration(3)
        self.counters.reset()
        self.assertEqual(self.counters.as_dict(), dict.fromkeys(COUNTER_FIELDS, 0))

    def test_delta_subtracts_two_snapshots(self):
        self.counters.record_hit(2)
        before = self.counters.as_dict()
        self.counters.record_hit(3)
        self.counters.record_miss(1)
        self.assertEqual(
            counters_delta(before, self.counters.as_dict()),
            {"hits": 3, "misses": 1, "evictions": 0, "expirations": 0, "writes": 0},
        )

    def test_delta_rejects_backwards_counter(self):
        before = self.counters.as_dict()
        after = dict(before, evictions=-1)
        with self.assertRaises(ValueError) as caught:
            counters_delta(before, after)
        self.assertEqual(str(caught.exception), "counter evictions went backwards")

    def test_report_lists_every_counter(self):
        self.counters.record_hit(3)
        self.counters.record_miss(1)
        self.counters.record_eviction(2)
        self.counters.record_expiration(4)
        self.counters.record_write(5)
        self.assertEqual(
            counters_report(self.counters),
            "hits=3 misses=1 evictions=2 expirations=4 writes=5 ratio=75%",
        )

    def test_lookup_helper_counts_a_hit(self):
        self.assertEqual(counters_lookup(self.counters, {"k": 5}, "k"), 5)
        self.assertEqual((self.counters.hits, self.counters.misses), (1, 0))

    def test_lookup_helper_counts_a_miss(self):
        self.assertEqual(counters_lookup(self.counters, {}, "k", "dflt"), "dflt")
        self.assertEqual((self.counters.hits, self.counters.misses), (0, 1))

    def test_lookup_helper_default_is_none(self):
        self.assertIsNone(counters_lookup(self.counters, {}, "k"))

    def test_lookup_helper_counts_stored_none_as_hit(self):
        self.assertIsNone(counters_lookup(self.counters, {"k": None}, "k", "dflt"))
        self.assertEqual(self.counters.hits, 1)


if __name__ == "__main__":
    unittest.main()
