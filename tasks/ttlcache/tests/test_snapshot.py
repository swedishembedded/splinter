# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
import json
import unittest

from ttlcache.snapshot import (
    SNAPSHOT_VERSION,
    snapshot_compare,
    snapshot_decode,
    snapshot_dump,
    snapshot_encode,
    snapshot_keys,
    snapshot_restore,
)


class RecordingStore:
    def __init__(self, live=()):
        self.live = list(live)
        self.puts = []

    def live_entries(self):
        return self.live

    def put(self, key, value, ttl=None):
        self.puts.append((key, value, ttl))


def snap(*triples):
    return snapshot_encode(list(triples))


class T(unittest.TestCase):
    def test_version_constant_is_one(self):
        self.assertEqual(SNAPSHOT_VERSION, 1)

    def test_encode_empty_input_gives_empty_snapshot(self):
        self.assertEqual(snapshot_encode([]), {"version": 1, "entries": []})

    def test_encode_builds_plain_entry_dicts(self):
        self.assertEqual(
            snapshot_encode([("a", [1, 2], 30)]),
            {"version": 1, "entries": [{"key": "a", "value": [1, 2], "ttl_left": 30}]},
        )

    def test_encode_preserves_input_order(self):
        encoded = snapshot_encode([("b", 1, None), ("a", 2, None)])
        self.assertEqual([e["key"] for e in encoded["entries"]], ["b", "a"])

    def test_encode_rejects_zero_remaining_lifetime(self):
        with self.assertRaises(ValueError) as caught:
            snapshot_encode([("a", 1, 0)])
        self.assertEqual(str(caught.exception), "ttl_left must be positive for key 'a'")

    def test_encode_rejects_negative_remaining_lifetime(self):
        with self.assertRaises(ValueError):
            snapshot_encode([("a", 1, -2)])

    def test_encode_accepts_smallest_positive_lifetime(self):
        self.assertEqual(snapshot_encode([("a", 1, 1)])["entries"][0]["ttl_left"], 1)

    def test_snapshot_survives_json_round_trip(self):
        original = snap(("a", {"n": [1, 2]}, 5), ("b", "text", None))
        revived = json.loads(json.dumps(original))
        self.assertEqual(snapshot_decode(revived), snapshot_decode(original))

    def test_decode_returns_triples_in_order(self):
        data = snap(("a", 1, 10), ("b", 2, None))
        self.assertEqual(snapshot_decode(data), [("a", 1, 10), ("b", 2, None)])

    def test_decode_rejects_non_mapping(self):
        with self.assertRaises(ValueError) as caught:
            snapshot_decode([])
        self.assertEqual(str(caught.exception), "snapshot must be a mapping")

    def test_decode_rejects_missing_version(self):
        with self.assertRaises(ValueError) as caught:
            snapshot_decode({"entries": []})
        self.assertEqual(str(caught.exception), "unsupported snapshot version")

    def test_decode_rejects_non_list_entries(self):
        with self.assertRaises(ValueError) as caught:
            snapshot_decode({"version": 1, "entries": ()})
        self.assertEqual(str(caught.exception), "snapshot entries must be a list")

    def test_decode_rejects_missing_entries_field(self):
        with self.assertRaises(ValueError):
            snapshot_decode({"version": 1})

    def test_decode_rejects_scalar_entry_item(self):
        with self.assertRaises(ValueError) as caught:
            snapshot_decode({"version": 1, "entries": ["x"]})
        self.assertEqual(str(caught.exception), "snapshot entry 0 is malformed")

    def test_decode_rejects_entry_missing_a_field(self):
        for missing in ("key", "value", "ttl_left"):
            entry = {"key": "a", "value": 1, "ttl_left": None}
            del entry[missing]
            with self.assertRaises(ValueError):
                snapshot_decode({"version": 1, "entries": [entry]})

    def test_decode_tolerates_extra_entry_fields(self):
        entry = {"key": "a", "value": 1, "ttl_left": None, "note": "x"}
        self.assertEqual(snapshot_decode({"version": 1, "entries": [entry]}), [("a", 1, None)])

    def test_decode_reports_position_of_bad_entry(self):
        good = {"key": "a", "value": 1, "ttl_left": None}
        with self.assertRaises(ValueError) as caught:
            snapshot_decode({"version": 1, "entries": [good, good, {}]})
        self.assertEqual(str(caught.exception), "snapshot entry 2 is malformed")

    def test_decode_rejects_zero_lifetime_entry(self):
        data = {"version": 1, "entries": [{"key": "a", "value": 1, "ttl_left": 0}]}
        with self.assertRaises(ValueError) as caught:
            snapshot_decode(data)
        self.assertEqual(str(caught.exception), "snapshot entry 0 has non-positive ttl")

    def test_decode_accepts_smallest_positive_lifetime(self):
        data = {"version": 1, "entries": [{"key": "a", "value": 1, "ttl_left": 1}]}
        self.assertEqual(snapshot_decode(data), [("a", 1, 1)])

    def test_dump_reads_live_entries_of_store(self):
        store = RecordingStore([("a", 1, 9), ("b", 2, None)])
        self.assertEqual(snapshot_dump(store), snap(("a", 1, 9), ("b", 2, None)))

    def test_restore_puts_each_entry_with_remaining_ttl(self):
        store = RecordingStore()
        count = snapshot_restore(store, snap(("a", 1, 9), ("b", 2, None)))
        self.assertEqual(count, 2)
        self.assertEqual(store.puts, [("a", 1, 9), ("b", 2, None)])

    def test_restore_of_invalid_snapshot_touches_nothing(self):
        store = RecordingStore()
        bad = {"version": 1, "entries": [{"key": "a", "value": 1, "ttl_left": None}, 5]}
        with self.assertRaises(ValueError):
            snapshot_restore(store, bad)
        self.assertEqual(store.puts, [])

    def test_keys_lists_snapshot_keys_in_order(self):
        self.assertEqual(snapshot_keys(snap(("b", 1, None), ("a", 2, 3))), ["b", "a"])

    def test_compare_finds_added_keys(self):
        report = snapshot_compare(snap(("a", 1, None)), snap(("a", 1, None), ("c", 3, None), ("b", 2, None)))
        self.assertEqual(report, {"added": ["b", "c"], "removed": [], "changed": []})

    def test_compare_finds_removed_keys(self):
        report = snapshot_compare(snap(("z", 1, None), ("a", 1, None)), snap())
        self.assertEqual(report, {"added": [], "removed": ["a", "z"], "changed": []})

    def test_compare_finds_changed_values(self):
        report = snapshot_compare(snap(("a", 1, None), ("b", 2, None)), snap(("a", 1, None), ("b", 9, None)))
        self.assertEqual(report, {"added": [], "removed": [], "changed": ["b"]})

    def test_compare_ignores_lifetime_differences(self):
        report = snapshot_compare(snap(("a", 1, 5)), snap(("a", 1, 50)))
        self.assertEqual(report, {"added": [], "removed": [], "changed": []})

    def test_compare_validates_both_snapshots(self):
        with self.assertRaises(ValueError):
            snapshot_compare({}, snap())
        with self.assertRaises(ValueError):
            snapshot_compare(snap(), {})


if __name__ == "__main__":
    unittest.main()
