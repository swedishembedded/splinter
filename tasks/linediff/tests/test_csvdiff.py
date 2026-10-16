# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

import unittest
from dataclasses import FrozenInstanceError

from linediff.csvdiff import (
    LinediffKeyError,
    LinediffTableDiff,
    linediff_cell_changes,
    linediff_diff_tables,
    linediff_index_records,
    linediff_render_table_diff,
    linediff_table_diff_counts,
    linediff_table_diff_is_empty,
)

OLD = [["id", "name", "city"], ["1", "ann", "oslo"], ["2", "bob", "rome"], ["3", "cy", "kyiv"]]


def key_error(call, *args):
    with unittest.TestCase().assertRaises(LinediffKeyError) as caught:
        call(*args)
    return str(caught.exception)


class IndexRecordsTests(unittest.TestCase):
    def test_records_are_keyed_by_column_values(self):
        records = [{"id": "1", "n": "a"}, {"id": "2", "n": "b"}]
        self.assertEqual(
            linediff_index_records(records, ["id"]),
            {("1",): records[0], ("2",): records[1]},
        )

    def test_compound_key_uses_column_order(self):
        records = [{"a": "x", "b": "y"}]
        self.assertEqual(list(linediff_index_records(records, ["b", "a"])), [("y", "x")])

    def test_repeated_key_is_rejected(self):
        records = [{"a": "x", "b": "1"}, {"a": "x", "b": "2"}]
        self.assertEqual(key_error(linediff_index_records, records, ["a"]), "duplicate key x")

    def test_repeated_compound_key_lists_all_parts(self):
        records = [{"a": "x", "b": "1"}, {"a": "x", "b": "1"}]
        self.assertEqual(
            key_error(linediff_index_records, records, ["a", "b"]), "duplicate key x/1"
        )

    def test_shared_first_part_stays_unique(self):
        records = [{"a": "x", "b": "1"}, {"a": "x", "b": "2"}]
        self.assertEqual(len(linediff_index_records(records, ["a", "b"])), 2)


class CellChangesTests(unittest.TestCase):
    def test_differing_cells_are_reported_with_both_values(self):
        old = {"a": "1", "b": "2"}
        new = {"a": "1", "b": "3"}
        self.assertEqual(linediff_cell_changes(old, new, ["a", "b"]), {"b": ("2", "3")})

    def test_equal_records_have_no_changes(self):
        self.assertEqual(linediff_cell_changes({"a": "1"}, {"a": "1"}, ["a"]), {})

    def test_unlisted_columns_are_ignored(self):
        self.assertEqual(linediff_cell_changes({"a": "1"}, {"a": "2"}, []), {})

    def test_changes_follow_column_order(self):
        old = {"a": "1", "b": "1"}
        new = {"a": "2", "b": "2"}
        self.assertEqual(list(linediff_cell_changes(old, new, ["b", "a"])), ["b", "a"])


class DiffTablesTests(unittest.TestCase):
    def test_equal_tables_have_empty_diff(self):
        diff = linediff_diff_tables(OLD, [list(row) for row in OLD], ["id"])
        self.assertEqual(diff, LinediffTableDiff((), (), (), (), ()))

    def test_new_key_is_reported_as_added(self):
        new = OLD + [["4", "di", "lima"]]
        diff = linediff_diff_tables(OLD, new, ["id"])
        self.assertEqual(
            diff.added, ((("4",), {"id": "4", "name": "di", "city": "lima"}),)
        )

    def test_vanished_key_is_reported_as_removed(self):
        diff = linediff_diff_tables(OLD, OLD[:2] + OLD[3:], ["id"])
        self.assertEqual(
            diff.removed, ((("2",), {"id": "2", "name": "bob", "city": "rome"}),)
        )

    def test_changed_cells_are_grouped_by_key(self):
        new = [OLD[0], ["1", "anne", "oslo"], OLD[2], ["3", "cy", "bern"]]
        diff = linediff_diff_tables(OLD, new, ["id"])
        self.assertEqual(
            diff.changed,
            ((("1",), {"name": ("ann", "anne")}), (("3",), {"city": ("kyiv", "bern")})),
        )

    def test_reordered_rows_are_not_changes(self):
        new = [OLD[0], OLD[3], OLD[1], OLD[2]]
        self.assertTrue(linediff_table_diff_is_empty(linediff_diff_tables(OLD, new, ["id"])))

    def test_added_rows_follow_new_order(self):
        new = OLD + [["9", "x", "y"], ["5", "z", "w"]]
        diff = linediff_diff_tables(OLD, new, ["id"])
        self.assertEqual([key for key, _ in diff.added], [("9",), ("5",)])

    def test_removed_rows_follow_old_order(self):
        diff = linediff_diff_tables(OLD, [OLD[0], OLD[2]], ["id"])
        self.assertEqual([key for key, _ in diff.removed], [("1",), ("3",)])

    def test_new_column_is_listed_and_not_compared(self):
        new = [["id", "name", "city", "zip"], ["1", "ann", "oslo", "0150"],
               ["2", "bob", "rome", "00100"], ["3", "cy", "kyiv", "01001"]]
        diff = linediff_diff_tables(OLD, new, ["id"])
        self.assertEqual(diff.added_columns, ("zip",))
        self.assertEqual(diff.changed, ())

    def test_dropped_column_is_listed(self):
        new = [["id", "name"], ["1", "ann"], ["2", "bob"], ["3", "cy"]]
        diff = linediff_diff_tables(OLD, new, ["id"])
        self.assertEqual(diff.removed_columns, ("city",))

    def test_compound_key_matches_rows_by_both_columns(self):
        old = [["a", "b", "v"], ["1", "x", "p"], ["1", "y", "q"]]
        new = [["a", "b", "v"], ["1", "x", "p"], ["1", "y", "r"]]
        diff = linediff_diff_tables(old, new, ["a", "b"])
        self.assertEqual(diff.changed, ((("1", "y"), {"v": ("q", "r")}),))

    def test_short_rows_are_padded_before_comparison(self):
        old = [["id", "n"], ["1"]]
        new = [["id", "n"], ["1", "x"]]
        self.assertEqual(
            linediff_diff_tables(old, new, ["id"]).changed, ((("1",), {"n": ("", "x")}),)
        )

    def test_missing_key_list_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            linediff_diff_tables(OLD, OLD, [])
        self.assertEqual(str(caught.exception), "at least one key column is required")

    def test_key_missing_in_old_header_is_named(self):
        self.assertEqual(
            key_error(linediff_diff_tables, [["a"]], [["a", "id"]], ["id"]),
            "key column 'id' is missing from the old header",
        )

    def test_key_missing_in_new_header_is_named(self):
        self.assertEqual(
            key_error(linediff_diff_tables, [["a", "id"]], [["a"]], ["id"]),
            "key column 'id' is missing from the new header",
        )

    def test_duplicate_keys_in_new_table_are_rejected(self):
        new = OLD + [["1", "dup", "dup"]]
        self.assertEqual(key_error(linediff_diff_tables, OLD, new, ["id"]), "duplicate key 1")


class TableDiffValueTests(unittest.TestCase):
    def test_table_diff_is_an_immutable_value(self):
        diff = LinediffTableDiff((), (), (), (), ())
        with self.assertRaises(FrozenInstanceError):
            diff.added = (1,)


class DiffSummaryTests(unittest.TestCase):
    def test_empty_diff_is_detected(self):
        self.assertTrue(linediff_table_diff_is_empty(LinediffTableDiff((), (), (), (), ())))

    def test_any_single_difference_makes_diff_nonempty(self):
        for field in range(5):
            parts = [(), (), (), (), ()]
            parts[field] = ("x",)
            self.assertFalse(linediff_table_diff_is_empty(LinediffTableDiff(*parts)))

    def test_counts_give_rows_and_cells(self):
        diff = LinediffTableDiff(
            (), (), ((("1",), {}),), ((("2",), {}), (("3",), {})),
            ((("4",), {"a": ("x", "y"), "b": ("p", "q")}),),
        )
        self.assertEqual(linediff_table_diff_counts(diff), (1, 2, 2))

    def test_report_lines_cover_every_difference(self):
        diff = LinediffTableDiff(
            ("zip",), ("fax",), ((("5", "b"), {}),), ((("2", "a"), {}),),
            ((("1", "c"), {"name": ("ann", "anne")}),),
        )
        self.assertEqual(
            linediff_render_table_diff(diff),
            [
                "column added: zip",
                "column removed: fax",
                "row removed: 2/a",
                "row added: 5/b",
                "cell changed: 1/c name: ann -> anne",
            ],
        )

    def test_report_of_equal_tables_is_empty(self):
        self.assertEqual(linediff_render_table_diff(LinediffTableDiff((), (), (), (), ())), [])
