# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
import unittest

from iniparse.diagnostics import IniMergeError
from iniparse.overlay import (
    ini_combine_arrays, ini_copy_value, ini_merge_documents, ini_merge_many, ini_shared_leaf_paths,
)


def refusal(call, *args, **kwargs):
    try:
        call(*args, **kwargs)
    except IniMergeError as error:
        return str(error)
    raise AssertionError("no error raised")


class CopyValueTests(unittest.TestCase):
    def test_scalar_is_returned_as_is(self):
        self.assertEqual(ini_copy_value(5), 5)

    def test_table_copy_is_independent(self):
        original = {"a": {"b": [1]}}
        copy = ini_copy_value(original)
        copy["a"]["b"].append(2)
        copy["a"]["c"] = 1
        self.assertEqual(original, {"a": {"b": [1]}})

    def test_array_copy_is_independent(self):
        original = [[1], [2]]
        copy = ini_copy_value(original)
        copy[0].append(9)
        self.assertEqual(original, [[1], [2]])

    def test_copy_preserves_key_order(self):
        self.assertEqual(list(ini_copy_value({"z": 1, "a": 2})), ["z", "a"])


class CombineArraysTests(unittest.TestCase):
    def test_replace_takes_the_new_array(self):
        self.assertEqual(ini_combine_arrays([1, 2], [3], "replace"), [3])

    def test_append_mode_keeps_repeated_items(self):
        self.assertEqual(ini_combine_arrays([1, 2], [2, 3], "append"), [1, 2, 2, 3])

    def test_union_skips_present_items(self):
        self.assertEqual(ini_combine_arrays([1, 2], [2, 3], "union"), [1, 2, 3])

    def test_union_mode_drops_repeats_inside_new_array(self):
        self.assertEqual(ini_combine_arrays([1], [2, 2], "union"), [1, 2])

    def test_append_does_not_touch_the_old_list(self):
        old = [1]
        ini_combine_arrays(old, [2], "append")
        self.assertEqual(old, [1])

    def test_replace_copies_nested_arrays(self):
        new = [[1]]
        result = ini_combine_arrays([], new, "replace")
        result[0].append(2)
        self.assertEqual(new, [[1]])


class MergeDocumentsTests(unittest.TestCase):
    def test_override_wins_for_scalars(self):
        self.assertEqual(ini_merge_documents({"a": 1}, {"a": 2}), {"a": 2})

    def test_new_keys_are_added(self):
        self.assertEqual(ini_merge_documents({"a": 1}, {"b": 2}), {"a": 1, "b": 2})

    def test_base_only_keys_survive_the_merge(self):
        self.assertEqual(ini_merge_documents({"a": 1, "b": 2}, {"b": 3}), {"a": 1, "b": 3})

    def test_nested_tables_combine_key_by_key(self):
        merged = ini_merge_documents({"s": {"a": 1, "b": 2}}, {"s": {"b": 3, "c": 4}})
        self.assertEqual(merged, {"s": {"a": 1, "b": 3, "c": 4}})

    def test_three_level_tables_combine_key_by_key(self):
        merged = ini_merge_documents({"a": {"b": {"c": 1, "d": 2}}}, {"a": {"b": {"d": 3}}})
        self.assertEqual(merged, {"a": {"b": {"c": 1, "d": 3}}})

    def test_arrays_are_replaced_by_default(self):
        self.assertEqual(ini_merge_documents({"a": [1, 2]}, {"a": [3]}), {"a": [3]})

    def test_arrays_can_be_appended(self):
        self.assertEqual(ini_merge_documents({"a": [1]}, {"a": [2]}, arrays="append"), {"a": [1, 2]})

    def test_arrays_can_be_united(self):
        self.assertEqual(ini_merge_documents({"a": [1, 2]}, {"a": [2, 3]}, arrays="union"), {"a": [1, 2, 3]})

    def test_array_mode_applies_in_nested_tables(self):
        merged = ini_merge_documents({"s": {"a": [1]}}, {"s": {"a": [2]}}, arrays="append")
        self.assertEqual(merged, {"s": {"a": [1, 2]}})

    def test_scalar_replaces_a_table_when_lenient(self):
        self.assertEqual(ini_merge_documents({"a": {"b": 1}}, {"a": 5}), {"a": 5})

    def test_table_replaces_a_scalar_when_lenient(self):
        self.assertEqual(ini_merge_documents({"a": 5}, {"a": {"b": 1}}), {"a": {"b": 1}})

    def test_inputs_are_not_modified(self):
        base = {"s": {"a": [1]}}
        override = {"s": {"b": [2]}}
        ini_merge_documents(base, override, arrays="append")
        self.assertEqual((base, override), ({"s": {"a": [1]}}, {"s": {"b": [2]}}))

    def test_result_shares_nothing_with_inputs(self):
        base = {"a": {"x": 1}}
        override = {"b": {"y": 2}}
        merged = ini_merge_documents(base, override)
        merged["a"]["x"] = 9
        merged["b"]["y"] = 9
        self.assertEqual((base, override), ({"a": {"x": 1}}, {"b": {"y": 2}}))

    def test_replaced_value_is_copied_from_override(self):
        override = {"a": [1]}
        merged = ini_merge_documents({"a": 0}, override)
        merged["a"].append(2)
        self.assertEqual(override, {"a": [1]})

    def test_key_order_is_base_then_new_keys(self):
        merged = ini_merge_documents({"b": 1, "a": 2}, {"c": 3, "a": 4})
        self.assertEqual(list(merged), ["b", "a", "c"])

    def test_unknown_array_mode_is_refused(self):
        self.assertEqual(refusal(ini_merge_documents, {}, {}, arrays="zip"), "unknown array mode 'zip'")


class StrictMergeTests(unittest.TestCase):
    def test_strict_allows_same_kind_replacement(self):
        self.assertEqual(ini_merge_documents({"a": 1}, {"a": 2}, strict=True), {"a": 2})

    def test_strict_refuses_kind_change(self):
        self.assertEqual(refusal(ini_merge_documents, {"a": 1}, {"a": "x"}, strict=True),
                         "a: cannot replace integer with string")

    def test_strict_refuses_integer_to_float(self):
        self.assertEqual(refusal(ini_merge_documents, {"a": 1}, {"a": 1.5}, strict=True),
                         "a: cannot replace integer with float")

    def test_strict_names_the_nested_path(self):
        self.assertEqual(refusal(ini_merge_documents, {"s": {"a b": {}}}, {"s": {"a b": 1}}, strict=True),
                         "s.\"a b\": cannot replace table with integer")

    def test_strict_refuses_array_to_scalar(self):
        self.assertEqual(refusal(ini_merge_documents, {"a": [1]}, {"a": 1}, strict=True),
                         "a: cannot replace array with integer")

    def test_strict_arrays_still_combine(self):
        self.assertEqual(ini_merge_documents({"a": [1]}, {"a": ["x"]}, strict=True), {"a": ["x"]})


class MergeManyTests(unittest.TestCase):
    def test_no_documents_give_empty(self):
        self.assertEqual(ini_merge_many([]), {})

    def test_last_document_in_the_list_wins(self):
        self.assertEqual(ini_merge_many([{"a": 1}, {"a": 2}, {"a": 3}]), {"a": 3})

    def test_all_keys_are_gathered(self):
        self.assertEqual(ini_merge_many([{"a": 1}, {"b": 2}, {"c": 3}]), {"a": 1, "b": 2, "c": 3})

    def test_default_merge_is_lenient(self):
        self.assertEqual(ini_merge_many([{"a": 1}, {"a": "x"}]), {"a": "x"})

    def test_options_are_passed_down(self):
        self.assertEqual(ini_merge_many([{"a": [1]}, {"a": [2]}, {"a": [3]}], arrays="append"), {"a": [1, 2, 3]})

    def test_strictness_is_passed_down(self):
        self.assertEqual(refusal(ini_merge_many, [{"a": 1}, {"a": "x"}], strict=True),
                         "a: cannot replace integer with string")


class SharedLeafPathTests(unittest.TestCase):
    def test_no_overlap_gives_nothing(self):
        self.assertEqual(ini_shared_leaf_paths({"a": 1}, {"b": 2}), [])

    def test_shared_scalar_is_listed(self):
        self.assertEqual(ini_shared_leaf_paths({"a": 1}, {"a": 1}), ["a"])

    def test_nested_paths_are_dotted(self):
        self.assertEqual(ini_shared_leaf_paths({"s": {"a": 1, "b": 2}}, {"s": {"b": 3, "c": 4}}), ["s.b"])

    def test_tables_on_both_sides_are_descended(self):
        self.assertEqual(ini_shared_leaf_paths({"s": {}}, {"s": {}}), [])

    def test_table_against_scalar_counts_as_a_leaf(self):
        self.assertEqual(ini_shared_leaf_paths({"s": {"a": 1}}, {"s": 2}), ["s"])

    def test_paths_follow_override_order(self):
        self.assertEqual(ini_shared_leaf_paths({"a": 1, "b": 2}, {"b": 0, "a": 0}), ["b", "a"])

    def test_odd_names_are_quoted(self):
        self.assertEqual(ini_shared_leaf_paths({"a b": {"c": 1}}, {"a b": {"c": 2}}), ['"a b".c'])


if __name__ == "__main__":
    unittest.main()
