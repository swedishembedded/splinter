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


class C1(unittest.TestCase):


    def test_array_copy_is_independent(self):
        original = [[1], [2]]
        copy = ini_copy_value(original)
        copy[0].append(9)
        self.assertEqual(original, [[1], [2]])


class C2(unittest.TestCase):

    def test_append_mode_keeps_repeated_items(self):
        self.assertEqual(ini_combine_arrays([1, 2], [2, 3], "append"), [1, 2, 2, 3])

    def test_union_skips_present_items(self):
        self.assertEqual(ini_combine_arrays([1, 2], [2, 3], "union"), [1, 2, 3])

    def test_union_mode_drops_repeats_inside_new_array(self):
        self.assertEqual(ini_combine_arrays([1], [2, 2], "union"), [1, 2])


class C3(unittest.TestCase):


    def test_nested_tables_combine_key_by_key(self):
        merged = ini_merge_documents({"s": {"a": 1, "b": 2}}, {"s": {"b": 3, "c": 4}})
        self.assertEqual(merged, {"s": {"a": 1, "b": 3, "c": 4}})


    def test_arrays_can_be_appended(self):
        self.assertEqual(ini_merge_documents({"a": [1]}, {"a": [2]}, arrays="append"), {"a": [1, 2]})

    def test_arrays_can_be_united(self):
        self.assertEqual(ini_merge_documents({"a": [1, 2]}, {"a": [2, 3]}, arrays="union"), {"a": [1, 2, 3]})


    def test_scalar_replaces_a_table_when_lenient(self):
        self.assertEqual(ini_merge_documents({"a": {"b": 1}}, {"a": 5}), {"a": 5})

    def test_table_replaces_a_scalar_when_lenient(self):
        self.assertEqual(ini_merge_documents({"a": 5}, {"a": {"b": 1}}), {"a": {"b": 1}})


    def test_replaced_value_is_copied_from_override(self):
        override = {"a": [1]}
        merged = ini_merge_documents({"a": 0}, override)
        merged["a"].append(2)
        self.assertEqual(override, {"a": [1]})


class C4(unittest.TestCase):

    def test_strict_refuses_kind_change(self):
        self.assertEqual(refusal(ini_merge_documents, {"a": 1}, {"a": "x"}, strict=True),
                         "a: cannot replace integer with string")


    def test_strict_names_the_nested_path(self):
        self.assertEqual(refusal(ini_merge_documents, {"s": {"a b": {}}}, {"s": {"a b": 1}}, strict=True),
                         "s.\"a b\": cannot replace table with integer")

    def test_strict_refuses_array_to_scalar(self):
        self.assertEqual(refusal(ini_merge_documents, {"a": [1]}, {"a": 1}, strict=True),
                         "a: cannot replace array with integer")


class C5(unittest.TestCase):

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


class C6(unittest.TestCase):

    def test_shared_scalar_is_listed(self):
        self.assertEqual(ini_shared_leaf_paths({"a": 1}, {"a": 1}), ["a"])

    def test_nested_paths_are_dotted(self):
        self.assertEqual(ini_shared_leaf_paths({"s": {"a": 1, "b": 2}}, {"s": {"b": 3, "c": 4}}), ["s.b"])


    def test_table_against_scalar_counts_as_a_leaf(self):
        self.assertEqual(ini_shared_leaf_paths({"s": {"a": 1}}, {"s": 2}), ["s"])


    def test_odd_names_are_quoted(self):
        self.assertEqual(ini_shared_leaf_paths({"a b": {"c": 1}}, {"a b": {"c": 2}}), ['"a b".c'])


if __name__ == "__main__":
    unittest.main()
