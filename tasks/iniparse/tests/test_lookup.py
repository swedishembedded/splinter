# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
import unittest

from iniparse.diagnostics import IniLookupError, IniSyntaxError
from iniparse.lookup import (
    INI_NO_DEFAULT, ini_array_index, ini_describe_path, ini_has_path, ini_list_paths, ini_lookup,
    ini_lookup_first, ini_lookup_kind, ini_path_segments,
)

DOC = {
    "name": "demo",
    "server": {"port": 8080, "hosts": ["a", "b", "c"], "tls": {"on": True}, "ratio": 0.5},
    "empty": {},
    "a b": {"c.d": 1},
}


def refusal(call, *args):
    try:
        call(*args)
    except IniLookupError as error:
        return str(error)
    raise AssertionError("no error raised")


class PathSegmentsTests(unittest.TestCase):
    def test_dotted_text_is_split(self):
        self.assertEqual(ini_path_segments("a.b"), ("a", "b"))

    def test_tuple_is_taken_as_is(self):
        self.assertEqual(ini_path_segments(("a.b", "c")), ("a.b", "c"))

    def test_list_becomes_a_tuple(self):
        self.assertEqual(ini_path_segments(["a", 0]), ("a", 0))

    def test_bad_text_raises_a_syntax_error(self):
        with self.assertRaises(IniSyntaxError):
            ini_path_segments("a..b")

    def test_describe_quotes_odd_segments(self):
        self.assertEqual(ini_describe_path(("a b", 3)), '"a b".3')


class ArrayIndexTests(unittest.TestCase):
    def test_integer_is_its_own_index(self):
        self.assertEqual(ini_array_index(2), 2)

    def test_digit_text_is_an_index(self):
        self.assertEqual(ini_array_index("12"), 12)

    def test_negative_text_is_an_index(self):
        self.assertEqual(ini_array_index("-1"), -1)

    def test_word_is_not_an_index(self):
        self.assertIsNone(ini_array_index("two"))

    def test_signed_plus_text_is_not_an_index(self):
        self.assertIsNone(ini_array_index("+1"))

    def test_mixed_text_is_not_an_index(self):
        self.assertIsNone(ini_array_index("1a"))


class LookupTests(unittest.TestCase):
    def test_key_at_the_top_level_is_found(self):
        self.assertEqual(ini_lookup(DOC, "name"), "demo")

    def test_key_inside_a_table_is_found(self):
        self.assertEqual(ini_lookup(DOC, "server.port"), 8080)

    def test_key_three_levels_down_is_found(self):
        self.assertIs(ini_lookup(DOC, "server.tls.on"), True)

    def test_table_value_is_returned_whole(self):
        self.assertEqual(ini_lookup(DOC, "server.tls"), {"on": True})

    def test_empty_path_returns_the_document(self):
        self.assertIs(ini_lookup(DOC, ()), DOC)

    def test_array_element_by_index(self):
        self.assertEqual(ini_lookup(DOC, "server.hosts.1"), "b")

    def test_array_first_and_last_elements(self):
        self.assertEqual((ini_lookup(DOC, "server.hosts.0"), ini_lookup(DOC, "server.hosts.2")), ("a", "c"))

    def test_negative_index_counts_from_the_end(self):
        self.assertEqual(ini_lookup(DOC, "server.hosts.-1"), "c")

    def test_first_negative_index_is_minus_length(self):
        self.assertEqual(ini_lookup(DOC, "server.hosts.-3"), "a")

    def test_index_past_the_end_is_missing(self):
        self.assertEqual(ini_lookup(DOC, "server.hosts.3", "dflt"), "dflt")

    def test_negative_index_past_the_start_is_missing(self):
        self.assertEqual(ini_lookup(DOC, "server.hosts.-4", "dflt"), "dflt")

    def test_non_numeric_array_segment_is_missing(self):
        self.assertEqual(ini_lookup(DOC, "server.hosts.x", "dflt"), "dflt")

    def test_integer_segment_in_a_tuple_path(self):
        self.assertEqual(ini_lookup(DOC, ("server", "hosts", 1)), "b")

    def test_quoted_segment_with_dot(self):
        self.assertEqual(ini_lookup(DOC, '"a b"."c.d"'), 1)

    def test_missing_key_gives_the_default(self):
        self.assertEqual(ini_lookup(DOC, "server.nope", 99), 99)

    def test_falsy_default_is_returned_as_is(self):
        self.assertEqual(ini_lookup(DOC, "nope", 0), 0)

    def test_none_default_is_allowed(self):
        self.assertIsNone(ini_lookup(DOC, "nope", None))

    def test_existing_value_beats_the_default(self):
        self.assertEqual(ini_lookup(DOC, "name", "other"), "demo")

    def test_missing_without_default_raises(self):
        self.assertEqual(refusal(ini_lookup, DOC, "server.nope"), "no value at server.nope")

    def test_error_path_quotes_odd_segments(self):
        self.assertEqual(refusal(ini_lookup, DOC, ('a b', 'x')), 'no value at "a b".x')

    def test_descending_through_a_scalar_is_missing(self):
        self.assertEqual(ini_lookup(DOC, "name.first", "dflt"), "dflt")

    def test_lookup_error_is_a_lookup_error(self):
        with self.assertRaises(LookupError):
            ini_lookup(DOC, "nope")

    def test_no_default_marker_is_distinct(self):
        self.assertIsNot(INI_NO_DEFAULT, None)


class HasPathTests(unittest.TestCase):
    def test_present_path_is_reported_present(self):
        self.assertTrue(ini_has_path(DOC, "server.port"))

    def test_absent_path_is_reported_absent(self):
        self.assertFalse(ini_has_path(DOC, "server.host"))

    def test_empty_table_counts_as_present(self):
        self.assertTrue(ini_has_path(DOC, "empty"))

    def test_array_index_in_range(self):
        self.assertTrue(ini_has_path(DOC, "server.hosts.2"))

    def test_array_index_out_of_range(self):
        self.assertFalse(ini_has_path(DOC, "server.hosts.3"))


class LookupKindTests(unittest.TestCase):
    def test_matching_kind_returns_the_value(self):
        self.assertEqual(ini_lookup_kind(DOC, "server.port", "integer"), 8080)

    def test_float_kind_is_distinct_from_integer(self):
        self.assertEqual(refusal(ini_lookup_kind, DOC, "server.ratio", "integer"),
                         "server.ratio is float, expected integer")

    def test_table_kind_is_recognized(self):
        self.assertEqual(ini_lookup_kind(DOC, "server.tls", "table"), {"on": True})

    def test_wrong_kind_raises_even_with_a_default(self):
        with self.assertRaises(IniLookupError):
            ini_lookup_kind(DOC, "name", "integer", 5)

    def test_missing_uses_the_default(self):
        self.assertEqual(ini_lookup_kind(DOC, "nope", "integer", 5), 5)

    def test_missing_without_default_raises(self):
        self.assertEqual(refusal(ini_lookup_kind, DOC, "nope", "integer"), "no value at nope")


class LookupFirstTests(unittest.TestCase):
    def test_first_existing_path_wins(self):
        self.assertEqual(ini_lookup_first(DOC, ["x.y", "server.port", "name"]), 8080)

    def test_earlier_path_beats_later_one(self):
        self.assertEqual(ini_lookup_first(DOC, ["name", "server.port"]), "demo")

    def test_nothing_found_uses_the_default(self):
        self.assertEqual(ini_lookup_first(DOC, ["x", "y"], "dflt"), "dflt")

    def test_nothing_found_without_default_lists_paths(self):
        self.assertEqual(refusal(ini_lookup_first, DOC, ["x", "y.z"]), "none of the paths has a value: x, y.z")

    def test_empty_path_list_with_default(self):
        self.assertEqual(ini_lookup_first(DOC, [], 1), 1)


class ListPathsTests(unittest.TestCase):
    def test_flat_document_lists_its_keys(self):
        self.assertEqual(ini_list_paths({"a": 1, "b": 2}), ["a", "b"])

    def test_nested_tables_are_walked_in_order(self):
        self.assertEqual(ini_list_paths(DOC), ["name", "server.port", "server.hosts", "server.tls.on", "server.ratio",
                                               "empty", '"a b"."c.d"'])

    def test_arrays_count_as_single_leaves(self):
        self.assertEqual(ini_list_paths({"a": [1, 2]}), ["a"])

    def test_empty_table_is_a_leaf(self):
        self.assertEqual(ini_list_paths({"a": {}}), ["a"])

    def test_empty_document_has_no_paths(self):
        self.assertEqual(ini_list_paths({}), [])


if __name__ == "__main__":
    unittest.main()
