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


class C1(unittest.TestCase):


    def test_array_first_and_last_elements(self):
        self.assertEqual((ini_lookup(DOC, "server.hosts.0"), ini_lookup(DOC, "server.hosts.2")), ("a", "c"))


    def test_first_negative_index_is_minus_length(self):
        self.assertEqual(ini_lookup(DOC, "server.hosts.-3"), "a")

    def test_index_past_the_end_is_missing(self):
        self.assertEqual(ini_lookup(DOC, "server.hosts.3", "dflt"), "dflt")

    def test_negative_index_past_the_start_is_missing(self):
        self.assertEqual(ini_lookup(DOC, "server.hosts.-4", "dflt"), "dflt")

    def test_non_numeric_array_segment_is_missing(self):
        self.assertEqual(ini_lookup(DOC, "server.hosts.x", "dflt"), "dflt")


    def test_none_default_is_allowed(self):
        self.assertIsNone(ini_lookup(DOC, "nope", None))


    def test_error_path_quotes_odd_segments(self):
        self.assertEqual(refusal(ini_lookup, DOC, ('a b', 'x')), 'no value at "a b".x')


class C2(unittest.TestCase):


    def test_array_index_out_of_range(self):
        self.assertFalse(ini_has_path(DOC, "server.hosts.3"))


class C3(unittest.TestCase):
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


class C4(unittest.TestCase):
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


class C5(unittest.TestCase):
    def test_flat_document_lists_its_keys(self):
        self.assertEqual(ini_list_paths({"a": 1, "b": 2}), ["a", "b"])

    def test_nested_tables_are_walked_in_order(self):
        self.assertEqual(ini_list_paths(DOC), ["name", "server.port", "server.hosts", "server.tls.on", "server.ratio",
                                               "empty", '"a b"."c.d"'])

    def test_arrays_count_as_single_leaves(self):
        self.assertEqual(ini_list_paths({"a": [1, 2]}), ["a"])

    def test_empty_table_is_a_leaf(self):
        self.assertEqual(ini_list_paths({"a": {}}), ["a"])


if __name__ == "__main__":
    unittest.main()
