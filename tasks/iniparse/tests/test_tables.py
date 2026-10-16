# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
import unittest

from iniparse.diagnostics import IniSyntaxError
from iniparse.tables import (
    ini_is_bare_key, ini_join_key_path, ini_open_table, ini_parse_section_header,
    ini_quote_key, ini_split_key_path,
)


def failure(call, *args):
    try:
        call(*args)
    except IniSyntaxError as error:
        return str(error)
    raise AssertionError("no error raised")


class C1(unittest.TestCase):


    def test_single_quoted_segment_is_literal(self):
        self.assertEqual(ini_split_key_path("'x y'.z"), ("x y", "z"))


    def test_bad_quoted_segment_reports_inside(self):
        self.assertEqual(failure(ini_split_key_path, 'a."b', 1, 1), "line 1, column 3: unterminated string")

    def test_text_after_quoted_segment_is_refused(self):
        self.assertEqual(failure(ini_split_key_path, '"a"b', 1, 1), "line 1, column 4: unexpected text after string")


class C2(unittest.TestCase):
    def test_empty_segment_message_without_location(self):
        self.assertEqual(failure(ini_split_key_path, "a..b"), "empty key segment")

    def test_malformed_header_message_stands_alone(self):
        self.assertEqual(failure(ini_parse_section_header, "[abc"), "malformed section header")

    def test_bad_header_segment_needs_no_location(self):
        self.assertEqual(failure(ini_parse_section_header, "[a b]"), "invalid key 'a b'")

    def test_blocked_table_message_has_no_prefix(self):
        self.assertEqual(failure(ini_open_table, {"a": 1}, ("a",)), "'a' is not a table")


class C3(unittest.TestCase):


    def test_join_then_split_returns_segments(self):
        segments = ("a b", "c.d", "", "e\n")
        self.assertEqual(ini_split_key_path(ini_join_key_path(segments)), segments)


class C4(unittest.TestCase):


    def test_lone_bracket_is_malformed(self):
        self.assertEqual(failure(ini_parse_section_header, "[", 1, 1), "line 1, column 1: malformed section header")

    def test_empty_header_has_an_empty_segment(self):
        self.assertEqual(failure(ini_parse_section_header, "[]", 1, 4), "line 1, column 5: empty key segment")

    def test_segment_error_column_is_inside_the_brackets(self):
        self.assertEqual(failure(ini_parse_section_header, "[a.b c]", 2, 3), "line 2, column 6: invalid key 'b c'")

    def test_double_brackets_are_refused(self):
        self.assertEqual(failure(ini_parse_section_header, "[[a]]", 1, 1), "line 1, column 2: invalid key '[a]'")


class C5(unittest.TestCase):


    def test_scalar_in_the_way_is_refused(self):
        self.assertEqual(failure(ini_open_table, {"a": 1}, ("a", "b"), 3, 4), "line 3, column 4: 'a' is not a table")

    def test_deeper_scalar_names_the_prefix(self):
        self.assertEqual(failure(ini_open_table, {"a": {"b": []}}, ("a", "b", "c")), "'a.b' is not a table")

    def test_quoted_prefix_in_message(self):
        self.assertEqual(failure(ini_open_table, {"a b": 1}, ("a b",)), "'\"a b\"' is not a table")


if __name__ == "__main__":
    unittest.main()
