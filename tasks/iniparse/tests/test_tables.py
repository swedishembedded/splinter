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


class BareKeyTests(unittest.TestCase):
    def test_letters_digits_dash_underscore_are_bare(self):
        self.assertTrue(ini_is_bare_key("a-B_9"))

    def test_space_makes_a_key_non_bare(self):
        self.assertFalse(ini_is_bare_key("a b"))

    def test_dot_makes_a_key_non_bare(self):
        self.assertFalse(ini_is_bare_key("a.b"))

    def test_empty_key_is_not_bare(self):
        self.assertFalse(ini_is_bare_key(""))

    def test_non_ascii_letter_is_not_bare(self):
        self.assertFalse(ini_is_bare_key("caf\u00e9"))

    def test_whole_text_must_match(self):
        self.assertFalse(ini_is_bare_key("ab\n"))


class SplitKeyPathTests(unittest.TestCase):
    def test_lone_name_is_one_segment(self):
        self.assertEqual(ini_split_key_path("name"), ("name",))

    def test_dots_split_names_into_segments(self):
        self.assertEqual(ini_split_key_path("a.b.c"), ("a", "b", "c"))

    def test_spaces_around_dots_are_ignored(self):
        self.assertEqual(ini_split_key_path(" a . b "), ("a", "b"))

    def test_quoted_segment_keeps_its_dots(self):
        self.assertEqual(ini_split_key_path('a."b.c".d'), ("a", "b.c", "d"))

    def test_single_quoted_segment_is_literal(self):
        self.assertEqual(ini_split_key_path("'x y'.z"), ("x y", "z"))

    def test_quoted_segment_decodes_escapes(self):
        self.assertEqual(ini_split_key_path('"a\\tb"'), ("a\tb",))

    def test_empty_quoted_segment_is_allowed(self):
        self.assertEqual(ini_split_key_path('a.""'), ("a", ""))

    def test_empty_text_has_an_empty_segment(self):
        self.assertEqual(failure(ini_split_key_path, "", 1, 1), "line 1, column 1: empty key segment")

    def test_trailing_dot_reports_its_column(self):
        self.assertEqual(failure(ini_split_key_path, "a.", 2, 5), "line 2, column 7: empty key segment")

    def test_leading_dot_reports_the_start(self):
        self.assertEqual(failure(ini_split_key_path, ".a", 1, 3), "line 1, column 3: empty key segment")

    def test_doubled_dot_is_refused(self):
        self.assertEqual(failure(ini_split_key_path, "a..b", 1, 1), "line 1, column 3: empty key segment")

    def test_invalid_character_reports_the_segment(self):
        self.assertEqual(failure(ini_split_key_path, "ok.b@d", 1, 1), "line 1, column 4: invalid key 'b@d'")

    def test_invalid_segment_column_skips_spaces(self):
        self.assertEqual(failure(ini_split_key_path, "a.  b c", 1, 10), "line 1, column 14: invalid key 'b c'")

    def test_bad_quoted_segment_reports_inside(self):
        self.assertEqual(failure(ini_split_key_path, 'a."b', 1, 1), "line 1, column 3: unterminated string")

    def test_text_after_quoted_segment_is_refused(self):
        self.assertEqual(failure(ini_split_key_path, '"a"b', 1, 1), "line 1, column 4: unexpected text after string")


class DefaultPositionTests(unittest.TestCase):
    def test_empty_segment_message_without_location(self):
        self.assertEqual(failure(ini_split_key_path, "a..b"), "empty key segment")

    def test_malformed_header_message_stands_alone(self):
        self.assertEqual(failure(ini_parse_section_header, "[abc"), "malformed section header")

    def test_bad_header_segment_needs_no_location(self):
        self.assertEqual(failure(ini_parse_section_header, "[a b]"), "invalid key 'a b'")

    def test_blocked_table_message_has_no_prefix(self):
        self.assertEqual(failure(ini_open_table, {"a": 1}, ("a",)), "'a' is not a table")


class JoinKeyPathTests(unittest.TestCase):
    def test_bare_segments_stay_bare(self):
        self.assertEqual(ini_join_key_path(("a", "b-1")), "a.b-1")

    def test_odd_segments_are_quoted(self):
        self.assertEqual(ini_join_key_path(("a b", "c")), '"a b".c')

    def test_empty_segment_becomes_empty_quotes(self):
        self.assertEqual(ini_join_key_path(("",)), '""')

    def test_quote_key_escapes_content(self):
        self.assertEqual(ini_quote_key('a"b'), '"a\\"b"')

    def test_quote_key_leaves_bare_keys(self):
        self.assertEqual(ini_quote_key("plain"), "plain")

    def test_join_then_split_returns_segments(self):
        segments = ("a b", "c.d", "", "e\n")
        self.assertEqual(ini_split_key_path(ini_join_key_path(segments)), segments)


class SectionHeaderTests(unittest.TestCase):
    def test_plain_header_gives_one_segment(self):
        self.assertEqual(ini_parse_section_header("[server]"), ("server",))

    def test_dotted_header_gives_each_segment(self):
        self.assertEqual(ini_parse_section_header("[a.b.c]"), ("a", "b", "c"))

    def test_spaces_inside_brackets_are_ignored(self):
        self.assertEqual(ini_parse_section_header("[ a . b ]"), ("a", "b"))

    def test_quotes_protect_dots_in_header(self):
        self.assertEqual(ini_parse_section_header('["a.b"]'), ("a.b",))

    def test_header_without_closing_bracket(self):
        self.assertEqual(failure(ini_parse_section_header, "[abc", 5, 2), "line 5, column 2: malformed section header")

    def test_text_after_header_is_malformed(self):
        self.assertEqual(failure(ini_parse_section_header, "[a] x", 1, 1), "line 1, column 1: malformed section header")

    def test_lone_bracket_is_malformed(self):
        self.assertEqual(failure(ini_parse_section_header, "[", 1, 1), "line 1, column 1: malformed section header")

    def test_empty_header_has_an_empty_segment(self):
        self.assertEqual(failure(ini_parse_section_header, "[]", 1, 4), "line 1, column 5: empty key segment")

    def test_segment_error_column_is_inside_the_brackets(self):
        self.assertEqual(failure(ini_parse_section_header, "[a.b c]", 2, 3), "line 2, column 6: invalid key 'b c'")

    def test_double_brackets_are_refused(self):
        self.assertEqual(failure(ini_parse_section_header, "[[a]]", 1, 1), "line 1, column 2: invalid key '[a]'")


class OpenTableTests(unittest.TestCase):
    def test_empty_path_returns_the_root(self):
        root = {}
        self.assertIs(ini_open_table(root, ()), root)

    def test_missing_tables_are_created(self):
        root = {}
        ini_open_table(root, ("a", "b"))
        self.assertEqual(root, {"a": {"b": {}}})

    def test_existing_table_is_reused(self):
        root = {"a": {"x": 1}}
        table = ini_open_table(root, ("a",))
        self.assertIs(table, root["a"])
        self.assertEqual(root, {"a": {"x": 1}})

    def test_returned_table_is_the_innermost(self):
        root = {}
        ini_open_table(root, ("a", "b"))["k"] = 1
        self.assertEqual(root, {"a": {"b": {"k": 1}}})

    def test_scalar_in_the_way_is_refused(self):
        self.assertEqual(failure(ini_open_table, {"a": 1}, ("a", "b"), 3, 4), "line 3, column 4: 'a' is not a table")

    def test_deeper_scalar_names_the_prefix(self):
        self.assertEqual(failure(ini_open_table, {"a": {"b": []}}, ("a", "b", "c")), "'a.b' is not a table")

    def test_quoted_prefix_in_message(self):
        self.assertEqual(failure(ini_open_table, {"a b": 1}, ("a b",)), "'\"a b\"' is not a table")


if __name__ == "__main__":
    unittest.main()
