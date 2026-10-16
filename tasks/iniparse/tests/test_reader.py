# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
import unittest

from iniparse.diagnostics import IniDuplicateError, IniSyntaxError
from iniparse.reader import ini_collect_diagnostics, ini_load_document, ini_report_problems


def failure(text):
    try:
        ini_load_document(text)
    except IniSyntaxError as error:
        return str(error)
    raise AssertionError("no error raised")


class C1(unittest.TestCase):

    def test_comments_and_blank_lines_are_ignored(self):
        self.assertEqual(ini_load_document("# one\n\n; two\n"), {})

    def test_top_level_pairs_land_in_the_root(self):
        self.assertEqual(ini_load_document("a = 1\nb = 'x'"), {"a": 1, "b": "x"})


    def test_array_over_several_lines(self):
        self.assertEqual(ini_load_document("a = [\n  1, # one\n  2,\n]\nb = 3"), {"a": [1, 2], "b": 3})


class C2(unittest.TestCase):


    def test_empty_section_makes_an_empty_table(self):
        self.assertEqual(ini_load_document("[s]"), {"s": {}})

    def test_dotted_key_creates_nested_tables(self):
        self.assertEqual(ini_load_document("a.b.c = 1"), {"a": {"b": {"c": 1}}})

    def test_dotted_key_inside_a_section(self):
        self.assertEqual(ini_load_document("[s]\nx.y = 2"), {"s": {"x": {"y": 2}}})

    def test_dotted_keys_share_a_table(self):
        self.assertEqual(ini_load_document("a.b = 1\na.c = 2"), {"a": {"b": 1, "c": 2}})


class C3(unittest.TestCase):


    def test_pair_with_nothing_after_equals_is_refused(self):
        self.assertEqual(failure("a ="), "line 1, column 4: missing value")

    def test_missing_value_column_follows_indent(self):
        self.assertEqual(failure("  key  ="), "line 1, column 9: missing value")


    def test_bad_array_element_points_inside(self):
        self.assertEqual(failure("k = [1, x]"), "line 1, column 9: unrecognized value 'x'")

    def test_unclosed_string_value_is_refused(self):
        self.assertEqual(failure('k = "abc'), "line 1, column 5: unterminated string")


    def test_key_with_a_space_is_refused(self):
        self.assertEqual(failure("a b = 1"), "line 1, column 1: invalid key 'a b'")

    def test_unclosed_section_header_is_refused(self):
        self.assertEqual(failure("a = 1\n[oops"), "line 2, column 1: malformed section header")

    def test_scalar_blocks_dotted_key(self):
        self.assertEqual(failure("a = 1\na.b = 2"), "line 2, column 1: 'a' is not a table")

    def test_scalar_blocks_section_path(self):
        self.assertEqual(failure("a = 1\n[a.b]"), "line 2, column 1: 'a' is not a table")

    def test_multiline_array_error_uses_its_first_line(self):
        self.assertEqual(failure("x = 1\ny = [1,\n  bad]"), "line 2, column 9: unrecognized value 'bad'")


class C4(unittest.TestCase):
    def test_duplicate_key_names_first_line(self):
        self.assertEqual(failure("a = 1\nb = 2\na = 3"), "line 3, column 1: duplicate key 'a' (first defined on line 1)")

    def test_duplicate_key_is_a_duplicate_error(self):
        with self.assertRaises(IniDuplicateError):
            ini_load_document("a = 1\na = 1")


    def test_duplicate_key_inside_section_shows_full_path(self):
        self.assertEqual(failure("[s]\na = 1\na = 2"), "line 3, column 1: duplicate key 's.a' (first defined on line 2)")

    def test_repeated_header_is_refused(self):
        self.assertEqual(failure("[s]\n[t]\n[s]"), "line 3, column 1: duplicate section 's' (first defined on line 1)")

    def test_duplicate_dotted_section_shows_path(self):
        self.assertEqual(failure("[a.b]\n[a.b]"), "line 2, column 1: duplicate section 'a.b' (first defined on line 1)")

    def test_key_over_a_dotted_key_table(self):
        self.assertEqual(failure("[s]\nx.y = 1\nx = 2"), "line 3, column 1: key 's.x' is already a table")


class C5(unittest.TestCase):


    def test_keys_after_a_bad_header_are_skipped(self):
        document, errors = ini_collect_diagnostics("[s]\na = 1\n[oops\nb = 2\n[t]\nc = 3")
        self.assertEqual(document, {"s": {"a": 1}, "t": {"c": 3}})
        self.assertEqual(len(errors), 1)


class C6(unittest.TestCase):
    def test_clean_text_gives_empty_report(self):
        self.assertEqual(ini_report_problems("a = 1"), "")

    def test_report_shows_excerpt_and_caret(self):
        self.assertEqual(ini_report_problems("k = ?"), "line 1, column 5: unrecognized value '?'\nk = ?\n    ^")

    def test_report_stops_at_the_given_limit(self):
        report = ini_report_problems("a = ?\nb = ?\nc = ?", 2)
        self.assertEqual(report.splitlines()[-1], "(and 1 more)")

    def test_default_limit_is_three(self):
        report = ini_report_problems("a = ?\nb = ?\nc = ?\nd = ?")
        self.assertEqual(report.splitlines()[-1], "(and 1 more)")
        self.assertEqual(report.count("unrecognized"), 3)


if __name__ == "__main__":
    unittest.main()
