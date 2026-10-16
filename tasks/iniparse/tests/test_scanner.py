# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
import dataclasses
import unittest

from iniparse.scanner import (
    IniLine, ini_bracket_depth, ini_classify_line, ini_find_comment_start, ini_scan_lines,
    ini_split_lines, ini_split_pair, ini_strip_comment, ini_unquoted_chars,
)


class C1(unittest.TestCase):


    def test_trailing_newline_adds_no_extra_line(self):
        self.assertEqual(ini_split_lines("a\nb\n"), ["a", "b"])

    def test_two_trailing_newlines_keep_one_blank(self):
        self.assertEqual(ini_split_lines("a\n\n"), ["a", ""])


class C2(unittest.TestCase):


    def test_escaped_quote_does_not_close_double_quotes(self):
        self.assertEqual(list(ini_unquoted_chars('"a\\"b"c')), [(6, "c")])


class C3(unittest.TestCase):


    def test_no_comment_gives_minus_one(self):
        self.assertEqual(ini_find_comment_start("a = 1"), -1)

    def test_marker_inside_quotes_is_ignored(self):
        self.assertEqual(ini_find_comment_start('a = "x # y"'), -1)


    def test_comment_text_starts_right_after_marker(self):
        self.assertEqual(ini_strip_comment("a #note"), ("a", "note"))

    def test_strip_comment_only_line(self):
        self.assertEqual(ini_strip_comment("# whole"), ("", "whole"))

    def test_comment_keeps_later_markers(self):
        self.assertEqual(ini_strip_comment("a # b # c"), ("a", "b # c"))


class C4(unittest.TestCase):
    def test_balanced_brackets_give_zero(self):
        self.assertEqual(ini_bracket_depth("[1, [2]]"), 0)


    def test_extra_close_gives_negative_depth(self):
        self.assertEqual(ini_bracket_depth("1]"), -1)

    def test_quoted_brackets_do_not_count(self):
        self.assertEqual(ini_bracket_depth('["]", "["]'), 0)


class C5(unittest.TestCase):


    def test_comment_only_is_a_comment(self):
        self.assertEqual(ini_classify_line("", "hi"), "comment")

    def test_empty_input_is_a_blank_line(self):
        self.assertEqual(ini_classify_line("", ""), "blank")


class C6(unittest.TestCase):

    def test_pair_line_reports_every_field(self):
        self.assertEqual(ini_scan_lines("a = 1 # n"), [IniLine(1, 1, "pair", "a = 1", "n", 0)])


    def test_scanned_line_cannot_be_modified(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            ini_scan_lines("a = 1")[0].text = "b"


    def test_kinds_follow_line_order(self):
        kinds = [line.kind for line in ini_scan_lines("# c\n\n[s]\nk = v\n")]
        self.assertEqual(kinds, ["comment", "blank", "section", "pair"])


if __name__ == "__main__":
    unittest.main()
