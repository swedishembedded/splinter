# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
import dataclasses
import unittest

from iniparse.scanner import (
    IniLine, ini_bracket_depth, ini_classify_line, ini_find_comment_start, ini_scan_lines,
    ini_split_lines, ini_split_pair, ini_strip_comment, ini_unquoted_chars,
)


class SplitLinesTests(unittest.TestCase):
    def test_empty_text_has_no_lines(self):
        self.assertEqual(ini_split_lines(""), [])

    def test_single_line_without_newline(self):
        self.assertEqual(ini_split_lines("a = 1"), ["a = 1"])

    def test_trailing_newline_adds_no_extra_line(self):
        self.assertEqual(ini_split_lines("a\nb\n"), ["a", "b"])

    def test_two_trailing_newlines_keep_one_blank(self):
        self.assertEqual(ini_split_lines("a\n\n"), ["a", ""])

    def test_windows_line_ends_are_unified(self):
        self.assertEqual(ini_split_lines("a\r\nb"), ["a", "b"])

    def test_bare_carriage_return_ends_a_line(self):
        self.assertEqual(ini_split_lines("a\rb"), ["a", "b"])

    def test_lone_newline_is_one_blank_line(self):
        self.assertEqual(ini_split_lines("\n"), [""])


class UnquotedCharsTests(unittest.TestCase):
    def test_plain_text_yields_every_character(self):
        self.assertEqual(list(ini_unquoted_chars("ab")), [(0, "a"), (1, "b")])

    def test_double_quoted_text_is_hidden(self):
        self.assertEqual(list(ini_unquoted_chars('a"x"b')), [(0, "a"), (4, "b")])

    def test_single_quoted_text_is_hidden(self):
        self.assertEqual(list(ini_unquoted_chars("a'x'b")), [(0, "a"), (4, "b")])

    def test_escaped_quote_does_not_close_double_quotes(self):
        self.assertEqual(list(ini_unquoted_chars('"a\\"b"c')), [(6, "c")])

    def test_backslash_is_literal_in_single_quotes(self):
        self.assertEqual(list(ini_unquoted_chars("'a\\'b")), [(4, "b")])

    def test_other_quote_kind_is_ordinary_inside_quotes(self):
        self.assertEqual(list(ini_unquoted_chars("\"a'b\"c")), [(5, "c")])

    def test_unterminated_quote_hides_the_rest(self):
        self.assertEqual(list(ini_unquoted_chars('a"bc')), [(0, "a")])


class CommentTests(unittest.TestCase):
    def test_hash_starts_a_comment(self):
        self.assertEqual(ini_find_comment_start("a = 1 # x"), 6)

    def test_semicolon_starts_a_comment(self):
        self.assertEqual(ini_find_comment_start("a = 1 ; x"), 6)

    def test_no_comment_gives_minus_one(self):
        self.assertEqual(ini_find_comment_start("a = 1"), -1)

    def test_marker_inside_quotes_is_ignored(self):
        self.assertEqual(ini_find_comment_start('a = "x # y"'), -1)

    def test_marker_after_closing_quote_counts(self):
        self.assertEqual(ini_find_comment_start('a = "x" # y'), 8)

    def test_earliest_marker_ends_the_content(self):
        self.assertEqual(ini_find_comment_start("a ;b #c"), 2)

    def test_strip_returns_content_and_comment(self):
        self.assertEqual(ini_strip_comment("  a = 1  # note "), ("a = 1", "note"))

    def test_strip_without_comment_trims_whitespace(self):
        self.assertEqual(ini_strip_comment("  a = 1  "), ("a = 1", ""))

    def test_comment_text_starts_right_after_marker(self):
        self.assertEqual(ini_strip_comment("a #note"), ("a", "note"))

    def test_strip_comment_only_line(self):
        self.assertEqual(ini_strip_comment("# whole"), ("", "whole"))

    def test_comment_keeps_later_markers(self):
        self.assertEqual(ini_strip_comment("a # b # c"), ("a", "b # c"))


class BracketDepthTests(unittest.TestCase):
    def test_balanced_brackets_give_zero(self):
        self.assertEqual(ini_bracket_depth("[1, [2]]"), 0)

    def test_open_bracket_gives_positive_depth(self):
        self.assertEqual(ini_bracket_depth("[1, [2"), 2)

    def test_extra_close_gives_negative_depth(self):
        self.assertEqual(ini_bracket_depth("1]"), -1)

    def test_quoted_brackets_do_not_count(self):
        self.assertEqual(ini_bracket_depth('["]", "["]'), 0)

    def test_no_brackets_gives_zero(self):
        self.assertEqual(ini_bracket_depth("abc"), 0)


class SplitPairTests(unittest.TestCase):
    def test_simple_pair_with_offset(self):
        self.assertEqual(ini_split_pair("a = 1"), ("a", "1", 4))

    def test_equals_sign_needs_no_surrounding_spaces(self):
        self.assertEqual(ini_split_pair("a=1"), ("a", "1", 2))

    def test_line_without_equals_is_none(self):
        self.assertIsNone(ini_split_pair("abc"))

    def test_equals_inside_quoted_key_is_skipped(self):
        self.assertEqual(ini_split_pair('"a=b" = 2'), ('"a=b"', "2", 8))

    def test_only_first_equals_splits(self):
        self.assertEqual(ini_split_pair("a = b = c"), ("a", "b = c", 4))

    def test_empty_value_is_empty_text(self):
        self.assertEqual(ini_split_pair("a ="), ("a", "", 3))

    def test_wide_gap_moves_the_offset(self):
        self.assertEqual(ini_split_pair("a =    7"), ("a", "7", 7))


class ClassifyTests(unittest.TestCase):
    def test_bracket_content_is_a_section(self):
        self.assertEqual(ini_classify_line("[a]", ""), "section")

    def test_other_content_is_a_pair(self):
        self.assertEqual(ini_classify_line("a = 1", ""), "pair")

    def test_comment_only_is_a_comment(self):
        self.assertEqual(ini_classify_line("", "hi"), "comment")

    def test_empty_input_is_a_blank_line(self):
        self.assertEqual(ini_classify_line("", ""), "blank")

    def test_content_wins_over_comment(self):
        self.assertEqual(ini_classify_line("a = 1", "hi"), "pair")


class ScanLinesTests(unittest.TestCase):
    def test_empty_text_scans_to_nothing(self):
        self.assertEqual(ini_scan_lines(""), [])

    def test_pair_line_reports_every_field(self):
        self.assertEqual(ini_scan_lines("a = 1 # n"), [IniLine(1, 1, "pair", "a = 1", "n", 0)])

    def test_indent_counts_leading_whitespace(self):
        self.assertEqual(ini_scan_lines("   a = 1")[0].indent, 3)

    def test_scanned_line_cannot_be_modified(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            ini_scan_lines("a = 1")[0].text = "b"

    def test_bracketed_line_is_a_section(self):
        self.assertEqual(ini_scan_lines("[a.b]")[0].kind, "section")

    def test_kinds_follow_line_order(self):
        kinds = [line.kind for line in ini_scan_lines("# c\n\n[s]\nk = v\n")]
        self.assertEqual(kinds, ["comment", "blank", "section", "pair"])

    def test_line_numbers_start_at_one(self):
        numbers = [line.number for line in ini_scan_lines("a = 1\n\nb = 2")]
        self.assertEqual(numbers, [1, 2, 3])

    def test_open_array_continues_over_lines(self):
        lines = ini_scan_lines("a = [1,\n  2]\nb = 3")
        self.assertEqual([(l.number, l.end, l.text) for l in lines],
                         [(1, 2, "a = [1, 2]"), (3, 3, "b = 3")])

    def test_continuation_drops_comments_of_later_lines(self):
        lines = ini_scan_lines("a = [1, # one\n 2 # two\n]")
        self.assertEqual(lines[0].text, "a = [1, 2 ]")
        self.assertEqual(lines[0].comment, "one")

    def test_unclosed_array_swallows_to_end_of_text(self):
        lines = ini_scan_lines("a = [1,\nb = 2")
        self.assertEqual(len(lines), 1)
        self.assertEqual((lines[0].end, lines[0].text), (2, "a = [1, b = 2"))

    def test_section_line_is_never_continued(self):
        self.assertEqual(len(ini_scan_lines("[a\nb = 1")), 2)

    def test_comment_marker_in_string_stays_in_text(self):
        self.assertEqual(ini_scan_lines('a = "x#y"')[0].text, 'a = "x#y"')


if __name__ == "__main__":
    unittest.main()
