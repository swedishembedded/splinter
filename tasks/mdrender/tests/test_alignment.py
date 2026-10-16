# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
import unittest

from mdrender.alignment import (
    mdr_align_lines,
    mdr_apply_hanging_indent,
    mdr_center_block,
    mdr_dedent_block,
    mdr_indent_block,
    mdr_join_columns,
    mdr_justify_line,
    mdr_justify_lines,
    mdr_pad_line,
    mdr_strip_trailing_space,
    mdr_trim_blank_edges,
)


class C0(unittest.TestCase):
    def test_right_alignment_pads_on_the_left(self):
        self.assertEqual(mdr_pad_line("ab", 5, "right"), "   ab")

    def test_unknown_alignment_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            mdr_pad_line("a", 3, "middle")
        self.assertEqual(str(caught.exception), "unknown alignment: middle")

    def test_two_character_fill_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            mdr_pad_line("a", 3, "left", "ab")
        self.assertEqual(str(caught.exception), "fill must be one character")

    def test_empty_fill_is_rejected(self):
        with self.assertRaises(ValueError):
            mdr_pad_line("a", 3, "left", "")


class C1(unittest.TestCase):
    def test_leftover_spaces_go_to_the_leftmost_gaps(self):
        self.assertEqual(mdr_justify_line("a b c d", 11), "a   b  c  d")

    def test_one_extra_space_fills_first_gap(self):
        self.assertEqual(mdr_justify_line("a b c", 6), "a  b c")

    def test_single_word_is_padded_on_the_right(self):
        self.assertEqual(mdr_justify_line("word", 6), "word  ")

    def test_final_paragraph_line_stays_ragged(self):
        self.assertEqual(mdr_justify_lines(["a b", "c d"], 5), ["a   b", "c d"])

    def test_justify_last_flag_justifies_the_final_line(self):
        self.assertEqual(mdr_justify_lines(["a b", "c d"], 5, justify_last=True), ["a   b", "c   d"])

    def test_line_before_blank_ends_paragraph(self):
        self.assertEqual(mdr_justify_lines(["a b", "c d", "", "e f", "g h"], 5),
                         ["a   b", "c d", "", "e   f", "g h"])

    def test_blank_lines_pass_through_unchanged(self):
        self.assertEqual(mdr_justify_lines(["", "a b", "c"], 4, justify_last=True), ["", "a  b", "c   "])


class C2(unittest.TestCase):
    def test_right_alignment_pads_every_line(self):
        self.assertEqual(mdr_align_lines(["a", "bb"], 3, "right"), ["  a", " bb"])

    def test_justify_mode_keeps_the_last_line_short(self):
        self.assertEqual(mdr_align_lines(["a b", "c"], 5, "justify"), ["a   b", "c"])

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            mdr_align_lines(["a"], 3, "full")

    def test_block_is_centered_on_its_widest_line(self):
        self.assertEqual(mdr_center_block(["a", "bbb"]), [" a ", "bbb"])

    def test_block_is_centered_on_a_given_width(self):
        self.assertEqual(mdr_center_block(["ab"], 6), ["  ab  "])


class C3(unittest.TestCase):
    def test_indent_prefixes_non_blank_lines(self):
        self.assertEqual(mdr_indent_block("a\n\nb", "  "), "  a\n\n  b")

    def test_indent_leaves_whitespace_only_lines_alone(self):
        self.assertEqual(mdr_indent_block("a\n  \nb", "-"), "-a\n  \n-b")

    def test_dedent_without_margin_changes_nothing(self):
        self.assertEqual(mdr_dedent_block("a\n  b"), "a\n  b")

    def test_dedent_stops_at_the_first_letter(self):
        self.assertEqual(mdr_dedent_block("  ab\n  ac"), "ab\nac")

    def test_hanging_indent_uses_two_prefixes(self):
        self.assertEqual(mdr_apply_hanging_indent(["a", "b", "c"], "1. ", "   "), ["1. a", "   b", "   c"])

    def test_hanging_indent_single_line_first_prefix(self):
        self.assertEqual(mdr_apply_hanging_indent(["a"], "* ", "  "), ["* a"])


class C4(unittest.TestCase):
    def test_trailing_space_is_stripped_from_each_line(self):
        self.assertEqual(mdr_strip_trailing_space(["a  ", " b\t", ""]), ["a", " b", ""])

    def test_single_trailing_blank_is_trimmed(self):
        self.assertEqual(mdr_trim_blank_edges(["a", "b", ""]), ["a", "b"])

    def test_mixed_blank_edges_all_trimmed(self):
        self.assertEqual(mdr_trim_blank_edges(["", "a", "", ""]), ["a"])

    def test_no_lines_trim_to_no_lines(self):
        self.assertEqual(mdr_trim_blank_edges([]), [])

    def test_all_blank_lines_trim_to_nothing(self):
        self.assertEqual(mdr_trim_blank_edges(["", "  "]), [])

    def test_lines_without_blank_edges_are_kept(self):
        self.assertEqual(mdr_trim_blank_edges(["a", "b"]), ["a", "b"])


class C5(unittest.TestCase):
    def test_columns_padded_and_separated_by_gap(self):
        self.assertEqual(mdr_join_columns(["ab", "c"], ["x", "y"], 3), ["ab   x", "c    y"])

    def test_shorter_left_block_gets_blank_cells(self):
        self.assertEqual(mdr_join_columns(["a"], ["x", "y"], 2), ["a   x", "    y"])


if __name__ == "__main__":
    unittest.main()
