# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
import unittest

from mdrender.wrapping import (
    mdr_break_long_word,
    mdr_count_words,
    mdr_fill_paragraphs,
    mdr_longest_line_length,
    mdr_split_hyphens,
    mdr_truncate_text,
    mdr_unwrap_lines,
    mdr_wrap_text,
)


class C0(unittest.TestCase):
    def test_several_hyphens_split_into_several_pieces(self):
        self.assertEqual(mdr_split_hyphens("a-b-c"), ["a-", "b-", "c"])

    def test_leading_hyphen_does_not_split(self):
        self.assertEqual(mdr_split_hyphens("-ab"), ["-ab"])

    def test_double_hyphen_does_not_split(self):
        self.assertEqual(mdr_split_hyphens("a--b"), ["a--b"])

    def test_hyphen_between_digits_allows_a_break(self):
        self.assertEqual(mdr_split_hyphens("1-2"), ["1-", "2"])


class C1(unittest.TestCase):
    def test_word_one_over_the_width_is_cut(self):
        self.assertEqual(mdr_break_long_word("abcd", 3), ["abc", "d"])

    def test_hyphenation_keeps_a_word_that_fits(self):
        self.assertEqual(mdr_break_long_word("abcd", 4, hyphenate=True), ["abcd"])

    def test_first_chunk_may_use_its_own_width(self):
        self.assertEqual(mdr_break_long_word("abcdefg", 3, first_width=2), ["ab", "cde", "fg"])

    def test_hyphenation_needs_two_columns(self):
        with self.assertRaises(ValueError) as caught:
            mdr_break_long_word("abc", 1, hyphenate=True)
        self.assertEqual(str(caught.exception), "hyphenation needs a width of at least 2")

    def test_width_two_hyphenation_advances_one_letter(self):
        self.assertEqual(mdr_break_long_word("abc", 2, hyphenate=True), ["a-", "bc"])


class C2(unittest.TestCase):
    def test_long_word_tail_accepts_more_words(self):
        self.assertEqual(mdr_wrap_text("abcde f", 4), ["abcd", "e f"])

    def test_long_word_after_text_moves_down(self):
        self.assertEqual(mdr_wrap_text("x abcdef", 4), ["x", "abcd", "ef"])

    def test_long_word_whole_without_breaking(self):
        self.assertEqual(mdr_wrap_text("abcdefgh ij", 3, break_long=False), ["abcdefgh", "ij"])

    def test_hyphenate_marks_every_cut_made(self):
        self.assertEqual(mdr_wrap_text("abcdefgh", 4, hyphenate=True), ["abc-", "def-", "gh"])

    def test_wrap_breaks_after_an_inner_hyphen(self):
        self.assertEqual(mdr_wrap_text("a well-known fact", 9), ["a well-", "known", "fact"])

    def test_hyphen_pieces_rejoin_without_a_space(self):
        self.assertEqual(mdr_wrap_text("a well-known", 12), ["a well-known"])

    def test_full_first_indent_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            mdr_wrap_text("a", 2, first_indent="  ")
        self.assertEqual(str(caught.exception), "indent leaves no room for text")

    def test_rest_indent_one_short_accepted(self):
        self.assertEqual(mdr_wrap_text("a b c", 3, rest_indent="  "), ["a b", "  c"])

    def test_long_word_on_indented_first_line(self):
        self.assertEqual(mdr_wrap_text("abcdefg", 4, first_indent="> "), ["> ab", "cdef", "g"])

    def test_long_word_after_flush_uses_rest(self):
        self.assertEqual(mdr_wrap_text("x abcdefg", 4, rest_indent="  "),
                         ["x", "  ab", "  cd", "  ef", "  g"])


class C3(unittest.TestCase):
    def test_paragraphs_stay_separated_by_one_blank_line(self):
        self.assertEqual(mdr_fill_paragraphs("aa bb\n\n\n\ncc", 5), "aa bb\n\ncc")

    def test_lines_of_a_paragraph_are_refilled(self):
        self.assertEqual(mdr_fill_paragraphs("aa\nbb\ncc", 8), "aa bb cc")

    def test_indents_apply_to_every_paragraph(self):
        self.assertEqual(mdr_fill_paragraphs("aa bb\n\ncc dd", 5, "* ", "  "), "* aa\n  bb\n\n* cc\n  dd")


class C4(unittest.TestCase):
    def test_unwrap_joins_lines_with_single_spaces(self):
        self.assertEqual(mdr_unwrap_lines(["  a ", "b", "", "  ", "c"]), "a b c")

    def test_truncate_keeps_text_that_fits(self):
        self.assertEqual(mdr_truncate_text("abcde", 5), "abcde")

    def test_truncate_width_equal_to_marker_ok(self):
        self.assertEqual(mdr_truncate_text("abcd", 3), "...")

    def test_truncate_rejects_a_width_below_the_marker(self):
        with self.assertRaises(ValueError) as caught:
            mdr_truncate_text("abcd", 2)
        self.assertEqual(str(caught.exception), "width is smaller than the marker")

    def test_truncate_uses_a_custom_marker(self):
        self.assertEqual(mdr_truncate_text("abcdef", 4, marker="~"), "abc~")

    def test_word_count_ignores_extra_whitespace(self):
        self.assertEqual(mdr_count_words(" a  b\nc "), 3)

    def test_longest_line_is_measured_in_characters(self):
        self.assertEqual(mdr_longest_line_length(["ab", "abcd", "abc"]), 4)

    def test_longest_line_of_nothing_is_zero(self):
        self.assertEqual(mdr_longest_line_length([]), 0)


if __name__ == "__main__":
    unittest.main()
