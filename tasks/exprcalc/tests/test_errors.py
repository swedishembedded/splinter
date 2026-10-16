# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
import unittest

from exprcalc.errors import (
    XprError,
    XprEvalError,
    XprSyntaxError,
    xpr_attach_position,
    xpr_describe_error,
    xpr_line_and_column,
    xpr_source_line,
)


class ErrorTypeTests(unittest.TestCase):
    def test_text_without_position_is_message(self):
        self.assertEqual(str(XprError("boom")), "boom")

    def test_text_mentions_position_when_known(self):
        self.assertEqual(str(XprError("boom", 3)), "boom at position 3")

    def test_position_zero_still_appears_in_text(self):
        self.assertEqual(str(XprEvalError("bad", 0)), "bad at position 0")

    def test_message_attribute_is_kept(self):
        self.assertEqual(XprSyntaxError("oops", 2).message, "oops")

    def test_position_attribute_is_kept(self):
        self.assertEqual(XprSyntaxError("oops", 2).position, 2)

    def test_missing_position_defaults_to_none(self):
        self.assertIsNone(XprEvalError("oops").position)

    def test_syntax_error_derives_from_base(self):
        self.assertTrue(issubclass(XprSyntaxError, XprError))

    def test_evaluation_error_derives_from_base(self):
        self.assertTrue(issubclass(XprEvalError, XprError))

    def test_two_error_kinds_are_unrelated(self):
        self.assertFalse(issubclass(XprSyntaxError, XprEvalError))
        self.assertFalse(issubclass(XprEvalError, XprSyntaxError))

    def test_base_error_can_be_raised_and_caught(self):
        with self.assertRaises(Exception) as caught:
            raise XprError("late", 1)
        self.assertEqual(caught.exception.args, ("late",))


class LineAndColumnTests(unittest.TestCase):
    def test_origin_is_line_one_column_one(self):
        self.assertEqual(xpr_line_and_column("abc", 0), (1, 1))

    def test_column_counts_from_line_start(self):
        self.assertEqual(xpr_line_and_column("abcdef", 4), (1, 5))

    def test_character_after_newline_opens_next_line(self):
        self.assertEqual(xpr_line_and_column("ab\ncd", 3), (2, 1))

    def test_second_character_of_second_line(self):
        self.assertEqual(xpr_line_and_column("ab\ncd", 4), (2, 2))

    def test_newline_belongs_to_the_line_it_ends(self):
        self.assertEqual(xpr_line_and_column("ab\ncd", 2), (1, 3))

    def test_third_line_is_counted(self):
        self.assertEqual(xpr_line_and_column("a\nb\nc", 4), (3, 1))

    def test_end_of_source_is_a_valid_position(self):
        self.assertEqual(xpr_line_and_column("abc", 3), (1, 4))

    def test_blank_first_line_still_counts(self):
        self.assertEqual(xpr_line_and_column("\nab", 2), (2, 2))
        self.assertEqual(xpr_line_and_column("\nab", 1), (2, 1))

    def test_negative_position_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "^position out of range$"):
            xpr_line_and_column("abc", -1)

    def test_position_beyond_end_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "^position out of range$"):
            xpr_line_and_column("abc", 4)


class SourceLineTests(unittest.TestCase):
    def test_first_line_is_returned_without_newline(self):
        self.assertEqual(xpr_source_line("ab\ncd", 1), "ab")

    def test_last_line_without_terminator_is_whole(self):
        self.assertEqual(xpr_source_line("ab\ncd", 4), "cd")

    def test_position_on_newline_gives_the_ended_line(self):
        self.assertEqual(xpr_source_line("ab\ncd", 2), "ab")

    def test_middle_line_is_isolated(self):
        self.assertEqual(xpr_source_line("a\nbb\nc", 3), "bb")

    def test_blank_first_line_leaves_second_readable(self):
        self.assertEqual(xpr_source_line("\nab", 2), "ab")
        self.assertEqual(xpr_source_line("\nab", 0), "")

    def test_empty_source_has_an_empty_line(self):
        self.assertEqual(xpr_source_line("", 0), "")

    def test_source_line_rejects_bad_position(self):
        with self.assertRaisesRegex(ValueError, "^position out of range$"):
            xpr_source_line("ab", 3)
        with self.assertRaises(ValueError):
            xpr_source_line("ab", -1)


class AttachPositionTests(unittest.TestCase):
    def test_missing_position_is_filled_in(self):
        fixed = xpr_attach_position(XprEvalError("bad"), 7)
        self.assertEqual((fixed.message, fixed.position), ("bad", 7))

    def test_existing_position_is_never_replaced(self):
        original = XprEvalError("bad", 2)
        self.assertIs(xpr_attach_position(original, 9), original)

    def test_copy_keeps_the_error_type(self):
        self.assertIs(type(xpr_attach_position(XprSyntaxError("bad"), 1)), XprSyntaxError)

    def test_position_zero_counts_as_existing(self):
        original = XprEvalError("bad", 0)
        self.assertEqual(xpr_attach_position(original, 5).position, 0)


class DescribeErrorTests(unittest.TestCase):
    def test_caret_sits_under_the_position(self):
        text = xpr_describe_error(XprSyntaxError("bad", 2), "1+*")
        self.assertEqual(text, "XprSyntaxError: bad (line 1, column 3)\n1+*\n  ^")

    def test_second_line_is_quoted_with_its_column(self):
        text = xpr_describe_error(XprEvalError("zero", 5), "1+\n2/0")
        self.assertEqual(text, "XprEvalError: zero (line 2, column 3)\n2/0\n  ^")

    def test_error_without_position_is_a_single_line(self):
        self.assertEqual(xpr_describe_error(XprEvalError("zero"), "1/0"), "XprEvalError: zero")

    def test_first_column_needs_no_indent(self):
        text = xpr_describe_error(XprSyntaxError("bad", 0), "?")
        self.assertEqual(text.splitlines()[-1], "^")
