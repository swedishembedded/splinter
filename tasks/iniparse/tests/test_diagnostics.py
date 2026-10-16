# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
import unittest

from iniparse.diagnostics import (
    IniDuplicateError, IniEncodeError, IniError, IniInterpolationError, IniLookupError,
    IniMergeError, IniSchemaError, IniSyntaxError, ini_describe_error, ini_error_report,
    ini_excerpt_with_caret, ini_line_column, ini_summarize_errors,
)


class C1(unittest.TestCase):
    def test_message_alone_without_a_position(self):
        self.assertEqual(ini_describe_error("bad"), "bad")

    def test_line_only_when_column_is_zero(self):
        self.assertEqual(ini_describe_error("bad", 4), "line 4: bad")


class C2(unittest.TestCase):
    def test_error_keeps_message_line_and_column(self):
        error = IniError("oops", 3, 7)
        self.assertEqual((error.message, error.line, error.column), ("oops", 3, 7))


    def test_schema_error_message_lists_each_problem(self):
        self.assertEqual(IniSchemaError(["a: bad", "b: worse"]).message, "a: bad\nb: worse")

    def test_schema_error_message_truncates_after_ten(self):
        error = IniSchemaError([f"p{n}" for n in range(11)])
        self.assertEqual(error.message.splitlines()[-1], "(and 1 more)")
        self.assertEqual(len(error.message.splitlines()), 11)

    def test_schema_error_shows_all_ten_problems(self):
        error = IniSchemaError([f"p{n}" for n in range(10)])
        self.assertEqual(len(error.message.splitlines()), 10)


class C3(unittest.TestCase):
    def test_offset_zero_is_line_one_column_one(self):
        self.assertEqual(ini_line_column("abc", 0), (1, 1))


    def test_offset_at_start_of_second_line(self):
        self.assertEqual(ini_line_column("abc\ndef", 4), (2, 1))


    def test_offset_at_end_of_text_is_allowed(self):
        self.assertEqual(ini_line_column("abc\ndef", 7), (2, 4))


    def test_newline_as_first_character_is_seen(self):
        self.assertEqual(ini_line_column("\nabc", 2), (2, 2))

    def test_third_line_is_counted(self):
        self.assertEqual(ini_line_column("a\nb\ncd", 5), (3, 2))


class C4(unittest.TestCase):
    def test_caret_points_below_the_column(self):
        self.assertEqual(ini_excerpt_with_caret("a = 1\nb = x", 2, 5), "b = x\n    ^")

    def test_caret_in_first_column_has_no_padding(self):
        self.assertEqual(ini_excerpt_with_caret("abc", 1, 1), "abc\n^")

    def test_zero_column_gives_the_bare_line(self):
        self.assertEqual(ini_excerpt_with_caret("abc\ndef", 2, 0), "def")

    def test_line_zero_gives_nothing(self):
        self.assertEqual(ini_excerpt_with_caret("abc", 0, 1), "")


    def test_last_line_is_reachable(self):
        self.assertEqual(ini_excerpt_with_caret("abc\ndef", 2, 2), "def\n ^")

    def test_trailing_carriage_return_is_trimmed_from_excerpt(self):
        self.assertEqual(ini_excerpt_with_caret("abc\r\ndef", 1, 2), "abc\n ^")


class C5(unittest.TestCase):
    def test_report_appends_the_excerpt(self):
        error = IniError("bad", 1, 3)
        self.assertEqual(ini_error_report(error, "a = ?"), "line 1, column 3: bad\na = ?\n  ^")

    def test_report_without_position_is_the_message(self):
        self.assertEqual(ini_error_report(IniError("bad"), "a = ?"), "bad")


class C6(unittest.TestCase):

    def test_up_to_limit_lists_every_message(self):
        self.assertEqual(ini_summarize_errors(["a", "b", "c"]), "a\nb\nc")

    def test_beyond_limit_counts_the_hidden_ones(self):
        self.assertEqual(ini_summarize_errors(["a", "b", "c", "d", "e"]), "a\nb\nc\n(and 2 more)")


    def test_exceptions_are_rendered_with_their_text(self):
        self.assertEqual(ini_summarize_errors([IniError("x", 2)]), "line 2: x")


if __name__ == "__main__":
    unittest.main()
