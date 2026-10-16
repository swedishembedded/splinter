# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
import unittest

from iniparse.diagnostics import IniEncodeError, IniSyntaxError
from iniparse.scalars import (
    ini_decode_escape, ini_decode_number, ini_decode_scalar, ini_decode_string,
    ini_escape_string, ini_quote_string, ini_value_kind,
)


def failure(call, *args):
    try:
        call(*args)
    except IniSyntaxError as error:
        return str(error)
    raise AssertionError("no error raised")


class C1(unittest.TestCase):


    def test_zero_alone_is_valid(self):
        self.assertEqual(ini_decode_number("0"), 0)


    def test_octal_numbers_start_with_zero_o(self):
        self.assertEqual(ini_decode_number("0o17"), 15)

    def test_binary_numbers_start_with_zero_b(self):
        self.assertEqual(ini_decode_number("0b101"), 5)

    def test_prefixed_digits_allow_underscores(self):
        self.assertEqual(ini_decode_number("0xde_ad"), 0xdead)


class C2(unittest.TestCase):
    def test_backslash_n_is_a_line_feed(self):
        self.assertEqual(ini_decode_escape("\\n", 0), ("\n", 2))

    def test_tab_and_return_escapes(self):
        self.assertEqual(ini_decode_escape("\\t", 0)[0] + ini_decode_escape("\\r", 0)[0], "\t\r")


    def test_two_digit_hex_escape(self):
        self.assertEqual(ini_decode_escape("\\x41", 0), ("A", 4))


    def test_hex_escape_accepts_capitals(self):
        self.assertEqual(ini_decode_escape("\\x4F", 0), ("O", 4))

    def test_short_hex_escape_is_refused(self):
        self.assertEqual(failure(ini_decode_escape, "\\x4", 0, 2, 10), "line 2, column 10: bad \\x escape")

    def test_non_hex_digits_are_refused(self):
        self.assertEqual(failure(ini_decode_escape, "\\u12g4", 0), "bad \\u escape")

    def test_plus_sign_is_not_a_hex_digit(self):
        self.assertEqual(failure(ini_decode_escape, "\\x+1", 0), "bad \\x escape")

    def test_surrogate_escape_is_refused(self):
        self.assertEqual(failure(ini_decode_escape, "\\ud800", 0), "surrogate in \\u escape")


    def test_trailing_backslash_is_incomplete(self):
        self.assertEqual(failure(ini_decode_escape, "ab\\", 2, 1, 3), "line 1, column 5: incomplete escape sequence")


class C3(unittest.TestCase):
    def test_bad_hex_escape_column_follows_its_index(self):
        self.assertEqual(failure(ini_decode_escape, "ab\\x4", 2, 1, 10), "line 1, column 12: bad \\x escape")

    def test_surrogate_escape_column_follows_its_index(self):
        self.assertEqual(failure(ini_decode_escape, "ab\\ud800", 2, 1, 10),
                         "line 1, column 12: surrogate in \\u escape")

    def test_unknown_escape_column_follows_its_index(self):
        self.assertEqual(failure(ini_decode_escape, "ab\\q", 2, 1, 10), "line 1, column 12: unknown escape '\\q'")

    def test_scalar_failure_omits_location_when_not_given(self):
        self.assertEqual(failure(ini_decode_scalar, "?"), "unrecognized value '?'")


class C4(unittest.TestCase):


    def test_single_quoted_string_is_literal(self):
        self.assertEqual(ini_decode_string("'a\\nb'"), "a\\nb")

    def test_empty_single_quoted_string(self):
        self.assertEqual(ini_decode_string("''"), "")


    def test_lone_double_quote_is_unterminated(self):
        self.assertEqual(failure(ini_decode_string, '"'), "unterminated string")

    def test_lone_single_quote_is_unterminated(self):
        self.assertEqual(failure(ini_decode_string, "'", 1, 1), "line 1, column 1: unterminated string")

    def test_single_quote_without_closing_quote(self):
        self.assertEqual(failure(ini_decode_string, "'abc", 1, 4), "line 1, column 4: unterminated string")

    def test_text_after_double_quoted_string(self):
        self.assertEqual(failure(ini_decode_string, '"a" b', 1, 5),
                         "line 1, column 8: unexpected text after string")

    def test_text_after_single_quoted_string(self):
        self.assertEqual(failure(ini_decode_string, "'a' b", 1, 5),
                         "line 1, column 8: unexpected text after string")

    def test_escaped_closing_quote_leaves_string_open(self):
        self.assertEqual(failure(ini_decode_string, '"a\\"', 1, 1), "line 1, column 1: unterminated string")

    def test_bad_escape_reports_its_column(self):
        self.assertEqual(failure(ini_decode_string, '"a\\qb"', 1, 10), "line 1, column 12: unknown escape '\\q'")


class C5(unittest.TestCase):
    def test_true_is_a_boolean(self):
        self.assertIs(ini_decode_scalar("true"), True)

    def test_false_is_a_boolean(self):
        self.assertIs(ini_decode_scalar("false"), False)


    def test_fractional_text_is_a_float(self):
        self.assertEqual(ini_decode_scalar("0.5"), 0.5)


    def test_bare_word_is_unrecognized(self):
        self.assertEqual(failure(ini_decode_scalar, "bare"), "unrecognized value 'bare'")

    def test_empty_text_is_unrecognized(self):
        self.assertEqual(failure(ini_decode_scalar, ""), "unrecognized value ''")


class C6(unittest.TestCase):


    def test_other_controls_use_hex(self):
        self.assertEqual(ini_escape_string("\x00\x1f"), "\\x00\\x1f")


    def test_quoted_text_decodes_back(self):
        text = 'q"\\\x01\t end'
        self.assertEqual(ini_decode_string(ini_quote_string(text)), text)


class C7(unittest.TestCase):


    def test_unsupported_type_is_rejected(self):
        with self.assertRaises(IniEncodeError) as caught:
            ini_value_kind(None)
        self.assertEqual(str(caught.exception), "unsupported value of type NoneType")


if __name__ == "__main__":
    unittest.main()
