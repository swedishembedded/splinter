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


class NumberTests(unittest.TestCase):
    def test_plain_integer_is_decoded(self):
        self.assertEqual(ini_decode_number("42"), 42)

    def test_signed_integers_are_decoded(self):
        self.assertEqual((ini_decode_number("+7"), ini_decode_number("-7")), (7, -7))

    def test_zero_alone_is_valid(self):
        self.assertEqual(ini_decode_number("0"), 0)

    def test_leading_zero_is_refused(self):
        self.assertIsNone(ini_decode_number("007"))

    def test_underscores_may_group_digits(self):
        self.assertEqual(ini_decode_number("1_000_000"), 1000000)

    def test_trailing_underscore_is_refused(self):
        self.assertIsNone(ini_decode_number("1_"))

    def test_doubled_underscore_is_refused(self):
        self.assertIsNone(ini_decode_number("1__0"))

    def test_hexadecimal_numbers_start_with_zero_x(self):
        self.assertEqual(ini_decode_number("0xff"), 255)

    def test_octal_numbers_start_with_zero_o(self):
        self.assertEqual(ini_decode_number("0o17"), 15)

    def test_binary_numbers_start_with_zero_b(self):
        self.assertEqual(ini_decode_number("0b101"), 5)

    def test_prefixed_digits_allow_underscores(self):
        self.assertEqual(ini_decode_number("0xde_ad"), 0xdead)

    def test_octal_rejects_digit_eight(self):
        self.assertIsNone(ini_decode_number("0o8"))

    def test_binary_rejects_digit_two(self):
        self.assertIsNone(ini_decode_number("0b2"))

    def test_signed_prefixed_number_is_refused(self):
        self.assertIsNone(ini_decode_number("-0x1"))

    def test_decimal_point_makes_a_float(self):
        self.assertEqual(ini_decode_number("2.5"), 2.5)

    def test_float_with_exponent_only(self):
        self.assertEqual(ini_decode_number("1e3"), 1000.0)

    def test_float_with_negative_exponent(self):
        self.assertEqual(ini_decode_number("5E-1"), 0.5)

    def test_float_with_signed_fraction_and_exponent(self):
        self.assertEqual(ini_decode_number("-1.25e+2"), -125.0)

    def test_float_result_is_a_float_type(self):
        self.assertIsInstance(ini_decode_number("1.0"), float)

    def test_float_needs_digits_after_the_dot(self):
        self.assertIsNone(ini_decode_number("1."))

    def test_float_needs_digits_before_the_dot(self):
        self.assertIsNone(ini_decode_number(".5"))

    def test_float_underscores_in_integer_part(self):
        self.assertEqual(ini_decode_number("1_0.5"), 10.5)

    def test_words_are_not_numbers(self):
        self.assertIsNone(ini_decode_number("abc"))


class EscapeTests(unittest.TestCase):
    def test_backslash_n_is_a_line_feed(self):
        self.assertEqual(ini_decode_escape("\\n", 0), ("\n", 2))

    def test_tab_and_return_escapes(self):
        self.assertEqual(ini_decode_escape("\\t", 0)[0] + ini_decode_escape("\\r", 0)[0], "\t\r")

    def test_quote_and_backslash_escapes(self):
        self.assertEqual(ini_decode_escape('\\"', 0)[0] + ini_decode_escape("\\\\", 0)[0], '"\\')

    def test_escape_read_from_inner_index(self):
        self.assertEqual(ini_decode_escape("ab\\tc", 2), ("\t", 4))

    def test_two_digit_hex_escape(self):
        self.assertEqual(ini_decode_escape("\\x41", 0), ("A", 4))

    def test_four_digit_unicode_escape(self):
        self.assertEqual(ini_decode_escape("\\u00e9", 0), ("\u00e9", 6))

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

    def test_last_unicode_below_surrogates_is_fine(self):
        self.assertEqual(ini_decode_escape("\\ud7ff", 0)[0], "\ud7ff")

    def test_first_unicode_above_surrogates_is_fine(self):
        self.assertEqual(ini_decode_escape("\\ue000", 0)[0], "\ue000")

    def test_unknown_escape_letter_is_refused(self):
        self.assertEqual(failure(ini_decode_escape, "\\q", 0, 1, 5), "line 1, column 5: unknown escape '\\q'")

    def test_trailing_backslash_is_incomplete(self):
        self.assertEqual(failure(ini_decode_escape, "ab\\", 2, 1, 3), "line 1, column 5: incomplete escape sequence")


class EscapePositionTests(unittest.TestCase):
    def test_bad_hex_escape_column_follows_its_index(self):
        self.assertEqual(failure(ini_decode_escape, "ab\\x4", 2, 1, 10), "line 1, column 12: bad \\x escape")

    def test_surrogate_escape_column_follows_its_index(self):
        self.assertEqual(failure(ini_decode_escape, "ab\\ud800", 2, 1, 10),
                         "line 1, column 12: surrogate in \\u escape")

    def test_unknown_escape_column_follows_its_index(self):
        self.assertEqual(failure(ini_decode_escape, "ab\\q", 2, 1, 10), "line 1, column 12: unknown escape '\\q'")

    def test_scalar_failure_omits_location_when_not_given(self):
        self.assertEqual(failure(ini_decode_scalar, "?"), "unrecognized value '?'")


class StringTests(unittest.TestCase):
    def test_plain_double_quoted_string(self):
        self.assertEqual(ini_decode_string('"hello"'), "hello")

    def test_empty_double_quoted_string(self):
        self.assertEqual(ini_decode_string('""'), "")

    def test_escapes_are_applied_in_double_quotes(self):
        self.assertEqual(ini_decode_string('"a\\tb\\n"'), "a\tb\n")

    def test_escaped_quote_stays_inside(self):
        self.assertEqual(ini_decode_string('"say \\"hi\\""'), 'say "hi"')

    def test_single_quoted_string_is_literal(self):
        self.assertEqual(ini_decode_string("'a\\nb'"), "a\\nb")

    def test_empty_single_quoted_string(self):
        self.assertEqual(ini_decode_string("''"), "")

    def test_unterminated_double_quote_reports_start(self):
        self.assertEqual(failure(ini_decode_string, '"abc', 3, 7), "line 3, column 7: unterminated string")

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

    def test_non_ascii_text_is_kept(self):
        self.assertEqual(ini_decode_string('"caf\u00e9"'), "caf\u00e9")


class ScalarTests(unittest.TestCase):
    def test_true_is_a_boolean(self):
        self.assertIs(ini_decode_scalar("true"), True)

    def test_false_is_a_boolean(self):
        self.assertIs(ini_decode_scalar("false"), False)

    def test_capitalised_boolean_is_unrecognized(self):
        self.assertEqual(failure(ini_decode_scalar, "True", 2, 3), "line 2, column 3: unrecognized value 'True'")

    def test_whole_number_text_is_an_integer(self):
        self.assertEqual(ini_decode_scalar("-12"), -12)

    def test_fractional_text_is_a_float(self):
        self.assertEqual(ini_decode_scalar("0.5"), 0.5)

    def test_double_quotes_make_a_string(self):
        self.assertEqual(ini_decode_scalar('"x y"'), "x y")

    def test_single_quotes_make_a_literal_string(self):
        self.assertEqual(ini_decode_scalar("'x y'"), "x y")

    def test_bare_word_is_unrecognized(self):
        self.assertEqual(failure(ini_decode_scalar, "bare"), "unrecognized value 'bare'")

    def test_empty_text_is_unrecognized(self):
        self.assertEqual(failure(ini_decode_scalar, ""), "unrecognized value ''")

    def test_string_errors_carry_the_given_position(self):
        self.assertEqual(failure(ini_decode_scalar, '"abc', 5, 9), "line 5, column 9: unterminated string")


class QuoteTests(unittest.TestCase):
    def test_plain_text_is_unchanged(self):
        self.assertEqual(ini_escape_string("abc d"), "abc d")

    def test_quote_and_backslash_are_escaped(self):
        self.assertEqual(ini_escape_string('a"b\\c'), 'a\\"b\\\\c')

    def test_whitespace_controls_use_letters(self):
        self.assertEqual(ini_escape_string("\n\t\r"), "\\n\\t\\r")

    def test_other_controls_use_hex(self):
        self.assertEqual(ini_escape_string("\x00\x1f"), "\\x00\\x1f")

    def test_space_is_not_a_control(self):
        self.assertEqual(ini_escape_string(" "), " ")

    def test_non_ascii_passes_through(self):
        self.assertEqual(ini_escape_string("\u00e9"), "\u00e9")

    def test_quote_wraps_in_double_quotes(self):
        self.assertEqual(ini_quote_string("a\nb"), '"a\\nb"')

    def test_quoted_text_decodes_back(self):
        text = 'q"\\\x01\t end'
        self.assertEqual(ini_decode_string(ini_quote_string(text)), text)


class ValueKindTests(unittest.TestCase):
    def test_bool_is_not_an_integer(self):
        self.assertEqual(ini_value_kind(True), "boolean")

    def test_whole_numbers_are_classified_integer(self):
        self.assertEqual(ini_value_kind(3), "integer")

    def test_fractions_are_classified_float(self):
        self.assertEqual(ini_value_kind(3.0), "float")

    def test_text_is_classified_string(self):
        self.assertEqual(ini_value_kind("x"), "string")

    def test_lists_are_classified_array(self):
        self.assertEqual(ini_value_kind([1]), "array")

    def test_dicts_are_classified_table(self):
        self.assertEqual(ini_value_kind({}), "table")

    def test_unsupported_type_is_rejected(self):
        with self.assertRaises(IniEncodeError) as caught:
            ini_value_kind(None)
        self.assertEqual(str(caught.exception), "unsupported value of type NoneType")


if __name__ == "__main__":
    unittest.main()
