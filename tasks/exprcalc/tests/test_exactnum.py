# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
import unittest
from fractions import Fraction

from exprcalc.errors import XprEvalError
from exprcalc.exactnum import (
    xpr_divide_exact,
    xpr_exact_square_root,
    xpr_is_terminating,
    xpr_normalize_number,
    xpr_number_from_text,
    xpr_number_to_text,
    xpr_power_exact,
    xpr_remainder_exact,
)

F = Fraction


class NormalizeTests(unittest.TestCase):
    def test_whole_fraction_becomes_int(self):
        result = xpr_normalize_number(F(6, 3))
        self.assertEqual(result, 2)
        self.assertIs(type(result), int)

    def test_proper_fraction_stays_fraction(self):
        result = xpr_normalize_number(F(1, 2))
        self.assertIs(type(result), Fraction)

    def test_plain_int_passes_through(self):
        self.assertEqual(xpr_normalize_number(5), 5)

    def test_zero_fraction_becomes_zero_int(self):
        self.assertIs(type(xpr_normalize_number(F(0))), int)


class FromTextTests(unittest.TestCase):
    def test_integer_text_gives_int(self):
        self.assertIs(type(xpr_number_from_text("42")), int)

    def test_decimal_text_gives_exact_fraction(self):
        self.assertEqual(xpr_number_from_text("2.5"), F(5, 2))

    def test_decimal_tenth_times_ten_is_exactly_one(self):
        self.assertEqual(xpr_number_from_text("0.1") * 10, 1)

    def test_exponent_text_scales_up(self):
        self.assertEqual(xpr_number_from_text("1e3"), 1000)

    def test_negative_exponent_scales_down(self):
        self.assertEqual(xpr_number_from_text("25E-2"), F(1, 4))

    def test_leading_dot_text_is_accepted(self):
        self.assertEqual(xpr_number_from_text(".75"), F(3, 4))

    def test_whole_decimal_text_gives_int(self):
        self.assertIs(type(xpr_number_from_text("3.0")), int)


class TerminatingTests(unittest.TestCase):
    def test_whole_numbers_have_finite_expansion(self):
        self.assertTrue(xpr_is_terminating(7))

    def test_fractions_over_two_end_in_decimals(self):
        self.assertTrue(xpr_is_terminating(F(1, 2)))

    def test_fractions_over_five_end_in_decimals(self):
        self.assertTrue(xpr_is_terminating(F(3, 5)))

    def test_mixed_two_and_five_terminate(self):
        self.assertTrue(xpr_is_terminating(F(1, 80)))

    def test_thirds_do_not_terminate(self):
        self.assertFalse(xpr_is_terminating(F(1, 3)))

    def test_sevenths_do_not_terminate(self):
        self.assertFalse(xpr_is_terminating(F(2, 7)))

    def test_factor_three_spoils_a_power_of_two(self):
        self.assertFalse(xpr_is_terminating(F(1, 6)))


class NumberToTextTests(unittest.TestCase):
    def test_whole_number_prints_without_decimal_point(self):
        self.assertEqual(xpr_number_to_text(120), "120")

    def test_negative_integer_keeps_sign(self):
        self.assertEqual(xpr_number_to_text(-4), "-4")

    def test_half_prints_with_leading_zero(self):
        self.assertEqual(xpr_number_to_text(F(1, 2)), "0.5")

    def test_small_fraction_keeps_inner_zeros(self):
        self.assertEqual(xpr_number_to_text(F(1, 8)), "0.125")

    def test_mixed_number_prints_whole_part(self):
        self.assertEqual(xpr_number_to_text(F(11, 4)), "2.75")

    def test_negative_fraction_has_one_sign(self):
        self.assertEqual(xpr_number_to_text(F(-5, 4)), "-1.25")

    def test_hundredth_keeps_leading_fraction_zero(self):
        self.assertEqual(xpr_number_to_text(F(1, 100)), "0.01")

    def test_endless_expansion_is_refused(self):
        with self.assertRaisesRegex(ValueError, "^number has no finite decimal form$"):
            xpr_number_to_text(F(1, 3))


class DivideTests(unittest.TestCase):
    def test_exact_division_gives_int(self):
        result = xpr_divide_exact(6, 3)
        self.assertEqual(result, 2)
        self.assertIs(type(result), int)

    def test_inexact_division_gives_fraction(self):
        self.assertEqual(xpr_divide_exact(1, 3), F(1, 3))

    def test_half_over_quarter_equals_two(self):
        self.assertEqual(xpr_divide_exact(F(1, 2), F(1, 4)), 2)

    def test_negative_divisor_flips_sign(self):
        self.assertEqual(xpr_divide_exact(3, -4), F(-3, 4))

    def test_zero_divisor_is_an_error(self):
        with self.assertRaisesRegex(XprEvalError, "^division by zero$"):
            xpr_divide_exact(1, 0)

    def test_zero_numerator_is_fine(self):
        self.assertEqual(xpr_divide_exact(0, 5), 0)


class RemainderTests(unittest.TestCase):
    def test_seven_modulo_three_leaves_one(self):
        self.assertEqual(xpr_remainder_exact(7, 3), 1)

    def test_negative_dividend_follows_divisor_sign(self):
        self.assertEqual(xpr_remainder_exact(-7, 3), 2)

    def test_negative_divisor_gives_negative_remainder(self):
        self.assertEqual(xpr_remainder_exact(7, -3), -2)

    def test_fractional_remainder_is_exact(self):
        self.assertEqual(xpr_remainder_exact(F(15, 2), 2), F(3, 2))

    def test_exact_multiple_leaves_no_remainder_left(self):
        self.assertEqual(xpr_remainder_exact(9, 3), 0)

    def test_modulo_by_zero_is_an_error(self):
        with self.assertRaisesRegex(XprEvalError, "^modulo by zero$"):
            xpr_remainder_exact(5, 0)


class PowerTests(unittest.TestCase):
    def test_two_to_the_tenth_is_1024(self):
        self.assertEqual(xpr_power_exact(2, 10), 1024)

    def test_zero_exponent_gives_one(self):
        self.assertEqual(xpr_power_exact(9, 0), 1)

    def test_negative_exponent_gives_reciprocal(self):
        self.assertEqual(xpr_power_exact(2, -2), F(1, 4))

    def test_negative_base_alternates_sign(self):
        self.assertEqual(xpr_power_exact(-2, 3), -8)

    def test_fraction_base_is_exact(self):
        self.assertEqual(xpr_power_exact(F(1, 2), 3), F(1, 8))

    def test_fractional_exponent_is_refused(self):
        with self.assertRaisesRegex(XprEvalError, "^exponent must be a whole number$"):
            xpr_power_exact(4, F(1, 2))

    def test_whole_fraction_exponent_is_accepted(self):
        self.assertEqual(xpr_power_exact(3, F(4, 2)), 9)

    def test_largest_exponent_is_allowed(self):
        self.assertEqual(xpr_power_exact(1, 200), 1)
        self.assertEqual(xpr_power_exact(2, 200), 2 ** 200)

    def test_exponent_above_limit_is_refused(self):
        with self.assertRaisesRegex(XprEvalError, "^exponent out of range$"):
            xpr_power_exact(1, 201)

    def test_exponent_below_limit_is_refused(self):
        self.assertEqual(xpr_power_exact(1, -200), 1)
        with self.assertRaisesRegex(XprEvalError, "^exponent out of range$"):
            xpr_power_exact(1, -201)

    def test_zero_to_negative_power_is_refused(self):
        with self.assertRaisesRegex(XprEvalError, "^zero cannot be raised to a negative power$"):
            xpr_power_exact(0, -1)

    def test_zero_to_zero_is_one(self):
        self.assertEqual(xpr_power_exact(0, 0), 1)


class SquareRootTests(unittest.TestCase):
    def test_forty_nine_has_root_seven(self):
        self.assertEqual(xpr_exact_square_root(49), 7)

    def test_zero_has_root_zero(self):
        self.assertEqual(xpr_exact_square_root(0), 0)

    def test_fraction_with_square_parts(self):
        self.assertEqual(xpr_exact_square_root(F(9, 4)), F(3, 2))

    def test_non_square_integer_is_refused(self):
        with self.assertRaisesRegex(XprEvalError, "^square root is not exact$"):
            xpr_exact_square_root(2)

    def test_non_square_denominator_is_refused(self):
        with self.assertRaisesRegex(XprEvalError, "^square root is not exact$"):
            xpr_exact_square_root(F(4, 3))

    def test_negative_input_is_refused(self):
        with self.assertRaisesRegex(XprEvalError, "^square root of a negative number$"):
            xpr_exact_square_root(-4)

    def test_one_has_root_one(self):
        self.assertEqual(xpr_exact_square_root(1), 1)
