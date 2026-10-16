# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
import unittest
from fractions import Fraction

from exprcalc.errors import XprEvalError
from exprcalc.operators import (
    XPR_LOWEST_PRECEDENCE,
    XPR_UNARY_PRECEDENCE,
    xpr_apply_binary,
    xpr_apply_unary,
    xpr_binary_precedence,
    xpr_is_binary_operator,
    xpr_is_right_associative,
    xpr_is_unary_operator,
)

F = Fraction


class PrecedenceTests(unittest.TestCase):
    def test_disjunction_has_the_lowest_level(self):
        self.assertEqual(xpr_binary_precedence("||"), 1)

    def test_and_binds_tighter_than_or(self):
        self.assertEqual(xpr_binary_precedence("&&"), 2)

    def test_equality_binds_tighter_than_and(self):
        self.assertEqual(xpr_binary_precedence("=="), 3)
        self.assertEqual(xpr_binary_precedence("!="), 3)

    def test_ordering_binds_tighter_than_equality(self):
        for op in ("<", "<=", ">", ">="):
            self.assertEqual(xpr_binary_precedence(op), 4)

    def test_sum_and_difference_share_level_five(self):
        self.assertEqual(xpr_binary_precedence("+"), 5)
        self.assertEqual(xpr_binary_precedence("-"), 5)

    def test_product_quotient_remainder_share_level_six(self):
        for op in ("*", "/", "%"):
            self.assertEqual(xpr_binary_precedence(op), 6)

    def test_caret_has_the_highest_level(self):
        self.assertEqual(xpr_binary_precedence("^"), 8)

    def test_unary_sits_between_product_and_power(self):
        self.assertEqual(XPR_UNARY_PRECEDENCE, 7)

    def test_lowest_precedence_is_the_disjunction_level(self):
        self.assertEqual(XPR_LOWEST_PRECEDENCE, 1)

    def test_unknown_operator_has_no_precedence(self):
        with self.assertRaises(KeyError):
            xpr_binary_precedence("~")


class ClassificationTests(unittest.TestCase):
    def test_binary_operators_are_recognised(self):
        for op in ("||", "&&", "==", "!=", "<", "<=", ">", ">=", "+", "-", "*", "/", "%", "^"):
            self.assertTrue(xpr_is_binary_operator(op), op)

    def test_bang_is_not_binary(self):
        self.assertFalse(xpr_is_binary_operator("!"))

    def test_unary_operators_are_recognised(self):
        for op in ("-", "+", "!"):
            self.assertTrue(xpr_is_unary_operator(op), op)

    def test_star_is_not_unary(self):
        self.assertFalse(xpr_is_unary_operator("*"))

    def test_only_power_is_right_associative(self):
        self.assertTrue(xpr_is_right_associative("^"))
        for op in ("+", "-", "*", "/", "%", "&&", "||", "<", "=="):
            self.assertFalse(xpr_is_right_associative(op), op)


class UnaryApplyTests(unittest.TestCase):
    def test_minus_flips_an_integer_sign(self):
        self.assertEqual(xpr_apply_unary("-", 5), -5)

    def test_minus_of_negative_fraction_is_positive(self):
        self.assertEqual(xpr_apply_unary("-", -F(1, 2)), F(1, 2))

    def test_unary_plus_returns_operand_unchanged(self):
        self.assertEqual(xpr_apply_unary("+", -3), -3)

    def test_not_of_zero_is_one(self):
        self.assertEqual(xpr_apply_unary("!", 0), 1)

    def test_not_of_nonzero_is_zero(self):
        self.assertEqual(xpr_apply_unary("!", 7), 0)
        self.assertEqual(xpr_apply_unary("!", F(1, 3)), 0)

    def test_unknown_unary_operator_is_refused(self):
        with self.assertRaisesRegex(XprEvalError, "^unknown unary operator '~'$"):
            xpr_apply_unary("~", 1)


class ArithmeticApplyTests(unittest.TestCase):
    def test_plus_adds_two_integers(self):
        self.assertEqual(xpr_apply_binary("+", 2, 3), 5)

    def test_minus_keeps_operand_order(self):
        self.assertEqual(xpr_apply_binary("-", 2, 3), -1)

    def test_star_multiplies_with_sign(self):
        self.assertEqual(xpr_apply_binary("*", 4, -3), -12)

    def test_slash_keeps_quarter_as_fraction(self):
        self.assertEqual(xpr_apply_binary("/", 1, 4), F(1, 4))

    def test_division_by_zero_fails(self):
        with self.assertRaisesRegex(XprEvalError, "^division by zero$"):
            xpr_apply_binary("/", 1, 0)

    def test_remainder_follows_divisor_sign(self):
        self.assertEqual(xpr_apply_binary("%", -7, 3), 2)

    def test_caret_raises_to_a_whole_power(self):
        self.assertEqual(xpr_apply_binary("^", 2, 8), 256)

    def test_results_are_normalised_to_int(self):
        self.assertIs(type(xpr_apply_binary("+", F(1, 2), F(1, 2))), int)
        self.assertIs(type(xpr_apply_binary("*", F(2, 3), F(3, 2))), int)

    def test_mixed_int_and_fraction(self):
        self.assertEqual(xpr_apply_binary("+", 1, F(1, 2)), F(3, 2))


class ComparisonApplyTests(unittest.TestCase):
    def test_less_than_is_strict_at_equality(self):
        self.assertEqual(xpr_apply_binary("<", 1, 2), 1)
        self.assertEqual(xpr_apply_binary("<", 2, 2), 0)
        self.assertEqual(xpr_apply_binary("<", 3, 2), 0)

    def test_less_or_equal_boundary(self):
        self.assertEqual(xpr_apply_binary("<=", 1, 2), 1)
        self.assertEqual(xpr_apply_binary("<=", 2, 2), 1)
        self.assertEqual(xpr_apply_binary("<=", 3, 2), 0)

    def test_greater_than_is_strict_at_equality(self):
        self.assertEqual(xpr_apply_binary(">", 3, 2), 1)
        self.assertEqual(xpr_apply_binary(">", 2, 2), 0)
        self.assertEqual(xpr_apply_binary(">", 1, 2), 0)

    def test_greater_or_equal_boundary(self):
        self.assertEqual(xpr_apply_binary(">=", 3, 2), 1)
        self.assertEqual(xpr_apply_binary(">=", 2, 2), 1)
        self.assertEqual(xpr_apply_binary(">=", 1, 2), 0)

    def test_equality_of_different_representations(self):
        self.assertEqual(xpr_apply_binary("==", F(4, 2), 2), 1)
        self.assertEqual(xpr_apply_binary("==", 1, 2), 0)

    def test_bang_equals_detects_difference(self):
        self.assertEqual(xpr_apply_binary("!=", 1, 2), 1)
        self.assertEqual(xpr_apply_binary("!=", 2, 2), 0)

    def test_comparison_result_is_a_plain_int(self):
        self.assertIs(type(xpr_apply_binary("<", 1, 2)), int)


class LogicApplyTests(unittest.TestCase):
    def test_conjunction_follows_the_boolean_table(self):
        self.assertEqual(
            [xpr_apply_binary("&&", a, b) for a, b in ((0, 0), (0, 5), (5, 0), (5, 7))],
            [0, 0, 0, 1],
        )

    def test_disjunction_follows_the_boolean_table(self):
        self.assertEqual(
            [xpr_apply_binary("||", a, b) for a, b in ((0, 0), (0, 5), (5, 0), (5, 7))],
            [0, 1, 1, 1],
        )

    def test_logic_treats_fractions_as_true(self):
        self.assertEqual(xpr_apply_binary("&&", F(1, 3), F(1, 7)), 1)

    def test_unknown_binary_operator_is_refused(self):
        with self.assertRaisesRegex(XprEvalError, "^unknown binary operator '~'$"):
            xpr_apply_binary("~", 1, 1)
