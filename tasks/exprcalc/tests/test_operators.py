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


class ZA(unittest.TestCase):
    def test_disjunction_has_the_lowest_level(self):
        self.assertEqual(xpr_binary_precedence("||"), 1)
        self.assertEqual(xpr_binary_precedence("&&"), 2)

    def test_equality_binds_tighter_than_and(self):
        self.assertEqual(xpr_binary_precedence("=="), 3)
        self.assertEqual(xpr_binary_precedence("!="), 3)
        for op in ("<", "<=", ">", ">="):
            self.assertEqual(xpr_binary_precedence(op), 4)

    def test_sum_and_difference_share_level_five(self):
        self.assertEqual(xpr_binary_precedence("+"), 5)
        self.assertEqual(xpr_binary_precedence("-"), 5)
        for op in ("*", "/", "%"):
            self.assertEqual(xpr_binary_precedence(op), 6)

    def test_caret_has_the_highest_level(self):
        self.assertEqual(xpr_binary_precedence("^"), 8)
        self.assertEqual(XPR_UNARY_PRECEDENCE, 7)

    def test_lowest_precedence_is_the_disjunction_level(self):
        self.assertEqual(XPR_LOWEST_PRECEDENCE, 1)
        with self.assertRaises(KeyError):
            xpr_binary_precedence("~")


class ZB(unittest.TestCase):
    def test_binary_operators_are_recognised(self):
        for op in ("||", "&&", "==", "!=", "<", "<=", ">", ">=", "+", "-", "*", "/", "%", "^"):
            self.assertTrue(xpr_is_binary_operator(op), op)
        self.assertFalse(xpr_is_binary_operator("!"))

    def test_unary_operators_are_recognised(self):
        for op in ("-", "+", "!"):
            self.assertTrue(xpr_is_unary_operator(op), op)
        self.assertFalse(xpr_is_unary_operator("*"))

    def test_only_power_is_right_associative(self):
        self.assertTrue(xpr_is_right_associative("^"))
        for op in ("+", "-", "*", "/", "%", "&&", "||", "<", "=="):
            self.assertFalse(xpr_is_right_associative(op), op)


class ZC(unittest.TestCase):
    def test_minus_flips_an_integer_sign(self):
        self.assertEqual(xpr_apply_unary("-", 5), -5)
        self.assertEqual(xpr_apply_unary("-", -F(1, 2)), F(1, 2))

    def test_unary_plus_returns_operand_unchanged(self):
        self.assertEqual(xpr_apply_unary("+", -3), -3)
        self.assertEqual(xpr_apply_unary("!", 0), 1)

    def test_not_of_nonzero_is_zero(self):
        self.assertEqual(xpr_apply_unary("!", 7), 0)
        self.assertEqual(xpr_apply_unary("!", F(1, 3)), 0)
        with self.assertRaisesRegex(XprEvalError, "^unknown unary operator '~'$"):
            xpr_apply_unary("~", 1)


class ZD(unittest.TestCase):
    def test_plus_adds_two_integers(self):
        self.assertEqual(xpr_apply_binary("+", 2, 3), 5)
        self.assertEqual(xpr_apply_binary("-", 2, 3), -1)

    def test_star_multiplies_with_sign(self):
        self.assertEqual(xpr_apply_binary("*", 4, -3), -12)
        self.assertEqual(xpr_apply_binary("/", 1, 4), F(1, 4))

    def test_division_by_zero_fails(self):
        with self.assertRaisesRegex(XprEvalError, "^division by zero$"):
            xpr_apply_binary("/", 1, 0)
        self.assertEqual(xpr_apply_binary("%", -7, 3), 2)

    def test_caret_raises_to_a_whole_power(self):
        self.assertEqual(xpr_apply_binary("^", 2, 8), 256)
        self.assertIs(type(xpr_apply_binary("+", F(1, 2), F(1, 2))), int)
        self.assertIs(type(xpr_apply_binary("*", F(2, 3), F(3, 2))), int)

    def test_mixed_int_and_fraction(self):
        self.assertEqual(xpr_apply_binary("+", 1, F(1, 2)), F(3, 2))


class ZE(unittest.TestCase):
    def test_less_than_is_strict_at_equality(self):
        self.assertEqual(xpr_apply_binary("<", 1, 2), 1)
        self.assertEqual(xpr_apply_binary("<", 2, 2), 0)
        self.assertEqual(xpr_apply_binary("<", 3, 2), 0)
        self.assertEqual(xpr_apply_binary("<=", 1, 2), 1)
        self.assertEqual(xpr_apply_binary("<=", 2, 2), 1)
        self.assertEqual(xpr_apply_binary("<=", 3, 2), 0)

    def test_greater_than_is_strict_at_equality(self):
        self.assertEqual(xpr_apply_binary(">", 3, 2), 1)
        self.assertEqual(xpr_apply_binary(">", 2, 2), 0)
        self.assertEqual(xpr_apply_binary(">", 1, 2), 0)
        self.assertEqual(xpr_apply_binary(">=", 3, 2), 1)
        self.assertEqual(xpr_apply_binary(">=", 2, 2), 1)
        self.assertEqual(xpr_apply_binary(">=", 1, 2), 0)

    def test_equality_of_different_representations(self):
        self.assertEqual(xpr_apply_binary("==", F(4, 2), 2), 1)
        self.assertEqual(xpr_apply_binary("==", 1, 2), 0)
        self.assertEqual(xpr_apply_binary("!=", 1, 2), 1)
        self.assertEqual(xpr_apply_binary("!=", 2, 2), 0)

    def test_comparison_result_is_a_plain_int(self):
        self.assertIs(type(xpr_apply_binary("<", 1, 2)), int)


class ZF(unittest.TestCase):
    def test_conjunction_follows_the_boolean_table(self):
        self.assertEqual(
            [xpr_apply_binary("&&", a, b) for a, b in ((0, 0), (0, 5), (5, 0), (5, 7))],
            [0, 0, 0, 1],
        )
        self.assertEqual(
            [xpr_apply_binary("||", a, b) for a, b in ((0, 0), (0, 5), (5, 0), (5, 7))],
            [0, 1, 1, 1],
        )

    def test_logic_treats_fractions_as_true(self):
        self.assertEqual(xpr_apply_binary("&&", F(1, 3), F(1, 7)), 1)
        with self.assertRaisesRegex(XprEvalError, "^unknown binary operator '~'$"):
            xpr_apply_binary("~", 1, 1)
