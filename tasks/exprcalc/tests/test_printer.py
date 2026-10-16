# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
import unittest
from fractions import Fraction

from exprcalc.astnodes import XprBinary, XprCall, XprNum, XprUnary, XprVar
from exprcalc.parser import xpr_parse_text
from exprcalc.printer import xpr_render_minimal, xpr_render_parenthesized, xpr_render_tree
from .sexpr import sx


def show(tree):
    return xpr_render_minimal(sx(tree))


class AtomRenderTests(unittest.TestCase):
    def test_whole_literal_prints_as_digits(self):
        self.assertEqual(show("12"), "12")

    def test_fractional_literal_prints_in_decimal_form(self):
        self.assertEqual(show("2.5"), "2.5")

    def test_variable_prints_as_its_name(self):
        self.assertEqual(show("speed"), "speed")

    def test_call_arguments_are_comma_separated(self):
        self.assertEqual(show("(call f 1 (+ x 2))"), "f(1, x + 2)")

    def test_call_with_nothing_inside_prints_empty_brackets(self):
        self.assertEqual(xpr_render_minimal(XprCall("f", ())), "f()")

    def test_negative_literal_node_in_product(self):
        self.assertEqual(xpr_render_minimal(XprBinary("*", XprNum(-3), XprVar("x"))), "-3 * x")


class SpacingTests(unittest.TestCase):
    def test_binary_operators_are_spaced(self):
        self.assertEqual(show("(+ a b)"), "a + b")

    def test_comparison_and_logic_are_spaced(self):
        self.assertEqual(show("(|| (&& a b) (<= c 1))"), "a && b || c <= 1")

    def test_caret_prints_without_surrounding_spaces(self):
        self.assertEqual(show("(^ x 2)"), "x^2")

    def test_unary_operator_is_tight(self):
        self.assertEqual(show("(! x)"), "!x")
        self.assertEqual(show("(- x)"), "-x")


class MinimalParenthesesTests(unittest.TestCase):
    def test_product_inside_sum_needs_none(self):
        self.assertEqual(show("(+ 1 (* 2 3))"), "1 + 2 * 3")

    def test_sum_inside_product_needs_them(self):
        self.assertEqual(show("(* (+ 1 2) 3)"), "(1 + 2) * 3")
        self.assertEqual(show("(* 3 (+ 1 2))"), "3 * (1 + 2)")

    def test_left_nested_subtraction_needs_none(self):
        self.assertEqual(show("(- (- a b) c)"), "a - b - c")

    def test_right_nested_subtraction_needs_them(self):
        self.assertEqual(show("(- a (- b c))"), "a - (b - c)")

    def test_right_nested_sum_keeps_its_shape(self):
        self.assertEqual(show("(+ a (+ b c))"), "a + (b + c)")

    def test_right_nested_division_needs_them(self):
        self.assertEqual(show("(/ a (* b c))"), "a / (b * c)")
        self.assertEqual(show("(/ (/ a b) c)"), "a / b / c")

    def test_right_nested_power_needs_none(self):
        self.assertEqual(show("(^ 2 (^ 3 2))"), "2^3^2")

    def test_left_nested_power_needs_them(self):
        self.assertEqual(show("(^ (^ 2 3) 2)"), "(2^3)^2")

    def test_negation_under_power_base_needs_them(self):
        self.assertEqual(show("(^ (- x) 2)"), "(-x)^2")

    def test_negation_of_power_needs_none(self):
        self.assertEqual(show("(- (^ x 2))"), "-x^2")

    def test_negated_exponent_needs_none(self):
        self.assertEqual(show("(^ 2 (- x))"), "2^-x")

    def test_sum_in_exponent_needs_them(self):
        self.assertEqual(show("(^ 2 (+ a 1))"), "2^(a + 1)")

    def test_sum_under_negation_needs_them(self):
        self.assertEqual(show("(- (+ a b))"), "-(a + b)")

    def test_product_under_negation_needs_them(self):
        self.assertEqual(show("(- (* a b))"), "-(a * b)")

    def test_negation_inside_product_needs_none(self):
        self.assertEqual(show("(* (- a) b)"), "-a * b")
        self.assertEqual(show("(* a (- b))"), "a * -b")

    def test_bang_then_minus_prints_without_gaps(self):
        self.assertEqual(show("(! (- x))"), "!-x")

    def test_equality_of_equalities_needs_them(self):
        self.assertEqual(show("(== a (== b c))"), "a == (b == c)")
        self.assertEqual(show("(== (== a b) c)"), "a == b == c")

    def test_comparison_inside_logic_needs_none(self):
        self.assertEqual(show("(&& (< a 1) (!= b 2))"), "a < 1 && b != 2")

    def test_or_inside_and_needs_them(self):
        self.assertEqual(show("(&& a (|| b c))"), "a && (b || c)")

    def test_call_argument_never_needs_them(self):
        self.assertEqual(show("(call f (+ a b) (|| c d))"), "f(a + b, c || d)")

    def test_zero_and_one_literals_need_no_brackets(self):
        self.assertEqual(xpr_render_minimal(XprBinary("^", XprNum(0), XprNum(2))), "0^2")
        self.assertEqual(xpr_render_minimal(XprBinary("^", XprNum(1), XprNum(2))), "1^2")

    def test_negative_literal_under_power_base(self):
        self.assertEqual(xpr_render_minimal(XprBinary("^", XprNum(-2), XprNum(2))), "(-2)^2")


class RoundTripTests(unittest.TestCase):
    def test_parse_of_rendering_restores_the_tree(self):
        sources = [
            "1 + 2 * 3 - 4 / (5 % 6)",
            "-x ^ 2 + (-x) ^ 2",
            "a && b || !c == d",
            "2 ^ 3 ^ 4 ^ -x",
            "max(a, b + 1, f(c)) <= 10",
            "a - (b - c) - (d + e)",
        ]
        for source in sources:
            tree = xpr_parse_text(source)
            self.assertEqual(xpr_parse_text(xpr_render_minimal(tree)), tree, source)

    def test_rendering_is_stable_on_canonical_text(self):
        canonical = "a - (b - c) * 2^3^d / (e + 1)"
        self.assertEqual(xpr_render_minimal(xpr_parse_text(canonical)), canonical)

    def test_unrepresentable_number_is_refused(self):
        with self.assertRaisesRegex(ValueError, "no finite decimal form"):
            xpr_render_minimal(XprNum(Fraction(1, 3)))


class ParenthesizedTests(unittest.TestCase):
    def test_every_binary_is_wrapped(self):
        self.assertEqual(xpr_render_parenthesized(sx("(+ 1 (* 2 3))")), "(1 + (2 * 3))")

    def test_prefix_operation_gets_its_own_brackets(self):
        self.assertEqual(xpr_render_parenthesized(sx("(- x)")), "(-x)")

    def test_leaf_nodes_get_no_wrapping_brackets(self):
        self.assertEqual(xpr_render_parenthesized(sx("x")), "x")
        self.assertEqual(xpr_render_parenthesized(sx("0.5")), "0.5")

    def test_calls_wrap_only_their_operators(self):
        self.assertEqual(xpr_render_parenthesized(sx("(call f (+ a 1) b)")), "f((a + 1), b)")

    def test_power_keeps_spaced_operator(self):
        self.assertEqual(xpr_render_parenthesized(sx("(^ 2 3)")), "(2 ^ 3)")


class TreeOutlineTests(unittest.TestCase):
    def test_single_variable_outline_is_one_line(self):
        self.assertEqual(xpr_render_tree(sx("x")), "Var x")

    def test_number_outline_uses_decimal_text(self):
        self.assertEqual(xpr_render_tree(sx("0.25")), "Num 0.25")

    def test_outline_indents_two_spaces_per_level(self):
        self.assertEqual(
            xpr_render_tree(sx("(+ 1 (- (call f x)))")),
            "Binary +\n  Num 1\n  Unary -\n    Call f\n      Var x",
        )

    def test_start_indent_is_honoured(self):
        self.assertEqual(xpr_render_tree(sx("(- 1)"), 2), "    Unary -\n      Num 1")

    def test_negation_outline_nests_its_number(self):
        self.assertEqual(xpr_render_tree(XprUnary("-", XprNum(2))), "Unary -\n  Num 2")
