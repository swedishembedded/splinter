# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
import unittest

from exprcalc.astnodes import XprCall, XprNum, XprVar
from exprcalc.derivative import (
    xpr_differentiate,
    xpr_gradient,
    xpr_make_difference,
    xpr_make_negation,
    xpr_make_power,
    xpr_make_product,
    xpr_make_quotient,
    xpr_make_sum,
    xpr_nth_derivative,
)
from exprcalc.errors import XprEvalError
from .sexpr import sx


def d(tree, variable="x"):
    return xpr_differentiate(sx(tree), variable)


def refusal(node, variable="x"):
    with unittest.TestCase().assertRaises(XprEvalError) as caught:
        xpr_differentiate(node, variable)
    return caught.exception.message, caught.exception.position


class ZA(unittest.TestCase):
    def test_constant_has_zero_derivative(self):
        self.assertEqual(d("7"), sx("0"))
        self.assertEqual(d("y"), sx("0"))
        self.assertEqual(d("x"), sx("1"))
        self.assertEqual(d("(+ x 1)"), sx("1"))

    def test_subtracting_constant_leaves_slope_alone(self):
        self.assertEqual(d("(- x 3)"), sx("1"))
        self.assertEqual(d("(- 3 x)"), sx("(- 1)"))
        self.assertEqual(d("(+ (* 2 x) (* 3 x))"), sx("5"))
        self.assertEqual(d("(- x)"), sx("(- 1)"))

    def test_negating_twice_restores_the_unit_slope(self):
        self.assertEqual(d("(- (- x))"), sx("1"))
        self.assertEqual(d("(+ x)"), sx("1"))
        self.assertEqual(d("(* 3 x)"), sx("3"))
        self.assertEqual(d("(* x y)"), sx("y"))

    def test_product_rule_for_square_by_multiplication(self):
        self.assertEqual(d("(* x x)"), sx("(+ x x)"))
        self.assertEqual(
            d("(* (+ x 1) (+ x 2))"), sx("(+ (+ x 2) (+ x 1))")
        )


class ZB(unittest.TestCase):
    def test_cubic_term_loses_one_power(self):
        self.assertEqual(d("(^ x 3)"), sx("(* 3 (^ x 2))"))
        self.assertEqual(d("(^ x 2)"), sx("(* 2 x)"))
        self.assertEqual(d("(^ x 1)"), sx("1"))
        self.assertEqual(d("(^ x 0)"), sx("0"))

    def test_reciprocal_power_gets_a_deeper_negative_exponent(self):
        self.assertEqual(d("(^ x (- 1))"), sx("(* (- 1) (^ x (- 2)))"))
        self.assertEqual(d("(^ x 0.5)"), sx("(* 0.5 (^ x (- 0.5)))"))
        self.assertEqual(d("(^ (+ x 1) 2)"), sx("(* 2 (+ x 1))"))
        self.assertEqual(d("(^ (* 2 x) 3)"), sx("(* 6 (^ (* 2 x) 2))"))

    def test_constant_base_is_constant(self):
        self.assertEqual(d("(^ 2 3)"), sx("0"))
        self.assertEqual(refusal(sx("(^ 2 x)")), ("exponent must not depend on the variable", None))
        with self.assertRaisesRegex(XprEvalError, "exponent must not depend"):
            d("(^ x x)")
        self.assertEqual(d("(^ y 2)", "y"), sx("(* 2 y)"))


class ZC(unittest.TestCase):
    def test_dividing_by_constant_keeps_denominator_only(self):
        self.assertEqual(d("(/ x 2)"), sx("(/ 1 2)"))
        self.assertEqual(d("(/ 1 x)"), sx("(/ (- 1) (^ x 2))"))
        self.assertEqual(d("(/ x (+ x 1))"), sx("(/ (- (+ x 1) x) (^ (+ x 1) 2))"))
        self.assertEqual(d("(* 3 (^ x 2))"), sx("(* 6 x)"))

    def test_full_polynomial_loses_its_constant_term(self):
        self.assertEqual(d("(+ (+ (* 3 (^ x 2)) (* 2 x)) 5)"), sx("(+ (* 6 x) 2)"))
        self.assertEqual(d("(- (^ x 3) (* 4 x))"), sx("(- (* 3 (^ x 2)) 4)"))


class ZD(unittest.TestCase):
    def test_root_of_variable_divides_by_double_root(self):
        self.assertEqual(d("(call sqrt x)"), sx("(/ 1 (* 2 (call sqrt x)))"))
        self.assertEqual(
            d("(call sqrt (* 2 x))"), sx("(/ 2 (* 2 (call sqrt (* 2 x))))")
        )
        self.assertEqual(d("(call abs x)"), sx("(call sign x)"))
        self.assertEqual(d("(call abs (* 3 x))"), sx("(* (call sign (* 3 x)) 3)"))

    def test_pow_function_acts_like_operator(self):
        self.assertEqual(d("(call pow x 3)"), sx("(* 3 (^ x 2))"))
        self.assertEqual(d("(call mystery y)"), sx("0"))
        node = XprCall("mystery", (XprVar("x"),), 4)
        self.assertEqual(refusal(node), ("cannot differentiate function 'mystery'", 4))
        with self.assertRaisesRegex(XprEvalError, "cannot differentiate function 'sqrt'"):
            d("(call sqrt x x)")
        with self.assertRaisesRegex(XprEvalError, "cannot differentiate function 'pow'"):
            d("(call pow x)")
        with self.assertRaisesRegex(XprEvalError, "cannot differentiate function 'abs'"):
            d("(call abs x x)")


class ZE(unittest.TestCase):
    def test_less_than_cannot_be_differentiated_anywhere(self):
        self.assertEqual(refusal(sx("(< x 1)")), ("cannot differentiate operator '<'", None))
        self.assertEqual(refusal(sx("(! x)")), ("cannot differentiate operator '!'", None))
        with self.assertRaisesRegex(XprEvalError, "cannot differentiate operator '%'"):
            d("(% x 2)")
        self.assertEqual(d("(+ x (< y 1))"), sx("1"))


class ZF(unittest.TestCase):
    def test_negating_a_literal_gives_negative_literal(self):
        self.assertEqual(xpr_make_negation(sx("3")), sx("(- 3)"))
        self.assertEqual(xpr_make_negation(sx("(- 3)")), sx("3"))
        self.assertEqual(xpr_make_negation(sx("0")), sx("0"))
        self.assertEqual(xpr_make_negation(sx("(- x)")), sx("x"))

    def test_negating_a_variable_wraps_a_minus_node(self):
        self.assertEqual(xpr_make_negation(sx("x")), sx("(- x)"))
        self.assertEqual(xpr_make_sum(sx("0"), sx("x")), sx("x"))
        self.assertEqual(xpr_make_sum(sx("x"), sx("0")), sx("x"))
        self.assertEqual(xpr_make_sum(sx("2"), sx("(- 5)")), sx("(- 3)"))
        self.assertEqual(xpr_make_sum(sx("x"), sx("1")), sx("(+ x 1)"))

    def test_difference_drops_zero_subtrahend(self):
        self.assertEqual(xpr_make_difference(sx("x"), sx("0")), sx("x"))
        self.assertEqual(xpr_make_difference(sx("0"), sx("x")), sx("(- x)"))
        self.assertEqual(xpr_make_difference(sx("5"), sx("3")), sx("2"))
        self.assertEqual(xpr_make_difference(sx("3"), sx("5")), sx("(- 2)"))
        self.assertEqual(xpr_make_difference(sx("x"), sx("y")), sx("(- x y)"))

    def test_product_with_zero_is_zero(self):
        self.assertEqual(xpr_make_product(sx("0"), sx("x")), sx("0"))
        self.assertEqual(xpr_make_product(sx("x"), sx("0")), sx("0"))
        self.assertEqual(xpr_make_product(sx("1"), sx("x")), sx("x"))
        self.assertEqual(xpr_make_product(sx("x"), sx("1")), sx("x"))
        self.assertEqual(xpr_make_product(sx("2"), sx("(- 3)")), sx("(- 6)"))
        self.assertEqual(xpr_make_product(sx("2"), sx("(* 3 x)")), sx("(* 6 x)"))

    def test_product_merges_coefficients_on_the_right(self):
        self.assertEqual(xpr_make_product(sx("(* 2 x)"), sx("3")), sx("(* 6 x)"))
        self.assertEqual(xpr_make_product(sx("0.5"), sx("(* 2 x)")), sx("x"))
        self.assertEqual(xpr_make_product(sx("2"), sx("(+ x 1)")), sx("(* 2 (+ x 1))"))
        self.assertEqual(xpr_make_product(sx("(* x 3)"), sx("y")), sx("(* (* x 3) y)"))
        self.assertEqual(xpr_make_quotient(sx("0"), sx("x")), sx("0"))

    def test_quotient_by_one_is_the_numerator(self):
        self.assertEqual(xpr_make_quotient(sx("x"), sx("1")), sx("x"))
        self.assertEqual(xpr_make_quotient(sx("6"), sx("3")), sx("(/ 6 3)"))
        self.assertEqual(xpr_make_power(sx("x"), sx("0")), sx("1"))
        self.assertEqual(xpr_make_power(sx("x"), sx("1")), sx("x"))
        self.assertEqual(xpr_make_power(sx("x"), sx("2")), sx("(^ x 2)"))


class ZG(unittest.TestCase):
    def test_order_zero_returns_the_input(self):
        tree = sx("(^ x 3)")
        self.assertIs(xpr_nth_derivative(tree, "x", 0), tree)
        self.assertEqual(xpr_nth_derivative(sx("(^ x 3)"), "x", 2), sx("(* 6 x)"))
        self.assertEqual(xpr_nth_derivative(sx("(^ x 3)"), "x", 3), sx("6"))
        self.assertEqual(xpr_nth_derivative(sx("(^ x 3)"), "x", 4), sx("0"))

    def test_negative_order_is_refused(self):
        with self.assertRaisesRegex(ValueError, "^order must not be negative$"):
            xpr_nth_derivative(sx("x"), "x", -1)
        tree = sx("(+ (* x y) (^ y 2))")
        self.assertEqual(
            xpr_gradient(tree, ["x", "y"]), {"x": sx("y"), "y": sx("(+ x (* 2 y))")}
        )
        self.assertEqual(list(xpr_gradient(sx("(+ x y)"), ["y", "x"])), ["y", "x"])
        self.assertEqual(xpr_gradient(sx("x"), []), {})
