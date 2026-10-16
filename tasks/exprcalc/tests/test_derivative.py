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


class BasicRuleTests(unittest.TestCase):
    def test_constant_has_zero_derivative(self):
        self.assertEqual(d("7"), sx("0"))

    def test_other_variable_has_zero_derivative(self):
        self.assertEqual(d("y"), sx("0"))

    def test_the_variable_itself_has_derivative_one(self):
        self.assertEqual(d("x"), sx("1"))

    def test_sum_rule_drops_constant_term(self):
        self.assertEqual(d("(+ x 1)"), sx("1"))

    def test_subtracting_constant_leaves_slope_alone(self):
        self.assertEqual(d("(- x 3)"), sx("1"))

    def test_difference_from_constant_negates(self):
        self.assertEqual(d("(- 3 x)"), sx("(- 1)"))

    def test_sum_of_two_variable_terms(self):
        self.assertEqual(d("(+ (* 2 x) (* 3 x))"), sx("5"))

    def test_minus_x_has_slope_minus_one(self):
        self.assertEqual(d("(- x)"), sx("(- 1)"))

    def test_negating_twice_restores_the_unit_slope(self):
        self.assertEqual(d("(- (- x))"), sx("1"))

    def test_unary_plus_passes_through(self):
        self.assertEqual(d("(+ x)"), sx("1"))

    def test_constant_factor_survives_differentiation_unchanged(self):
        self.assertEqual(d("(* 3 x)"), sx("3"))

    def test_product_with_independent_factor(self):
        self.assertEqual(d("(* x y)"), sx("y"))

    def test_product_rule_for_square_by_multiplication(self):
        self.assertEqual(d("(* x x)"), sx("(+ x x)"))

    def test_product_rule_general_case(self):
        self.assertEqual(
            d("(* (+ x 1) (+ x 2))"), sx("(+ (+ x 2) (+ x 1))")
        )


class PowerRuleTests(unittest.TestCase):
    def test_cubic_term_loses_one_power(self):
        self.assertEqual(d("(^ x 3)"), sx("(* 3 (^ x 2))"))

    def test_quadratic_term_becomes_twice_the_variable(self):
        self.assertEqual(d("(^ x 2)"), sx("(* 2 x)"))

    def test_linear_power_collapses_to_slope_one(self):
        self.assertEqual(d("(^ x 1)"), sx("1"))

    def test_power_zero_is_constant_so_slope_zero(self):
        self.assertEqual(d("(^ x 0)"), sx("0"))

    def test_reciprocal_power_gets_a_deeper_negative_exponent(self):
        self.assertEqual(d("(^ x (- 1))"), sx("(* (- 1) (^ x (- 2)))"))

    def test_half_exponent_yields_negative_half_power(self):
        self.assertEqual(d("(^ x 0.5)"), sx("(* 0.5 (^ x (- 0.5)))"))

    def test_chain_rule_with_linear_base(self):
        self.assertEqual(d("(^ (+ x 1) 2)"), sx("(* 2 (+ x 1))"))

    def test_chain_rule_multiplies_inner_derivative(self):
        self.assertEqual(d("(^ (* 2 x) 3)"), sx("(* 6 (^ (* 2 x) 2))"))

    def test_constant_base_is_constant(self):
        self.assertEqual(d("(^ 2 3)"), sx("0"))

    def test_exponent_depending_on_variable_is_refused(self):
        self.assertEqual(refusal(sx("(^ 2 x)")), ("exponent must not depend on the variable", None))

    def test_self_power_is_refused_with_exponent_message(self):
        with self.assertRaisesRegex(XprEvalError, "exponent must not depend"):
            d("(^ x x)")

    def test_differentiation_variable_can_be_any_name(self):
        self.assertEqual(d("(^ y 2)", "y"), sx("(* 2 y)"))


class QuotientAndCoefficientTests(unittest.TestCase):
    def test_dividing_by_constant_keeps_denominator_only(self):
        self.assertEqual(d("(/ x 2)"), sx("(/ 1 2)"))

    def test_reciprocal_slope_is_negative_inverse_square(self):
        self.assertEqual(d("(/ 1 x)"), sx("(/ (- 1) (^ x 2))"))

    def test_variable_denominator_triggers_the_full_quotient_formula(self):
        self.assertEqual(d("(/ x (+ x 1))"), sx("(/ (- (+ x 1) x) (^ (+ x 1) 2))"))

    def test_coefficient_merges_into_power_term(self):
        self.assertEqual(d("(* 3 (^ x 2))"), sx("(* 6 x)"))

    def test_full_polynomial_loses_its_constant_term(self):
        self.assertEqual(d("(+ (+ (* 3 (^ x 2)) (* 2 x)) 5)"), sx("(+ (* 6 x) 2)"))

    def test_cubic_minus_line_differentiates_termwise_exactly(self):
        self.assertEqual(d("(- (^ x 3) (* 4 x))"), sx("(- (* 3 (^ x 2)) 4)"))


class FunctionRuleTests(unittest.TestCase):
    def test_root_of_variable_divides_by_double_root(self):
        self.assertEqual(d("(call sqrt x)"), sx("(/ 1 (* 2 (call sqrt x)))"))

    def test_square_root_with_chain(self):
        self.assertEqual(
            d("(call sqrt (* 2 x))"), sx("(/ 2 (* 2 (call sqrt (* 2 x))))")
        )

    def test_abs_call_turns_into_sign_call(self):
        self.assertEqual(d("(call abs x)"), sx("(call sign x)"))

    def test_absolute_value_with_chain(self):
        self.assertEqual(d("(call abs (* 3 x))"), sx("(* (call sign (* 3 x)) 3)"))

    def test_pow_function_acts_like_operator(self):
        self.assertEqual(d("(call pow x 3)"), sx("(* 3 (^ x 2))"))

    def test_function_without_variable_is_constant(self):
        self.assertEqual(d("(call mystery y)"), sx("0"))

    def test_unsupported_function_is_refused_with_position(self):
        node = XprCall("mystery", (XprVar("x"),), 4)
        self.assertEqual(refusal(node), ("cannot differentiate function 'mystery'", 4))

    def test_supported_function_with_wrong_arity_is_refused(self):
        with self.assertRaisesRegex(XprEvalError, "cannot differentiate function 'sqrt'"):
            d("(call sqrt x x)")
        with self.assertRaisesRegex(XprEvalError, "cannot differentiate function 'pow'"):
            d("(call pow x)")
        with self.assertRaisesRegex(XprEvalError, "cannot differentiate function 'abs'"):
            d("(call abs x x)")


class UnsupportedOperatorTests(unittest.TestCase):
    def test_less_than_cannot_be_differentiated_anywhere(self):
        self.assertEqual(refusal(sx("(< x 1)")), ("cannot differentiate operator '<'", None))

    def test_logical_not_is_refused(self):
        self.assertEqual(refusal(sx("(! x)")), ("cannot differentiate operator '!'", None))

    def test_modulo_operator_is_not_differentiable_here(self):
        with self.assertRaisesRegex(XprEvalError, "cannot differentiate operator '%'"):
            d("(% x 2)")

    def test_constant_subtree_is_never_inspected_for_support(self):
        self.assertEqual(d("(+ x (< y 1))"), sx("1"))


class ConstructorTests(unittest.TestCase):
    def test_negating_a_literal_gives_negative_literal(self):
        self.assertEqual(xpr_make_negation(sx("3")), sx("(- 3)"))

    def test_negation_of_negative_literal(self):
        self.assertEqual(xpr_make_negation(sx("(- 3)")), sx("3"))

    def test_negation_of_zero_is_zero(self):
        self.assertEqual(xpr_make_negation(sx("0")), sx("0"))

    def test_negation_cancels_another_negation(self):
        self.assertEqual(xpr_make_negation(sx("(- x)")), sx("x"))

    def test_negating_a_variable_wraps_a_minus_node(self):
        self.assertEqual(xpr_make_negation(sx("x")), sx("(- x)"))

    def test_sum_drops_zero_on_either_side(self):
        self.assertEqual(xpr_make_sum(sx("0"), sx("x")), sx("x"))
        self.assertEqual(xpr_make_sum(sx("x"), sx("0")), sx("x"))

    def test_adding_literals_gives_one_literal_result(self):
        self.assertEqual(xpr_make_sum(sx("2"), sx("(- 5)")), sx("(- 3)"))

    def test_sum_of_terms_is_kept(self):
        self.assertEqual(xpr_make_sum(sx("x"), sx("1")), sx("(+ x 1)"))

    def test_difference_drops_zero_subtrahend(self):
        self.assertEqual(xpr_make_difference(sx("x"), sx("0")), sx("x"))

    def test_difference_from_zero_negates(self):
        self.assertEqual(xpr_make_difference(sx("0"), sx("x")), sx("(- x)"))

    def test_subtracting_literals_gives_one_literal_result(self):
        self.assertEqual(xpr_make_difference(sx("5"), sx("3")), sx("2"))
        self.assertEqual(xpr_make_difference(sx("3"), sx("5")), sx("(- 2)"))

    def test_difference_of_terms_is_kept(self):
        self.assertEqual(xpr_make_difference(sx("x"), sx("y")), sx("(- x y)"))

    def test_product_with_zero_is_zero(self):
        self.assertEqual(xpr_make_product(sx("0"), sx("x")), sx("0"))
        self.assertEqual(xpr_make_product(sx("x"), sx("0")), sx("0"))

    def test_product_drops_unit_factor(self):
        self.assertEqual(xpr_make_product(sx("1"), sx("x")), sx("x"))
        self.assertEqual(xpr_make_product(sx("x"), sx("1")), sx("x"))

    def test_multiplying_literals_gives_one_literal_result(self):
        self.assertEqual(xpr_make_product(sx("2"), sx("(- 3)")), sx("(- 6)"))

    def test_product_merges_coefficients_on_the_left(self):
        self.assertEqual(xpr_make_product(sx("2"), sx("(* 3 x)")), sx("(* 6 x)"))

    def test_product_merges_coefficients_on_the_right(self):
        self.assertEqual(xpr_make_product(sx("(* 2 x)"), sx("3")), sx("(* 6 x)"))

    def test_merged_unit_coefficient_disappears(self):
        self.assertEqual(xpr_make_product(sx("0.5"), sx("(* 2 x)")), sx("x"))

    def test_product_keeps_unrelated_factors(self):
        self.assertEqual(xpr_make_product(sx("2"), sx("(+ x 1)")), sx("(* 2 (+ x 1))"))
        self.assertEqual(xpr_make_product(sx("(* x 3)"), sx("y")), sx("(* (* x 3) y)"))

    def test_quotient_of_zero_is_zero(self):
        self.assertEqual(xpr_make_quotient(sx("0"), sx("x")), sx("0"))

    def test_quotient_by_one_is_the_numerator(self):
        self.assertEqual(xpr_make_quotient(sx("x"), sx("1")), sx("x"))

    def test_quotient_of_literals_is_kept(self):
        self.assertEqual(xpr_make_quotient(sx("6"), sx("3")), sx("(/ 6 3)"))

    def test_power_zero_and_one(self):
        self.assertEqual(xpr_make_power(sx("x"), sx("0")), sx("1"))
        self.assertEqual(xpr_make_power(sx("x"), sx("1")), sx("x"))

    def test_power_otherwise_is_kept(self):
        self.assertEqual(xpr_make_power(sx("x"), sx("2")), sx("(^ x 2)"))


class RepeatedDerivativeTests(unittest.TestCase):
    def test_order_zero_returns_the_input(self):
        tree = sx("(^ x 3)")
        self.assertIs(xpr_nth_derivative(tree, "x", 0), tree)

    def test_twice_differentiated_cubic_is_linear(self):
        self.assertEqual(xpr_nth_derivative(sx("(^ x 3)"), "x", 2), sx("(* 6 x)"))

    def test_third_derivative_is_constant(self):
        self.assertEqual(xpr_nth_derivative(sx("(^ x 3)"), "x", 3), sx("6"))

    def test_cubic_is_flat_after_four_rounds(self):
        self.assertEqual(xpr_nth_derivative(sx("(^ x 3)"), "x", 4), sx("0"))

    def test_negative_order_is_refused(self):
        with self.assertRaisesRegex(ValueError, "^order must not be negative$"):
            xpr_nth_derivative(sx("x"), "x", -1)

    def test_gradient_maps_each_variable(self):
        tree = sx("(+ (* x y) (^ y 2))")
        self.assertEqual(
            xpr_gradient(tree, ["x", "y"]), {"x": sx("y"), "y": sx("(+ x (* 2 y))")}
        )

    def test_gradient_keeps_requested_order(self):
        self.assertEqual(list(xpr_gradient(sx("(+ x y)"), ["y", "x"])), ["y", "x"])

    def test_gradient_of_nothing_is_empty(self):
        self.assertEqual(xpr_gradient(sx("x"), []), {})
