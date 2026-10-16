# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
import unittest
from fractions import Fraction

from exprcalc.errors import XprEvalError, XprSyntaxError
from exprcalc.evaluator import (
    XprScope,
    xpr_evaluate_definitions,
    xpr_evaluate_node,
    xpr_evaluate_text,
)
from exprcalc.functions import xpr_builtin_registry, xpr_register_function
from .sexpr import sx

F = Fraction
REG = xpr_builtin_registry()


def run(tree, **names):
    return xpr_evaluate_node(sx(tree), XprScope(names), REG)


def failure(tree, **names):
    with unittest.TestCase().assertRaises(XprEvalError) as caught:
        xpr_evaluate_node(sx(tree), XprScope(names), REG)
    return caught.exception.message, caught.exception.position


class ZA(unittest.TestCase):
    def test_lookup_finds_own_binding(self):
        self.assertEqual(XprScope({"a": 1}).lookup("a"), 1)
        with self.assertRaisesRegex(XprEvalError, "^undefined variable 'zz'$"):
            XprScope().lookup("zz")

    def test_child_sees_parent_bindings(self):
        self.assertEqual(XprScope({"a": 1}).child().lookup("a"), 1)
        self.assertEqual(XprScope({"a": 1}).child({"a": 2}).lookup("a"), 2)

    def test_parent_does_not_see_child_bindings(self):
        parent = XprScope()
        parent.child({"a": 1})
        with self.assertRaises(XprEvalError):
            parent.lookup("a")
        scope = XprScope()
        scope.define("k", 5)
        self.assertEqual(scope.lookup("k"), 5)

    def test_scope_copies_the_given_mapping(self):
        source = {"a": 1}
        XprScope(source).define("b", 2)
        self.assertEqual(source, {"a": 1})
        self.assertEqual(XprScope({"a": 0}, XprScope({"a": 5})).lookup("a"), 0)

    def test_names_merge_parent_and_child_sorted(self):
        scope = XprScope({"b": 1, "a": 2}).child({"c": 3, "a": 4})
        self.assertEqual(scope.names(), ["a", "b", "c"])


class ZB(unittest.TestCase):
    def test_plain_number_node_evaluates_to_itself(self):
        self.assertEqual(run("7"), 7)
        self.assertEqual(run("x", x=F(3, 2)), F(3, 2))

    def test_two_bound_names_add_up(self):
        self.assertEqual(run("(+ x y)", x=1, y=2), 3)
        self.assertEqual(run("(- x)", x=4), -4)

    def test_nested_operator_tree_collapses_to_seven(self):
        self.assertEqual(run("(- (* 3 (+ 1 2)) (/ 4 2))"), 7)
        self.assertEqual(run("(+ 0.1 0.2)"), F(3, 10))

    def test_power_feeds_into_remainder_operation(self):
        self.assertEqual(run("(% (^ 2 5) 7)"), 4)
        self.assertEqual(run("(call max 1 (+ x 2) 3)", x=5), 7)


class ZC(unittest.TestCase):
    def test_comparison_gives_one_or_zero(self):
        self.assertEqual(run("(< x 3)", x=2), 1)
        self.assertEqual(run("(< x 3)", x=3), 0)
        self.assertEqual(run("(&& 0 (/ 1 0))"), 0)

    def test_conjunction_still_checks_right_when_left_holds(self):
        self.assertEqual(run("(&& 1 7)"), 1)
        with self.assertRaises(XprEvalError):
            run("(&& 1 (/ 1 0))")
        self.assertEqual(run("(|| 5 (/ 1 0))"), 1)

    def test_disjunction_checks_right_when_left_is_false(self):
        self.assertEqual(run("(|| 0 0)"), 0)
        self.assertEqual(run("(|| 0 3)"), 1)
        with self.assertRaises(XprEvalError):
            run("(|| 0 (/ 1 0))")
        self.assertEqual(run("(&& 0 ghost)"), 0)

    def test_bang_operator_inverts_truthiness(self):
        self.assertEqual(run("(! x)", x=0), 1)


class ZD(unittest.TestCase):
    def test_division_error_points_at_operator(self):
        self.assertEqual(xpr_evaluate_text_error("10 / (5 - 5)"), ("division by zero", 3))
        self.assertEqual(xpr_evaluate_text_error("1 + ghost"), ("undefined variable 'ghost'", 4))

    def test_unknown_function_points_at_the_call(self):
        self.assertEqual(xpr_evaluate_text_error("2 * nope(1)"), ("unknown function 'nope'", 4))
        self.assertEqual(
            xpr_evaluate_text_error("abs(1, 2)"), ("function 'abs' expects 1 argument, got 2", 0)
        )

    def test_deepest_failing_node_supplies_the_position(self):
        self.assertEqual(xpr_evaluate_text_error("1 + (2 + 3 / 0)"), ("division by zero", 11))
        self.assertEqual(xpr_evaluate_text_error("abs(1 % 0)"), ("modulo by zero", 6))

    def test_error_text_includes_position(self):
        with self.assertRaisesRegex(XprEvalError, "^division by zero at position 2$"):
            xpr_evaluate_text("1 / 0")
        self.assertEqual(failure("(/ 1 0)"), ("division by zero", None))


def xpr_evaluate_text_error(source):
    with unittest.TestCase().assertRaises(XprEvalError) as caught:
        xpr_evaluate_text(source)
    return caught.exception.message, caught.exception.position


class ZE(unittest.TestCase):
    def test_evaluation_text_honours_operator_precedence(self):
        self.assertEqual(xpr_evaluate_text("1 + 2 * 3 ^ 2"), 19)
        self.assertEqual(xpr_evaluate_text("a * b", {"a": 6, "b": F(1, 3)}), 2)

    def test_default_registry_has_builtins(self):
        self.assertEqual(xpr_evaluate_text("gcd(12, 18) + sqrt(16)"), 10)
        reg = xpr_register_function({}, "triple", 1, 1, lambda x: 3 * x)
        self.assertEqual(xpr_evaluate_text("triple(4)", registry=reg), 12)
        with self.assertRaises(XprEvalError):
            xpr_evaluate_text("abs(1)", registry=reg)

    def test_empty_registry_is_respected_not_defaulted(self):
        with self.assertRaisesRegex(XprEvalError, "unknown function 'abs'"):
            xpr_evaluate_text("abs(1)", registry={})
        with self.assertRaises(XprSyntaxError):
            xpr_evaluate_text("1 +")

    def test_negation_binds_below_power(self):
        self.assertEqual(xpr_evaluate_text("-2 ^ 2"), -4)
        self.assertEqual(xpr_evaluate_text("if(x > 2, 10, 20)", {"x": 3}), 10)


class ZF(unittest.TestCase):
    def test_unrelated_definitions_evaluate_side_by_side(self):
        self.assertEqual(xpr_evaluate_definitions({"a": "1 + 1", "b": "3"}), {"a": 2, "b": 3})
        result = xpr_evaluate_definitions({"total": "base * 2", "base": "5"})
        self.assertEqual(result, {"base": 5, "total": 10})

    def test_three_step_chain_is_evaluated_in_order(self):
        result = xpr_evaluate_definitions({"c": "b + 1", "b": "a + 1", "a": "1"})
        self.assertEqual(list(result), ["a", "b", "c"])
        self.assertEqual(result["c"], 3)
        self.assertEqual(xpr_evaluate_definitions({"y": "x * x"}, {"x": 4}), {"y": 16})

    def test_circular_definitions_raise_with_the_cycle(self):
        with self.assertRaisesRegex(XprEvalError, "^circular definition: a -> b -> a$"):
            xpr_evaluate_definitions({"a": "b", "b": "a"})
        with self.assertRaisesRegex(XprEvalError, "undefined variable 'q'"):
            xpr_evaluate_definitions({"a": "q + 1"})

    def test_custom_registry_is_used(self):
        reg = xpr_register_function({}, "one", 0, 0, lambda: 1)
        self.assertEqual(xpr_evaluate_definitions({"a": "one() + 1"}, registry=reg), {"a": 2})
        names = {"x": 1}
        xpr_evaluate_definitions({"y": "x + 1"}, names)
        self.assertEqual(names, {"x": 1})
