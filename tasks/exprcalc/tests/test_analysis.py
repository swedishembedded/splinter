# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
import unittest

from exprcalc.analysis import (
    xpr_called_functions,
    xpr_dependency_order,
    xpr_free_variables,
    xpr_is_constant_expression,
    xpr_operator_census,
    xpr_substitute_variable,
)
from exprcalc.errors import XprEvalError
from .sexpr import sx


def cycle_message(definitions):
    with unittest.TestCase().assertRaises(XprEvalError) as caught:
        xpr_dependency_order({name: sx(tree) for name, tree in definitions.items()})
    return caught.exception.message


def order(**definitions):
    return xpr_dependency_order({name: sx(tree) for name, tree in definitions.items()})


class FreeVariableTests(unittest.TestCase):
    def test_a_literal_mentions_no_variables_at_all(self):
        self.assertEqual(xpr_free_variables(sx("5")), [])

    def test_lone_variable_node_lists_itself_only(self):
        self.assertEqual(xpr_free_variables(sx("x")), ["x"])

    def test_variables_listed_by_first_appearance_order(self):
        self.assertEqual(xpr_free_variables(sx("(+ b (* a c))")), ["b", "a", "c"])

    def test_repeated_variable_uses_are_listed_once(self):
        self.assertEqual(xpr_free_variables(sx("(+ (* a a) (- b a))")), ["a", "b"])

    def test_variables_inside_calls_are_found(self):
        self.assertEqual(xpr_free_variables(sx("(call f x (call g y) z)")), ["x", "y", "z"])

    def test_function_names_are_not_variables(self):
        self.assertEqual(xpr_free_variables(sx("(call f 1)")), [])


class CalledFunctionTests(unittest.TestCase):
    def test_expression_without_calls_lists_no_functions(self):
        self.assertEqual(xpr_called_functions(sx("(+ x 1)")), [])

    def test_outer_call_comes_first(self):
        self.assertEqual(xpr_called_functions(sx("(call f (call g 1))")), ["f", "g"])

    def test_repeated_calls_listed_once(self):
        self.assertEqual(xpr_called_functions(sx("(+ (call f 1) (call f 2))")), ["f"])

    def test_variables_are_not_functions(self):
        self.assertEqual(xpr_called_functions(sx("(+ f (call g f))")), ["g"])


class ConstantExpressionTests(unittest.TestCase):
    def test_literal_arithmetic_is_constant(self):
        self.assertTrue(xpr_is_constant_expression(sx("(+ 1 (* 2 3))")))

    def test_call_with_literals_is_constant(self):
        self.assertTrue(xpr_is_constant_expression(sx("(call max 1 2)")))

    def test_variable_makes_it_non_constant(self):
        self.assertFalse(xpr_is_constant_expression(sx("(+ 1 x)")))


class OperatorCensusTests(unittest.TestCase):
    def test_leaf_has_empty_census(self):
        self.assertEqual(xpr_operator_census(sx("x")), {})

    def test_operator_census_counts_each_symbol_once(self):
        self.assertEqual(xpr_operator_census(sx("(+ (+ a b) (* c d))")), {"+": 2, "*": 1})

    def test_census_order_is_first_appearance(self):
        self.assertEqual(list(xpr_operator_census(sx("(* (+ a b) (+ c d))"))), ["*", "+"])

    def test_unary_and_binary_minus_are_separate(self):
        self.assertEqual(xpr_operator_census(sx("(- (- a) b)")), {"-": 1, "unary -": 1})

    def test_calls_do_not_count_as_operators(self):
        self.assertEqual(xpr_operator_census(sx("(call f (! a))")), {"unary !": 1})


class SubstitutionTests(unittest.TestCase):
    def test_substitution_swaps_variable_for_given_tree(self):
        self.assertEqual(xpr_substitute_variable(sx("(+ x 1)"), "x", sx("(* 2 y)")), sx("(+ (* 2 y) 1)"))

    def test_every_use_is_replaced(self):
        self.assertEqual(xpr_substitute_variable(sx("(* x x)"), "x", sx("3")), sx("(* 3 3)"))

    def test_other_variables_are_untouched(self):
        self.assertEqual(xpr_substitute_variable(sx("(+ x y)"), "x", sx("1")), sx("(+ 1 y)"))

    def test_replacement_is_not_substituted_again(self):
        self.assertEqual(xpr_substitute_variable(sx("x"), "x", sx("(+ x 1)")), sx("(+ x 1)"))

    def test_substitution_reaches_call_arguments(self):
        self.assertEqual(xpr_substitute_variable(sx("(call f x)"), "x", sx("2")), sx("(call f 2)"))

    def test_missing_variable_changes_nothing(self):
        tree = sx("(+ a b)")
        self.assertEqual(xpr_substitute_variable(tree, "z", sx("1")), tree)

    def test_function_name_is_not_a_variable(self):
        self.assertEqual(xpr_substitute_variable(sx("(call x 1)"), "x", sx("9")), sx("(call x 1)"))


class DependencyOrderTests(unittest.TestCase):
    def test_no_definitions_yield_an_empty_order(self):
        self.assertEqual(xpr_dependency_order({}), [])

    def test_independent_names_keep_insertion_order(self):
        self.assertEqual(order(b="1", a="2", c="3"), ["b", "a", "c"])

    def test_dependency_moves_before_its_user(self):
        self.assertEqual(order(total="(+ base 1)", base="5"), ["base", "total"])

    def test_chain_is_resolved_transitively(self):
        self.assertEqual(order(c="b", b="a", a="1"), ["a", "b", "c"])

    def test_shared_dependency_appears_once(self):
        self.assertEqual(order(x="(+ s 1)", y="(* s 2)", s="3"), ["s", "x", "y"])

    def test_undefined_names_are_ignored(self):
        self.assertEqual(order(a="(+ outside 1)"), ["a"])

    def test_dependencies_follow_first_use_order(self):
        self.assertEqual(order(r="(+ q p)", p="1", q="2"), ["q", "p", "r"])

    def test_self_reference_is_a_cycle(self):
        self.assertEqual(cycle_message({"a": "(+ a 1)"}), "circular definition: a -> a")

    def test_mutual_cycle_is_spelled_out(self):
        self.assertEqual(cycle_message({"a": "b", "b": "a"}), "circular definition: a -> b -> a")

    def test_cycle_message_starts_at_the_cycle_entry(self):
        self.assertEqual(
            cycle_message({"top": "a", "a": "b", "b": "c", "c": "a"}),
            "circular definition: a -> b -> c -> a",
        )

    def test_diamond_is_not_a_cycle(self):
        self.assertEqual(
            order(d="(+ b c)", b="a", c="a", a="1"),
            ["a", "b", "c", "d"],
        )
