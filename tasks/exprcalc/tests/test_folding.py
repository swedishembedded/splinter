# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
import unittest

from exprcalc.astnodes import XprBinary, XprNum
from exprcalc.folding import xpr_fold_constants
from exprcalc.functions import xpr_builtin_registry, xpr_register_function
from .sexpr import sx

REG = xpr_builtin_registry()


def fold(tree, registry=None):
    return xpr_fold_constants(sx(tree), registry)


class ZA(unittest.TestCase):
    def test_adding_two_literals_gives_their_total(self):
        self.assertEqual(fold("(+ 1 2)"), sx("3"))
        self.assertEqual(fold("(* (+ 1 2) (- 10 4))"), sx("18"))

    def test_decimal_results_stay_exact(self):
        self.assertEqual(fold("(+ 0.1 0.2)"), sx("0.3"))
        self.assertEqual(fold("(- 2 5)"), sx("(- 3)"))

    def test_two_negative_literals_multiply_to_positive(self):
        self.assertEqual(fold("(* (- 3) (- 2))"), sx("6"))
        self.assertEqual(fold("(- 3)"), sx("(- 3)"))

    def test_double_negation_of_literal(self):
        self.assertEqual(fold("(- (- 3))"), sx("3"))
        self.assertEqual(fold("(+ 4)"), sx("4"))

    def test_bang_on_zero_literal_becomes_one(self):
        self.assertEqual(fold("(! 0)"), sx("1"))
        self.assertEqual(fold("(<= 2 2)"), sx("1"))
        self.assertEqual(fold("(> 2 2)"), sx("0"))

    def test_nested_conjunction_and_disjunction_collapse_to_one(self):
        self.assertEqual(fold("(&& 1 (|| 0 5))"), sx("1"))
        self.assertEqual(fold("(% (^ 2 4) 5)"), sx("1"))

    def test_variables_block_folding_of_their_parents(self):
        self.assertEqual(fold("(+ x (+ 1 2))"), sx("(+ x 3)"))
        self.assertEqual(fold("(* (+ 1 1) (+ x (* 2 3)))"), sx("(* 2 (+ x 6))"))

    def test_variable_tree_is_unchanged(self):
        self.assertEqual(fold("(+ x (* y z))"), sx("(+ x (* y z))"))
        node = XprBinary("+", XprNum(1), XprNum(2), 9)
        self.assertEqual(xpr_fold_constants(node).position, 9)


class ZB(unittest.TestCase):
    def test_division_by_zero_is_left_alone(self):
        self.assertEqual(fold("(/ 1 0)"), sx("(/ 1 0)"))
        self.assertEqual(fold("(% 1 0)"), sx("(% 1 0)"))

    def test_endless_decimal_is_left_alone(self):
        self.assertEqual(fold("(/ 1 3)"), sx("(/ 1 3)"))
        self.assertEqual(fold("(/ 1 8)"), sx("0.125"))

    def test_fractional_exponent_is_left_alone(self):
        self.assertEqual(fold("(^ 2 0.5)"), sx("(^ 2 0.5)"))
        self.assertEqual(fold("(+ (/ 1 0) (+ 1 2))"), sx("(+ (/ 1 0) 3)"))


class ZC(unittest.TestCase):
    def test_zero_on_the_left_of_sum(self):
        self.assertEqual(fold("(+ 0 x)"), sx("x"))
        self.assertEqual(fold("(+ x 0)"), sx("x"))

    def test_zero_on_the_right_of_difference(self):
        self.assertEqual(fold("(- x 0)"), sx("x"))
        self.assertEqual(fold("(- 0 x)"), sx("(- 0 x)"))

    def test_one_on_either_side_of_product(self):
        self.assertEqual(fold("(* 1 x)"), sx("x"))
        self.assertEqual(fold("(* x 1)"), sx("x"))
        self.assertEqual(fold("(* x 0)"), sx("(* x 0)"))

    def test_dividing_by_one_returns_the_dividend(self):
        self.assertEqual(fold("(/ x 1)"), sx("x"))
        self.assertEqual(fold("(/ 1 x)"), sx("(/ 1 x)"))

    def test_raising_to_first_power_returns_the_base(self):
        self.assertEqual(fold("(^ x 1)"), sx("x"))
        self.assertEqual(fold("(^ 1 x)"), sx("(^ 1 x)"))

    def test_identities_combine_with_folding(self):
        self.assertEqual(fold("(+ x (- 3 3))"), sx("x"))
        self.assertEqual(fold("(* (- 5 4) y)"), sx("y"))
        self.assertEqual(fold("(- (- x))"), sx("x"))

    def test_single_negation_of_variable_stays(self):
        self.assertEqual(fold("(- x)"), sx("(- x)"))
        self.assertEqual(fold("(+ x 1)"), sx("(+ x 1)"))
        self.assertEqual(fold("(* x 0.5)"), sx("(* x 0.5)"))

    def test_other_unary_operators_stay(self):
        self.assertEqual(fold("(! x)"), sx("(! x)"))
        self.assertEqual(fold("(+ x)"), sx("(+ x)"))


class ZD(unittest.TestCase):
    def test_calls_are_kept_without_a_registry(self):
        self.assertEqual(fold("(call max 1 2)"), sx("(call max 1 2)"))
        self.assertEqual(fold("(call max 1 (+ 1 2))", REG), sx("3"))

    def test_call_with_variable_argument_stays(self):
        self.assertEqual(fold("(call max x (+ 1 2))", REG), sx("(call max x 3)"))
        self.assertEqual(fold("(call sqrt 2)", REG), sx("(call sqrt 2)"))

    def test_unregistered_name_call_is_kept_verbatim(self):
        self.assertEqual(fold("(call nope 2)", REG), sx("(call nope 2)"))
        reg = xpr_register_function({}, "seven", 0, 0, lambda: 7)
        self.assertEqual(fold("(call seven)", reg), XprNum(7))

    def test_negative_argument_literals_are_used(self):
        self.assertEqual(fold("(call abs (- 4))", REG), sx("4"))
        self.assertEqual(fold("(+ (call abs (- 4)) x)", REG), sx("(+ 4 x)"))
