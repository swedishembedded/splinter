# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
import dataclasses
import unittest
from fractions import Fraction

from exprcalc.errors import XprEvalError
from exprcalc.functions import (
    XprFunction,
    xpr_builtin_registry,
    xpr_call_function,
    xpr_describe_arity,
    xpr_register_function,
)

F = Fraction
REG = xpr_builtin_registry()


def call(name, *args):
    return xpr_call_function(REG, name, list(args))


class ZA(unittest.TestCase):
    def test_standard_library_lists_twelve_known_names(self):
        self.assertEqual(
            sorted(REG),
            ["abs", "ceil", "clamp", "floor", "gcd", "if", "max", "min", "pow", "round", "sign", "sqrt"],
        )
        self.assertTrue(all(entry.name == name for name, entry in REG.items()))

    def test_each_call_returns_a_fresh_registry(self):
        first = xpr_builtin_registry()
        first.clear()
        self.assertIn("abs", xpr_builtin_registry())
        extended = xpr_register_function(REG, "twice", 1, 1, lambda x: x * 2)
        self.assertEqual(xpr_call_function(extended, "twice", [4]), 8)

    def test_registering_leaves_the_original_alone(self):
        xpr_register_function(REG, "twice", 1, 1, lambda x: x * 2)
        self.assertNotIn("twice", REG)
        impl = lambda a, b: a
        entry = xpr_register_function({}, "first", 2, None, impl)["first"]
        self.assertEqual(entry, XprFunction("first", 2, None, impl))

    def test_registering_a_taken_name_fails(self):
        with self.assertRaisesRegex(ValueError, "^function 'abs' is already registered$"):
            xpr_register_function(REG, "abs", 1, 1, abs)


class ZB(unittest.TestCase):
    def test_unknown_function_is_reported_by_name(self):
        with self.assertRaisesRegex(XprEvalError, "^unknown function 'nope'$"):
            call("nope", 1)
        with self.assertRaisesRegex(XprEvalError, "^function 'abs' expects 1 argument, got 0$"):
            call("abs")

    def test_surplus_argument_is_reported_with_counts(self):
        with self.assertRaisesRegex(XprEvalError, "^function 'gcd' expects 2 arguments, got 3$"):
            call("gcd", 1, 2, 3)
        with self.assertRaisesRegex(XprEvalError, "^function 'min' expects at least 1, got 0$"):
            call("min")
        with self.assertRaisesRegex(XprEvalError, "^function 'max' expects at least 1, got 0$"):
            call("max")

    def test_function_entries_are_immutable(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            REG["abs"].minimum = 0
        for name in ("abs", "sqrt", "floor", "ceil", "round", "sign"):
            with self.assertRaisesRegex(XprEvalError, "expects 1 argument, got 0", msg=name):
                call(name)

    def test_ternary_builtins_reject_two_arguments(self):
        for name in ("clamp", "if"):
            with self.assertRaisesRegex(XprEvalError, "expects 3 arguments, got 2", msg=name):
                call(name, 1, 2)
        for name in ("gcd", "pow"):
            with self.assertRaisesRegex(XprEvalError, "expects 2 arguments, got 1", msg=name):
                call(name, 1)

    def test_variadic_minimum_accepts_a_single_value(self):
        self.assertEqual(call("min", 5), 5)
        self.assertEqual(call("max", 5), 5)
        self.assertEqual(call("max", *range(100)), 99)

    def test_exact_arity_boundaries_are_accepted(self):
        self.assertEqual(call("gcd", 12, 18), 6)
        extended = xpr_register_function({}, "half", 1, 1, lambda x: F(x) / 2)
        result = xpr_call_function(extended, "half", [4])
        self.assertEqual(result, 2)
        self.assertIs(type(result), int)

    def test_ranged_arity_accepts_both_ends(self):
        reg = xpr_register_function({}, "span", 1, 3, lambda *a: len(a))
        self.assertEqual(xpr_call_function(reg, "span", [1]), 1)
        self.assertEqual(xpr_call_function(reg, "span", [1, 2, 3]), 3)
        reg = xpr_register_function({}, "span", 1, 3, lambda *a: len(a))
        with self.assertRaisesRegex(XprEvalError, "expects 1 to 3 arguments, got 4"):
            xpr_call_function(reg, "span", [1, 2, 3, 4])
        with self.assertRaisesRegex(XprEvalError, "expects 1 to 3 arguments, got 0"):
            xpr_call_function(reg, "span", [])


class ZC(unittest.TestCase):
    def test_single_argument_is_singular(self):
        self.assertEqual(xpr_describe_arity(XprFunction("f", 1, 1, abs)), "1 argument")
        self.assertEqual(xpr_describe_arity(XprFunction("f", 3, 3, abs)), "3 arguments")

    def test_zero_arguments_are_plural(self):
        self.assertEqual(xpr_describe_arity(XprFunction("f", 0, 0, abs)), "0 arguments")
        self.assertEqual(xpr_describe_arity(XprFunction("f", 2, None, abs)), "at least 2")

    def test_bounded_functions_name_both_limits(self):
        self.assertEqual(xpr_describe_arity(XprFunction("f", 1, 2, abs)), "1 to 2 arguments")


class ZD(unittest.TestCase):
    def test_abs_returns_positive_fraction_magnitude(self):
        self.assertEqual(call("abs", F(-3, 2)), F(3, 2))
        self.assertEqual(call("min", 4, F(1, 2), 3), F(1, 2))

    def test_max_scans_every_argument_for_largest(self):
        self.assertEqual(call("max", 4, F(1, 2), 9, 3), 9)
        self.assertEqual(call("sqrt", F(25, 4)), F(5, 2))

    def test_sqrt_of_non_square_fails(self):
        with self.assertRaisesRegex(XprEvalError, "square root is not exact"):
            call("sqrt", 2)
        self.assertEqual(call("floor", F(-3, 2)), -2)
        self.assertEqual(call("floor", F(7, 2)), 3)

    def test_ceil_moves_toward_positive_infinity(self):
        self.assertEqual(call("ceil", F(-3, 2)), -1)
        self.assertEqual(call("ceil", F(7, 2)), 4)
        self.assertEqual(call("round", F(5, 2)), 3)
        self.assertEqual(call("round", F(-5, 2)), -2)

    def test_round_below_half_goes_down(self):
        self.assertEqual(call("round", F(7, 3)), 2)
        self.assertEqual(call("round", F(8, 3)), 3)
        self.assertEqual(call("sign", 1), 1)
        self.assertEqual(call("sign", F(1, 2)), 1)

    def test_sign_of_each_kind(self):
        self.assertEqual([call("sign", v) for v in (-F(1, 2), 0, 9)], [-1, 0, 1])
        self.assertEqual(call("pow", 2, -3), F(1, 8))


class ZE(unittest.TestCase):
    def test_gcd_of_whole_numbers(self):
        self.assertEqual(call("gcd", 84, 36), 12)
        self.assertEqual(call("gcd", -4, 6), 2)

    def test_gcd_accepts_whole_fractions(self):
        self.assertEqual(call("gcd", F(8, 2), F(6, 1)), 2)
        with self.assertRaisesRegex(XprEvalError, "^gcd needs whole numbers$"):
            call("gcd", F(1, 2), 4)

    def test_gcd_rejects_fractional_right(self):
        with self.assertRaisesRegex(XprEvalError, "^gcd needs whole numbers$"):
            call("gcd", 4, F(1, 2))
        self.assertEqual(call("clamp", 5, 0, 10), 5)

    def test_clamp_returns_lower_bound_for_small_input(self):
        self.assertEqual(call("clamp", -5, 0, 10), 0)
        self.assertEqual(call("clamp", 15, 0, 10), 10)

    def test_clamp_with_equal_bounds(self):
        self.assertEqual(call("clamp", 3, 4, 4), 4)
        with self.assertRaisesRegex(XprEvalError, "^clamp bounds are reversed$"):
            call("clamp", 3, 5, 4)


class ZF(unittest.TestCase):
    def test_if_selects_then_branch_for_nonzero(self):
        self.assertEqual(call("if", 1, 10, 20), 10)
        self.assertEqual(call("if", F(1, 5), 10, 20), 10)
        self.assertEqual(call("if", 0, 10, 20), 20)
