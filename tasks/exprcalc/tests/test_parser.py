# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
import unittest
from fractions import Fraction

from exprcalc.astnodes import XprBinary, XprCall, XprNum, XprUnary, XprVar
from exprcalc.errors import XprSyntaxError
from exprcalc.parser import XPR_MAX_NESTING, XprParser, xpr_parse_text, xpr_parse_tokens
from exprcalc.tokenizer import XprToken, xpr_scan_tokens
from .sexpr import sx


def failure(source):
    with unittest.TestCase().assertRaises(XprSyntaxError) as caught:
        xpr_parse_text(source)
    return caught.exception.message, caught.exception.position


class ZA(unittest.TestCase):
    def test_digits_become_a_number_node(self):
        self.assertEqual(xpr_parse_text("42"), XprNum(42))
        self.assertEqual(xpr_parse_text("0.125"), XprNum(Fraction(1, 8)))
        self.assertEqual(xpr_parse_text("speed"), XprVar("speed"))
        node = xpr_parse_text("  7")
        self.assertEqual(node.position, 2)

    def test_parentheses_do_not_add_nodes(self):
        self.assertEqual(xpr_parse_text("((x))"), XprVar("x"))


class ZB(unittest.TestCase):
    def test_product_binds_tighter_than_sum(self):
        self.assertEqual(xpr_parse_text("1+2*3"), sx("(+ 1 (* 2 3))"))
        self.assertEqual(xpr_parse_text("1*2+3"), sx("(+ (* 1 2) 3)"))
        self.assertEqual(xpr_parse_text("(1+2)*3"), sx("(* (+ 1 2) 3)"))
        self.assertEqual(xpr_parse_text("8-3-2"), sx("(- (- 8 3) 2)"))

    def test_division_is_left_associative(self):
        self.assertEqual(xpr_parse_text("8/4/2"), sx("(/ (/ 8 4) 2)"))
        self.assertEqual(xpr_parse_text("7%4*2"), sx("(* (% 7 4) 2)"))
        self.assertEqual(xpr_parse_text("2^3^2"), sx("(^ 2 (^ 3 2))"))
        self.assertEqual(xpr_parse_text("2*3^2"), sx("(* 2 (^ 3 2))"))

    def test_comparison_binds_looser_than_sum(self):
        self.assertEqual(xpr_parse_text("1+2<4"), sx("(< (+ 1 2) 4)"))
        self.assertEqual(xpr_parse_text("a<b==c>d"), sx("(== (< a b) (> c d))"))
        self.assertEqual(xpr_parse_text("a||b&&c"), sx("(|| a (&& b c))"))
        self.assertEqual(xpr_parse_text("a<1&&b>2"), sx("(&& (< a 1) (> b 2))"))

    def test_chained_comparison_groups_left(self):
        self.assertEqual(xpr_parse_text("1<2<3"), sx("(< (< 1 2) 3)"))


class ZC(unittest.TestCase):
    def test_leading_minus_makes_a_unary_node(self):
        self.assertEqual(xpr_parse_text("-x"), sx("(- x)"))
        self.assertEqual(xpr_parse_text("-2^2"), sx("(- (^ 2 2))"))
        self.assertEqual(xpr_parse_text("-a*b"), sx("(* (- a) b)"))
        self.assertEqual(xpr_parse_text("2^-3"), sx("(^ 2 (- 3))"))

    def test_exponent_negation_stops_before_product(self):
        self.assertEqual(xpr_parse_text("2^-3*4"), sx("(* (^ 2 (- 3)) 4)"))
        self.assertEqual(xpr_parse_text("!-+x"), sx("(! (- (+ x)))"))
        self.assertEqual(xpr_parse_text("!a==b"), sx("(== (! a) b)"))
        self.assertEqual(xpr_parse_text("1--2"), sx("(- 1 (- 2))"))


class ZD(unittest.TestCase):
    def test_empty_argument_list_gives_empty_tuple(self):
        self.assertEqual(xpr_parse_text("now()"), XprCall("now", ()))
        self.assertEqual(xpr_parse_text("abs(x)"), sx("(call abs x)"))
        self.assertEqual(xpr_parse_text("f(1+2, g(3))"), sx("(call f (+ 1 2) (call g 3))"))
        self.assertIsInstance(xpr_parse_text("f(1,2)").args, tuple)

    def test_call_position_is_the_name_start(self):
        self.assertEqual(xpr_parse_text("1 + max(2)").right.position, 4)
        self.assertEqual(xpr_parse_text("f(1)^2"), sx("(^ (call f 1) 2)"))
        self.assertEqual(xpr_parse_text("x (1)"), sx("(call x 1)"))


class ZE(unittest.TestCase):
    def test_binary_node_sits_at_its_operator(self):
        self.assertEqual(xpr_parse_text("12 + 3").position, 3)
        self.assertEqual(xpr_parse_text("  -x").position, 2)


class ZF(unittest.TestCase):
    def test_empty_input_ends_unexpectedly(self):
        self.assertEqual(failure(""), ("unexpected end of input", 0))
        self.assertEqual(failure("1 +"), ("unexpected end of input", 3))
        self.assertEqual(failure("* 2"), ("unexpected '*'", 0))
        self.assertEqual(failure("1 2"), ("unexpected '2'", 2))

    def test_group_missing_closing_bracket_expects_one(self):
        self.assertEqual(failure("(1+2"), ("expected ')'", 4))
        self.assertEqual(failure("1)"), ("unexpected ')'", 1))
        self.assertEqual(failure("f(1, 2"), ("expected ')'", 6))
        self.assertEqual(failure("f(1,)"), ("unexpected ')'", 4))

    def test_adjacent_arguments_expect_closing_bracket(self):
        self.assertEqual(failure("f(1 2)"), ("expected ')'", 4))
        self.assertEqual(failure("1, 2"), ("unexpected ','", 1))
        self.assertEqual(failure("1 + * 2"), ("unexpected '*'", 4))
        self.assertEqual(failure("()"), ("unexpected ')'", 1))

    def test_lexical_errors_pass_through(self):
        self.assertEqual(failure("1 $ 2"), ("unexpected character '$'", 2))


class ZG(unittest.TestCase):
    def test_nesting_limit_constant_is_fifty(self):
        self.assertEqual(XPR_MAX_NESTING, 50)
        self.assertEqual(xpr_parse_text("(" * 49 + "1" + ")" * 49), XprNum(1))
        with self.assertRaises(XprSyntaxError) as caught:
            xpr_parse_text("(" * 50 + "1" + ")" * 50)
        self.assertEqual(caught.exception.message, "expression nested too deeply")
        self.assertEqual(caught.exception.position, 50)
        with self.assertRaisesRegex(XprSyntaxError, "nested too deeply"):
            xpr_parse_text("(1) + (2) + " + "(" * 50 + "1" + ")" * 50)

    def test_lowest_precedence_operator_inside_brackets(self):
        self.assertEqual(xpr_parse_text("(a || b) && c"), sx("(&& (|| a b) c)"))
        self.assertEqual(xpr_parse_text("f(a || b, c || d)"), sx("(call f (|| a b) (|| c d))"))
        node = xpr_parse_text("+".join(["1"] * 100))
        self.assertIsInstance(node, XprBinary)
        self.assertIsInstance(xpr_parse_text("-" * 49 + "1"), XprUnary)
        with self.assertRaisesRegex(XprSyntaxError, "nested too deeply"):
            xpr_parse_text("-" * 50 + "1")


class ZH(unittest.TestCase):
    def test_parse_tokens_matches_parse_text(self):
        self.assertEqual(xpr_parse_tokens(xpr_scan_tokens("a*(b+1)")), xpr_parse_text("a*(b+1)"))
        tokens = [XprToken("number", "5", 0), XprToken("end", "", 1)]
        self.assertEqual(xpr_parse_tokens(tokens), XprNum(5))
        parser = XprParser(xpr_scan_tokens("x"))
        self.assertEqual(parser.peek().text, "x")
        self.assertEqual(parser.advance().text, "x")
        self.assertEqual(parser.peek().kind, "end")
