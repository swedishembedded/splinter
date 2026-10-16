# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
import dataclasses
import unittest

from exprcalc.errors import XprSyntaxError
from exprcalc.tokenizer import (
    XprToken,
    xpr_check_number_text,
    xpr_scan_tokens,
    xpr_token_is,
)


def kinds(source):
    return [token.kind for token in xpr_scan_tokens(source)]


def texts(source):
    return [token.text for token in xpr_scan_tokens(source)]


class ZA(unittest.TestCase):
    def test_empty_source_gives_only_end_token(self):
        self.assertEqual(xpr_scan_tokens(""), [XprToken("end", "", 0)])
        self.assertEqual(xpr_scan_tokens("x ")[-1], XprToken("end", "", 2))

    def test_whitespace_is_skipped_everywhere(self):
        self.assertEqual(texts(" 1\t+\r\n2 "), ["1", "+", "2", ""])
        positions = [token.position for token in xpr_scan_tokens("ab + 12")]
        self.assertEqual(positions, [0, 3, 5, 7])

    def test_plus_expression_has_three_kinds_and_end(self):
        self.assertEqual(kinds("1+x"), ["number", "op", "name", "end"])
        self.assertEqual(
            kinds("f(a,b)"),
            ["name", "lparen", "name", "comma", "name", "rparen", "end"],
        )

    def test_token_fields_cannot_be_reassigned(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            XprToken("op", "+", 0).text = "-"
        self.assertEqual(XprToken("op", "+", 4), XprToken("op", "+", 4))
        self.assertNotEqual(XprToken("op", "+", 4), XprToken("op", "+", 5))


class ZB(unittest.TestCase):
    def test_every_single_character_operator(self):
        self.assertEqual(texts("+-*/%^<>!")[:-1], list("+-*/%^<>!"))
        self.assertEqual(set(kinds("+-*/%^<>!")[:-1]), {"op"})
        self.assertEqual(texts("<= >= == != && ||")[:-1], ["<=", ">=", "==", "!=", "&&", "||"])

    def test_less_equal_beats_less_then_assign(self):
        self.assertEqual(texts("a<=b"), ["a", "<=", "b", ""])
        self.assertEqual(texts("a!=b")[1], "!=")

    def test_bang_before_name_is_a_lone_not(self):
        self.assertEqual(texts("!a"), ["!", "a", ""])
        self.assertEqual(texts("!!a"), ["!", "!", "a", ""])

    def test_lone_equals_sign_is_rejected(self):
        with self.assertRaises(XprSyntaxError) as caught:
            xpr_scan_tokens("a = b")
        self.assertEqual(caught.exception.message, "unexpected character '='")
        self.assertEqual(caught.exception.position, 2)
        with self.assertRaisesRegex(XprSyntaxError, "unexpected character '&'"):
            xpr_scan_tokens("a & b")

    def test_lone_pipe_is_rejected(self):
        with self.assertRaisesRegex(XprSyntaxError, "unexpected character '[|]'"):
            xpr_scan_tokens("a | b")
        with self.assertRaises(XprSyntaxError) as caught:
            xpr_scan_tokens("12 + $")
        self.assertEqual(caught.exception.position, 5)

    def test_non_ascii_letter_is_rejected(self):
        with self.assertRaisesRegex(XprSyntaxError, "unexpected character"):
            xpr_scan_tokens("\u00e9")


class ZC(unittest.TestCase):
    def test_leading_zeros_are_preserved_in_text(self):
        self.assertEqual(texts("007")[0], "007")
        self.assertEqual(texts("3.25")[0], "3.25")

    def test_number_may_start_with_a_dot(self):
        self.assertEqual(texts(".5 + 1"), [".5", "+", "1", ""])
        self.assertEqual(texts("1.5e-3")[0], "1.5e-3")

    def test_exponent_with_plus_and_capital(self):
        self.assertEqual(texts("2E+10")[0], "2E+10")
        self.assertEqual(texts("4e2")[0], "4e2")

    def test_trailing_dot_is_malformed(self):
        with self.assertRaises(XprSyntaxError) as caught:
            xpr_scan_tokens("1 + 2.")
        self.assertEqual((caught.exception.message, caught.exception.position), ("malformed number", 4))
        with self.assertRaisesRegex(XprSyntaxError, "^malformed number"):
            xpr_scan_tokens("1.e3")

    def test_exponent_without_digits_is_malformed(self):
        with self.assertRaises(XprSyntaxError) as caught:
            xpr_scan_tokens("5e")
        self.assertEqual((caught.exception.message, caught.exception.position), ("malformed exponent", 0))
        with self.assertRaisesRegex(XprSyntaxError, "^malformed exponent"):
            xpr_scan_tokens("5e+ 1")

    def test_second_dot_starts_another_number(self):
        self.assertEqual(texts("1.2.3"), ["1.2", ".3", ""])
        with self.assertRaisesRegex(XprSyntaxError, "unexpected character '[.]'"):
            xpr_scan_tokens("1 . 2")


    def test_letters_after_digits_start_a_new_token(self):
        self.assertEqual(kinds("12ab"), ["number", "name", "end"])


class ZD(unittest.TestCase):
    def test_name_with_digits_and_underscores(self):
        self.assertEqual(texts("_a1_b2")[0], "_a1_b2")
        self.assertEqual(texts("abc+d"), ["abc", "+", "d", ""])








class ZE(unittest.TestCase):
    def test_valid_literals_pass_the_check(self):
        for text in ("1", "10.5", ".5", "2e3", "2.5E-3", "0.0e+0"):
            self.assertIsNone(xpr_check_number_text(text, 0), text)
        with self.assertRaises(XprSyntaxError) as caught:
            xpr_check_number_text("7.", 3)
        self.assertEqual((caught.exception.message, caught.exception.position), ("malformed number", 3))

    def test_missing_exponent_digits_are_malformed(self):
        with self.assertRaises(XprSyntaxError) as caught:
            xpr_check_number_text("7e-", 6)
        self.assertEqual((caught.exception.message, caught.exception.position), ("malformed exponent", 6))
        with self.assertRaisesRegex(XprSyntaxError, "^malformed number"):
            xpr_check_number_text("7.e", 0)


class ZF(unittest.TestCase):
    def test_kind_match_without_text(self):
        self.assertTrue(xpr_token_is(XprToken("op", "+", 0), "op"))
        self.assertFalse(xpr_token_is(XprToken("name", "+", 0), "op"))

    def test_text_must_match_when_given(self):
        token = XprToken("op", "+", 0)
        self.assertTrue(xpr_token_is(token, "op", "+"))
        self.assertFalse(xpr_token_is(token, "op", "-"))
        self.assertFalse(xpr_token_is(XprToken("end", "", 0), "end", "x"))
        self.assertTrue(xpr_token_is(XprToken("end", "", 0), "end", ""))
