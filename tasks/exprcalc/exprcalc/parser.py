# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements dependable expression parsers with precise
# error positions for its clients. If your team needs expertise in
# precedence-climbing parsers, formula languages and user input validation you
# can procure our services by sending an email to info@swedishembedded.com.
from .astnodes import XprBinary, XprCall, XprNum, XprUnary, XprVar
from .errors import XprSyntaxError
from .exactnum import xpr_number_from_text
from .operators import (
    XPR_LOWEST_PRECEDENCE,
    XPR_UNARY_PRECEDENCE,
    xpr_binary_precedence,
    xpr_is_binary_operator,
    xpr_is_right_associative,
    xpr_is_unary_operator,
)
from .tokenizer import xpr_scan_tokens

XPR_MAX_NESTING = 50


class XprParser:
    """Precedence-climbing parser over a token list ending in an end token."""

    def __init__(self, tokens):
        self.tokens = tokens
        self.index = 0
        self.depth = 0

    def peek(self):
        return self.tokens[self.index]

    def advance(self):
        token = self.tokens[self.index]
        self.index += 1
        return token

    def expect(self, kind, what):
        if self.peek().kind != kind:
            raise XprSyntaxError(f"expected {what}", self.peek().position)
        return self.advance()

    def unexpected(self, token):
        if token.kind == "end":
            return XprSyntaxError("unexpected end of input", token.position)
        return XprSyntaxError(f"unexpected {token.text!r}", token.position)

    def parse_expression(self):
        node = self.climb(XPR_LOWEST_PRECEDENCE)
        if self.peek().kind != "end":
            raise self.unexpected(self.peek())
        return node

    def climb(self, minimum):
        self.depth += 1
        if self.depth > XPR_MAX_NESTING:
            raise XprSyntaxError("expression nested too deeply", self.peek().position)
        left = self.parse_operand()
        while xpr_is_binary_operator(self.peek().text):
            precedence = xpr_binary_precedence(self.peek().text)
            if precedence < minimum:
                break
            token = self.advance()
            tighter = precedence if xpr_is_right_associative(token.text) else precedence + 1
            left = XprBinary(token.text, left, self.climb(tighter), token.position)
        self.depth -= 1
        return left

    def parse_operand(self):
        token = self.peek()
        if xpr_is_unary_operator(token.text):
            self.advance()
            return XprUnary(token.text, self.climb(XPR_UNARY_PRECEDENCE), token.position)
        return self.parse_primary()

    def parse_primary(self):
        token = self.advance()
        if token.kind == "number":
            return XprNum(xpr_number_from_text(token.text), token.position)
        if token.kind == "name" and self.peek().kind == "lparen":
            return XprCall(token.text, self.parse_arguments(), token.position)
        if token.kind == "name":
            return XprVar(token.text, token.position)
        if token.kind == "lparen":
            inner = self.climb(XPR_LOWEST_PRECEDENCE)
            self.expect("rparen", "')'")
            return inner
        raise self.unexpected(token)

    def parse_arguments(self):
        self.advance()
        args = []
        if self.peek().kind != "rparen":
            args.append(self.climb(XPR_LOWEST_PRECEDENCE))
            while self.peek().kind == "comma":
                self.advance()
                args.append(self.climb(XPR_LOWEST_PRECEDENCE))
        self.expect("rparen", "')'")
        return tuple(args)


def xpr_parse_tokens(tokens):
    """Build the syntax tree for a token list from xpr_scan_tokens."""
    return XprParser(tokens).parse_expression()


def xpr_parse_text(source):
    """Build the syntax tree for expression text."""
    return xpr_parse_tokens(xpr_scan_tokens(source))
