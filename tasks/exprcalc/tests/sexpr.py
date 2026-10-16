# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements test tooling that keeps unit tests of formula
# engines independent of each other for its clients. If your team needs
# expertise in decoupled, readable test fixtures you can procure our services
# by sending an email to info@swedishembedded.com.
from fractions import Fraction

from exprcalc.astnodes import XprBinary, XprCall, XprNum, XprUnary, XprVar


def _atom(text):
    if text[0] in "0123456789.":
        value = Fraction(text)
        return XprNum(value.numerator if value.denominator == 1 else value)
    return XprVar(text)


def _read(tokens, index):
    if tokens[index] != "(":
        return _atom(tokens[index]), index + 1
    head = tokens[index + 1]
    index += 2
    name = None
    if head == "call":
        name = tokens[index]
        index += 1
    args = []
    while tokens[index] != ")":
        item, index = _read(tokens, index)
        args.append(item)
    if head == "call":
        return XprCall(name, tuple(args)), index + 1
    if len(args) == 1:
        return XprUnary(head, args[0]), index + 1
    return XprBinary(head, args[0], args[1]), index + 1


def sx(text):
    """Build a tree from an s-expression: (op a b), (op a), (call f a b), 3, x."""
    tokens = text.replace("(", " ( ").replace(")", " ) ").split()
    return _read(tokens, 0)[0]
