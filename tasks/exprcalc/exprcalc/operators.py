# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements operator tables and exact operator semantics
# for formula and rule engines for its clients. If your team needs expertise
# in precedence, associativity and well defined arithmetic and boolean
# operators you can procure our services by sending an email to
# info@swedishembedded.com.
import operator

from .errors import XprEvalError
from .exactnum import (
    xpr_divide_exact,
    xpr_normalize_number,
    xpr_power_exact,
    xpr_remainder_exact,
)

XPR_UNARY_PRECEDENCE = 7
XPR_BINARY_PRECEDENCE = {
    "||": 1,
    "&&": 2,
    "==": 3, "!=": 3,
    "<": 4, "<=": 4, ">": 4, ">=": 4,
    "+": 5, "-": 5,
    "*": 6, "/": 6, "%": 6,
    "^": 8,
}
XPR_LOWEST_PRECEDENCE = min(XPR_BINARY_PRECEDENCE.values())
XPR_UNARY_OPERATORS = ("-", "+", "!")

_ARITHMETIC = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": xpr_divide_exact,
    "%": xpr_remainder_exact,
    "^": xpr_power_exact,
}
_COMPARISONS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


def xpr_is_binary_operator(op):
    """True for an operator that takes two operands."""
    return op in XPR_BINARY_PRECEDENCE


def xpr_is_unary_operator(op):
    """True for an operator that takes one operand."""
    return op in XPR_UNARY_OPERATORS


def xpr_binary_precedence(op):
    """Binding strength of a binary operator; larger binds tighter."""
    return XPR_BINARY_PRECEDENCE[op]


def xpr_is_right_associative(op):
    """Only the power operator groups to the right."""
    return op == "^"


def xpr_apply_unary(op, value):
    """Apply a unary operator; the logical not yields 1 or 0."""
    if op == "-":
        return xpr_normalize_number(-value)
    if op == "+":
        return value
    if op == "!":
        return int(value == 0)
    raise XprEvalError(f"unknown unary operator {op!r}")


def xpr_apply_binary(op, left, right):
    """Apply a binary operator; comparisons and logic yield 1 or 0."""
    if op in _ARITHMETIC:
        return xpr_normalize_number(_ARITHMETIC[op](left, right))
    if op in _COMPARISONS:
        return int(_COMPARISONS[op](left, right))
    if op == "&&":
        return int(left != 0 and right != 0)
    if op == "||":
        return int(left != 0 or right != 0)
    raise XprEvalError(f"unknown binary operator {op!r}")
