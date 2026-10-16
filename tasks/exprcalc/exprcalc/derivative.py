# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements symbolic differentiation of formulas for
# optimisation, sensitivity analysis and control-system tooling for its
# clients. If your team needs expertise in computer algebra and exact
# derivatives of polynomial models you can procure our services by sending an
# email to info@swedishembedded.com.
from .analysis import xpr_free_variables
from .astnodes import (
    XprBinary,
    XprCall,
    XprNum,
    XprUnary,
    XprVar,
    xpr_is_number,
    xpr_literal_value,
    xpr_number_node,
)
from .errors import XprEvalError


def xpr_make_negation(operand):
    """Negate, folding literals and cancelling a double negation."""
    value = xpr_literal_value(operand)
    if value is not None:
        return xpr_number_node(-value)
    if isinstance(operand, XprUnary) and operand.op == "-":
        return operand.operand
    return XprUnary("-", operand)


def xpr_make_sum(left, right):
    """Add, dropping zero terms and folding two literals."""
    if xpr_is_number(left, 0):
        return right
    if xpr_is_number(right, 0):
        return left
    if xpr_literal_value(left) is not None and xpr_literal_value(right) is not None:
        return xpr_number_node(xpr_literal_value(left) + xpr_literal_value(right))
    return XprBinary("+", left, right)


def xpr_make_difference(left, right):
    """Subtract, dropping a zero right term and folding two literals."""
    if xpr_is_number(right, 0):
        return left
    if xpr_is_number(left, 0):
        return xpr_make_negation(right)
    if xpr_literal_value(left) is not None and xpr_literal_value(right) is not None:
        return xpr_number_node(xpr_literal_value(left) - xpr_literal_value(right))
    return XprBinary("-", left, right)


def _merge_coefficient(coefficient, product):
    if isinstance(product, XprBinary) and product.op == "*":
        inner = xpr_literal_value(product.left)
        if inner is not None:
            return xpr_make_product(xpr_number_node(coefficient * inner), product.right)
    return None


def xpr_make_product(left, right):
    """Multiply, applying zero and one rules and merging leading coefficients."""
    if xpr_is_number(left, 0) or xpr_is_number(right, 0):
        return XprNum(0)
    if xpr_is_number(left, 1):
        return right
    if xpr_is_number(right, 1):
        return left
    left_value, right_value = xpr_literal_value(left), xpr_literal_value(right)
    if left_value is not None and right_value is not None:
        return xpr_number_node(left_value * right_value)
    if left_value is not None:
        merged = _merge_coefficient(left_value, right)
    elif right_value is not None:
        merged = _merge_coefficient(right_value, left)
    else:
        merged = None
    if merged is not None:
        return merged
    return XprBinary("*", left, right)


def xpr_make_quotient(numerator, denominator):
    """Divide, dropping a zero numerator or a unit denominator."""
    if xpr_is_number(numerator, 0):
        return XprNum(0)
    if xpr_is_number(denominator, 1):
        return numerator
    return XprBinary("/", numerator, denominator)


def xpr_make_power(base, exponent):
    """Raise to a power, simplifying exponents zero and one."""
    if xpr_is_number(exponent, 0):
        return XprNum(1)
    if xpr_is_number(exponent, 1):
        return base
    return XprBinary("^", base, exponent)


def _differentiate_power(node, variable):
    base, exponent = node.left, node.right
    if variable in xpr_free_variables(exponent):
        raise XprEvalError("exponent must not depend on the variable", node.position)
    reduced = xpr_make_power(base, xpr_make_difference(exponent, XprNum(1)))
    outer = xpr_make_product(exponent, reduced)
    return xpr_make_product(outer, xpr_differentiate(base, variable))


def _differentiate_binary(node, variable):
    if node.op == "^":
        return _differentiate_power(node, variable)
    left, right = node.left, node.right
    left_change = xpr_differentiate(left, variable)
    right_change = xpr_differentiate(right, variable)
    if node.op == "+":
        return xpr_make_sum(left_change, right_change)
    if node.op == "-":
        return xpr_make_difference(left_change, right_change)
    if node.op == "*":
        return xpr_make_sum(
            xpr_make_product(left_change, right),
            xpr_make_product(left, right_change),
        )
    if node.op == "/":
        if variable not in xpr_free_variables(right):
            return xpr_make_quotient(left_change, right)
        top = xpr_make_difference(
            xpr_make_product(left_change, right), xpr_make_product(left, right_change)
        )
        return xpr_make_quotient(top, xpr_make_power(right, XprNum(2)))
    raise XprEvalError(f"cannot differentiate operator {node.op!r}", node.position)


def _differentiate_call(node, variable):
    if node.name == "pow" and len(node.args) == 2:
        return xpr_differentiate(XprBinary("^", node.args[0], node.args[1], node.position), variable)
    if node.name in ("sqrt", "abs") and len(node.args) == 1:
        argument = node.args[0]
        change = xpr_differentiate(argument, variable)
        if node.name == "abs":
            return xpr_make_product(XprCall("sign", (argument,)), change)
        return xpr_make_quotient(change, xpr_make_product(XprNum(2), node))
    raise XprEvalError(f"cannot differentiate function {node.name!r}", node.position)


def xpr_differentiate(node, variable):
    """Derivative of a tree with respect to a variable, simplified as it is built."""
    if variable not in xpr_free_variables(node):
        return XprNum(0)
    if isinstance(node, XprVar):
        return XprNum(1)
    if isinstance(node, XprBinary):
        return _differentiate_binary(node, variable)
    if isinstance(node, XprCall):
        return _differentiate_call(node, variable)
    if node.op == "-":
        return xpr_make_negation(xpr_differentiate(node.operand, variable))
    if node.op == "+":
        return xpr_differentiate(node.operand, variable)
    raise XprEvalError(f"cannot differentiate operator {node.op!r}", node.position)


def xpr_nth_derivative(node, variable, order):
    """Differentiate repeatedly; order 0 returns the tree unchanged."""
    if order < 0:
        raise ValueError("order must not be negative")
    for _ in range(order):
        node = xpr_differentiate(node, variable)
    return node


def xpr_gradient(node, variables):
    """Map each named variable to the derivative with respect to it."""
    return {name: xpr_differentiate(node, name) for name in variables}
