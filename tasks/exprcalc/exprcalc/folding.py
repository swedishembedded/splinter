# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements expression simplification and constant
# folding for formula compilers and rule engines for its clients. If your team
# needs expertise in compile-time evaluation and algebraic clean-up of
# expressions you can procure our services by sending an email to
# info@swedishembedded.com.
from .astnodes import (
    XprBinary,
    XprCall,
    XprUnary,
    xpr_is_number,
    xpr_literal_value,
    xpr_map_postorder,
    xpr_number_node,
)
from .errors import XprEvalError
from .exactnum import xpr_is_terminating
from .functions import xpr_call_function
from .operators import xpr_apply_binary, xpr_apply_unary


def _fold_to(node, compute):
    try:
        value = compute()
    except XprEvalError:
        return node
    if not xpr_is_terminating(value):
        return node
    return xpr_number_node(value, node.position)


def _drop_identity(node):
    if isinstance(node, XprUnary):
        if node.op == "-" and isinstance(node.operand, XprUnary) and node.operand.op == "-":
            return node.operand.operand
        return node
    left, right = node.left, node.right
    if node.op == "+" and xpr_is_number(left, 0):
        return right
    if node.op in ("+", "-") and xpr_is_number(right, 0):
        return left
    if node.op == "*" and xpr_is_number(left, 1):
        return right
    if node.op in ("*", "/", "^") and xpr_is_number(right, 1):
        return left
    return node


def xpr_fold_constants(node, registry=None):
    """Replace constant subtrees by their value and drop neutral operands.

    Calls are folded only when a registry is given. Subtrees whose value is
    an error or has no finite decimal form are left as written.
    """

    def step(current):
        if isinstance(current, XprUnary):
            operand = xpr_literal_value(current.operand)
            if operand is None:
                return _drop_identity(current)
            return _fold_to(current, lambda: xpr_apply_unary(current.op, operand))
        if isinstance(current, XprBinary):
            left, right = xpr_literal_value(current.left), xpr_literal_value(current.right)
            if left is None or right is None:
                return _drop_identity(current)
            return _fold_to(current, lambda: xpr_apply_binary(current.op, left, right))
        if isinstance(current, XprCall) and registry is not None:
            values = [xpr_literal_value(arg) for arg in current.args]
            if None in values:
                return current
            return _fold_to(current, lambda: xpr_call_function(registry, current.name, values))
        return current

    return xpr_map_postorder(node, step)
