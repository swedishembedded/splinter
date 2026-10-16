# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements readable canonical formatting of formulas and
# syntax trees for its clients. If your team needs expertise in pretty
# printing, round-trip safe serialisation and debugging output for expression
# languages you can procure our services by sending an email to
# info@swedishembedded.com.
from .astnodes import XprBinary, XprCall, XprNum, XprUnary, XprVar, xpr_child_nodes
from .exactnum import xpr_number_to_text
from .operators import (
    XPR_UNARY_PRECEDENCE,
    xpr_binary_precedence,
    xpr_is_right_associative,
)

_ATOM_PRECEDENCE = 9


def _precedence(node):
    if isinstance(node, XprBinary):
        return xpr_binary_precedence(node.op)
    if isinstance(node, XprUnary) or (isinstance(node, XprNum) and node.value < 0):
        return XPR_UNARY_PRECEDENCE
    return _ATOM_PRECEDENCE


def _needs_parentheses(child, parent_op, on_right):
    parent = xpr_binary_precedence(parent_op)
    child_precedence = _precedence(child)
    if xpr_is_right_associative(parent_op):
        return child_precedence < XPR_UNARY_PRECEDENCE if on_right else child_precedence <= parent
    return child_precedence <= parent if on_right else child_precedence < parent


def _render_child(child, parent_op, on_right):
    text = xpr_render_minimal(child)
    return f"({text})" if _needs_parentheses(child, parent_op, on_right) else text


def xpr_render_minimal(node):
    """Source text with only the parentheses needed to keep the tree's meaning."""
    if isinstance(node, XprNum):
        return xpr_number_to_text(node.value)
    if isinstance(node, XprVar):
        return node.name
    if isinstance(node, XprUnary):
        operand = xpr_render_minimal(node.operand)
        if _precedence(node.operand) < XPR_UNARY_PRECEDENCE:
            operand = f"({operand})"
        return node.op + operand
    if isinstance(node, XprCall):
        return f"{node.name}({', '.join(xpr_render_minimal(arg) for arg in node.args)})"
    left = _render_child(node.left, node.op, False)
    right = _render_child(node.right, node.op, True)
    joiner = node.op if node.op == "^" else f" {node.op} "
    return f"{left}{joiner}{right}"


def xpr_render_parenthesized(node):
    """Source text with every operator application wrapped in parentheses."""
    if isinstance(node, XprUnary):
        return f"({node.op}{xpr_render_parenthesized(node.operand)})"
    if isinstance(node, XprBinary):
        left, right = xpr_render_parenthesized(node.left), xpr_render_parenthesized(node.right)
        return f"({left} {node.op} {right})"
    if isinstance(node, XprCall):
        return f"{node.name}({', '.join(xpr_render_parenthesized(arg) for arg in node.args)})"
    return xpr_render_minimal(node)


def _node_label(node):
    if isinstance(node, XprNum):
        return f"Num {xpr_number_to_text(node.value)}"
    if isinstance(node, XprVar):
        return f"Var {node.name}"
    if isinstance(node, XprUnary):
        return f"Unary {node.op}"
    if isinstance(node, XprBinary):
        return f"Binary {node.op}"
    return f"Call {node.name}"


def xpr_render_tree(node, indent=0):
    """Indented outline of the tree, two spaces per level, one node per line."""
    lines = ["  " * indent + _node_label(node)]
    lines.extend(xpr_render_tree(child, indent + 1) for child in xpr_child_nodes(node))
    return "\n".join(lines)
