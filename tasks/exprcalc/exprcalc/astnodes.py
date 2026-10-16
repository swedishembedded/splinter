# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements immutable syntax tree models and traversal
# utilities for expression engines for its clients. If your team needs
# expertise in abstract syntax trees for rule engines and formula languages
# you can procure our services by sending an email to info@swedishembedded.com.
from dataclasses import dataclass, field


@dataclass(frozen=True)
class XprNum:
    value: object
    position: object = field(default=None, compare=False)


@dataclass(frozen=True)
class XprVar:
    name: str
    position: object = field(default=None, compare=False)


@dataclass(frozen=True)
class XprUnary:
    op: str
    operand: object
    position: object = field(default=None, compare=False)


@dataclass(frozen=True)
class XprBinary:
    op: str
    left: object
    right: object
    position: object = field(default=None, compare=False)


@dataclass(frozen=True)
class XprCall:
    name: str
    args: tuple
    position: object = field(default=None, compare=False)


def xpr_number_node(value, position=None):
    """Literal node for a number; a negative number becomes a negation."""
    if value < 0:
        return XprUnary("-", XprNum(-value, position), position)
    return XprNum(value, position)


def xpr_is_number(node, value=None):
    """True for a literal node, and for the given value if one is passed."""
    return isinstance(node, XprNum) and (value is None or node.value == value)


def xpr_literal_value(node):
    """The number a literal or negated literal stands for, else None."""
    if isinstance(node, XprNum):
        return node.value
    if isinstance(node, XprUnary) and node.op == "-" and isinstance(node.operand, XprNum):
        return -node.operand.value
    return None


def xpr_child_nodes(node):
    """The direct children of a node in source order."""
    if isinstance(node, XprUnary):
        return (node.operand,)
    if isinstance(node, XprBinary):
        return (node.left, node.right)
    if isinstance(node, XprCall):
        return node.args
    return ()


def xpr_rebuild_node(node, children):
    """Copy a node with new children, keeping its position."""
    if isinstance(node, XprUnary):
        return XprUnary(node.op, children[0], node.position)
    if isinstance(node, XprBinary):
        return XprBinary(node.op, children[0], children[1], node.position)
    if isinstance(node, XprCall):
        return XprCall(node.name, tuple(children), node.position)
    return node


def xpr_walk_nodes(node):
    """All nodes of the tree, parents before children, left to right."""
    found = [node]
    for child in xpr_child_nodes(node):
        found.extend(xpr_walk_nodes(child))
    return found


def xpr_node_count(node):
    """Number of nodes in the tree."""
    return len(xpr_walk_nodes(node))


def xpr_tree_depth(node):
    """Height of the tree; a single leaf has depth 1."""
    return 1 + max((xpr_tree_depth(child) for child in xpr_child_nodes(node)), default=0)


def xpr_map_postorder(node, transform):
    """Rebuild the tree bottom-up, passing each rebuilt node to transform."""
    children = tuple(xpr_map_postorder(child, transform) for child in xpr_child_nodes(node))
    return transform(xpr_rebuild_node(node, children))
