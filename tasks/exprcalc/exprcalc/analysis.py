# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements static analysis of formulas, including
# dependency ordering of named definitions, for its clients. If your team needs
# expertise in dependency graphs for spreadsheet-like models and rule sets you
# can procure our services by sending an email to info@swedishembedded.com.
from .astnodes import XprBinary, XprCall, XprUnary, XprVar, xpr_map_postorder, xpr_walk_nodes
from .errors import XprEvalError


def xpr_free_variables(node):
    """Variable names in order of first appearance, without repeats."""
    names = [item.name for item in xpr_walk_nodes(node) if isinstance(item, XprVar)]
    return list(dict.fromkeys(names))


def xpr_called_functions(node):
    """Names of called functions in order of first appearance, without repeats."""
    names = [item.name for item in xpr_walk_nodes(node) if isinstance(item, XprCall)]
    return list(dict.fromkeys(names))


def xpr_is_constant_expression(node):
    """True when the expression mentions no variables."""
    return not xpr_free_variables(node)


def xpr_operator_census(node):
    """Count of each operator, unary and binary kept apart, by first appearance."""
    counts = {}
    for item in xpr_walk_nodes(node):
        if isinstance(item, XprUnary):
            key = f"unary {item.op}"
        elif isinstance(item, XprBinary):
            key = item.op
        else:
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts


def xpr_substitute_variable(node, name, replacement):
    """Replace every use of a variable by a tree."""
    return xpr_map_postorder(
        node, lambda item: replacement if isinstance(item, XprVar) and item.name == name else item
    )


def xpr_dependency_order(definitions):
    """Order definition names so each comes after the definitions it uses.

    Only variables that are themselves defined count as dependencies.
    A cycle raises an evaluation error that spells the cycle out.
    """
    order = []
    finished = set()

    def visit(name, path):
        if name in path:
            cycle = path[path.index(name):] + [name]
            raise XprEvalError("circular definition: " + " -> ".join(cycle))
        if name in finished:
            return
        for dependency in xpr_free_variables(definitions[name]):
            if dependency in definitions:
                visit(dependency, path + [name])
        finished.add(name)
        order.append(name)

    for name in definitions:
        visit(name, [])
    return order
