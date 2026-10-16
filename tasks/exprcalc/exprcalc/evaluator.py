# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements safe formula evaluation with scoped variables
# and dependency-ordered definitions for its clients. If your team needs
# expertise in embedding calculators and spreadsheet-like rule evaluation you
# can procure our services by sending an email to info@swedishembedded.com.
from .analysis import xpr_dependency_order
from .astnodes import XprBinary, XprCall, XprNum, XprUnary, XprVar
from .errors import XprEvalError, xpr_attach_position
from .functions import xpr_builtin_registry, xpr_call_function
from .operators import xpr_apply_binary, xpr_apply_unary
from .parser import xpr_parse_text


class XprScope:
    """Variable bindings with an optional enclosing scope."""

    def __init__(self, variables=None, parent=None):
        self.variables = dict(variables or {})
        self.parent = parent

    def lookup(self, name):
        if name in self.variables:
            return self.variables[name]
        if self.parent is None:
            raise XprEvalError(f"undefined variable {name!r}")
        return self.parent.lookup(name)

    def define(self, name, value):
        self.variables[name] = value

    def child(self, variables=None):
        return XprScope(variables, self)

    def names(self):
        inherited = set() if self.parent is None else set(self.parent.names())
        return sorted(inherited | set(self.variables))


def xpr_evaluate_node(node, scope, registry):
    """Evaluate a tree; errors carry the position of the offending node."""
    if isinstance(node, XprNum):
        return node.value
    try:
        if isinstance(node, XprVar):
            return scope.lookup(node.name)
        if isinstance(node, XprUnary):
            return xpr_apply_unary(node.op, xpr_evaluate_node(node.operand, scope, registry))
        if isinstance(node, XprBinary):
            return _evaluate_binary(node, scope, registry)
        args = [xpr_evaluate_node(arg, scope, registry) for arg in node.args]
        return xpr_call_function(registry, node.name, args)
    except XprEvalError as error:
        raise xpr_attach_position(error, node.position) from None


def _evaluate_binary(node, scope, registry):
    left = xpr_evaluate_node(node.left, scope, registry)
    if node.op == "&&" and left == 0:
        return 0
    if node.op == "||" and left != 0:
        return 1
    return xpr_apply_binary(node.op, left, xpr_evaluate_node(node.right, scope, registry))


def xpr_evaluate_text(source, variables=None, registry=None):
    """Parse and evaluate expression text with a variable mapping."""
    registry = xpr_builtin_registry() if registry is None else registry
    return xpr_evaluate_node(xpr_parse_text(source), XprScope(variables), registry)


def xpr_evaluate_definitions(sources, variables=None, registry=None):
    """Evaluate named expressions in dependency order; later ones may use earlier."""
    registry = xpr_builtin_registry() if registry is None else registry
    trees = {name: xpr_parse_text(text) for name, text in sources.items()}
    scope = XprScope(variables)
    results = {}
    for name in xpr_dependency_order(trees):
        results[name] = xpr_evaluate_node(trees[name], scope, registry)
        scope.define(name, results[name])
    return results
