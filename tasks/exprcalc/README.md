<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# exprcalc

An exact arithmetic expression engine in pure Python (standard library only):
a tokenizer with source positions, a precedence-climbing parser producing an
immutable syntax tree, an evaluator with scoped variables and a function
registry, constant folding, a minimal-parentheses printer, static analysis,
dependency-ordered definitions and symbolic differentiation.

Numbers are exact: integers and `fractions.Fraction` values, never floats.
Results that are whole numbers are always returned as `int`.

## Running the tests

    python3 -m unittest

Run it in this directory. No installation and no `PYTHONPATH` are needed.

## Language

Precedence from loosest to tightest binding:

| level | operators                  | grouping                  |
|-------|----------------------------|---------------------------|
| 1     | `\|\|`                     | left                      |
| 2     | `&&`                       | left                      |
| 3     | `==` `!=`                  | left                      |
| 4     | `<` `<=` `>` `>=`          | left                      |
| 5     | `+` `-`                    | left                      |
| 6     | `*` `/` `%`                | left                      |
| 7     | unary `-` `+` `!`          | prefix                    |
| 8     | `^`                        | right (`2^3^2 = 2^9`)     |

Comparisons and logic yield `1` or `0`; any nonzero value is true, `&&` and
`||` short-circuit. `%` takes the sign of the divisor. `^` needs a whole
exponent between -200 and 200. `-2^2` is `-4`. Calls are `name(a, b)`;
numbers are `12`, `2.5`, `.5`, `1e3`, `2.5E-2`; names are ASCII letters,
digits and underscores. Parentheses may nest at most 50 levels deep.

Built-in functions: `abs(x)`, `min(a, ...)`, `max(a, ...)`, `sqrt(x)` (exact
results only), `floor(x)`, `ceil(x)`, `round(x)` (halves go up), `sign(x)`,
`gcd(a, b)` (whole numbers), `clamp(x, low, high)`, `if(cond, then, else)`
(both branches are evaluated), `pow(a, b)`.

## Public API

All names are importable from `exprcalc`. Syntax trees are the frozen
dataclasses `XprNum(value)`, `XprVar(name)`, `XprUnary(op, operand)`,
`XprBinary(op, left, right)` and `XprCall(name, args)`; each has a `position`
field (a character offset or `None`) that is ignored by equality.

### errors

- `XprError(message, position=None)`: base class; `str()` appends
  `" at position N"` when the position is known. `XprSyntaxError` and
  `XprEvalError` derive from it.
- `xpr_line_and_column(source, position)`: 1-based `(line, column)`;
  `ValueError("position out of range")` unless `0 <= position <= len(source)`.
- `xpr_source_line(source, position)`: the text of the line holding the
  position, same error.
- `xpr_attach_position(error, position)`: the error itself when it already has
  a position, otherwise a copy of the same type with the position set.
- `xpr_describe_error(error, source)`: multi-line report with the source line
  and a caret.

### exactnum

- `xpr_normalize_number(value)`: whole `Fraction` to `int`.
- `xpr_number_from_text(text)`: exact value of a literal such as `"2.5e1"`.
- `xpr_is_terminating(value)`: true if the decimal expansion is finite.
- `xpr_number_to_text(value)`: decimal text; `ValueError` if not finite.
- `xpr_divide_exact(a, b)`, `xpr_remainder_exact(a, b)`,
  `xpr_power_exact(base, exponent)`, `xpr_exact_square_root(value)`: raise
  `XprEvalError` for zero divisors, fractional or out-of-range exponents, zero
  to a negative power, negative or inexact roots.

### tokenizer

- `xpr_scan_tokens(source)`: list of `XprToken(kind, text, position)`, always
  ending with an `end` token. Kinds: `number`, `name`, `op`, `lparen`,
  `rparen`, `comma`, `end`. `XprSyntaxError` for unknown characters,
  `malformed number` and `malformed exponent`.
- `xpr_token_is(token, kind, text=None)`, `xpr_is_name_start(char)`,
  `xpr_is_name_char(char)`, `xpr_scan_name_end(source, start)`,
  `xpr_scan_number_end(source, start)`.

### astnodes

- `xpr_number_node(value, position=None)`: literal, or a negation of a literal
  for negative values.
- `xpr_is_number(node, value=None)`, `xpr_literal_value(node)` (a literal or
  negated literal, else `None`).
- `xpr_child_nodes(node)`, `xpr_rebuild_node(node, children)`,
  `xpr_walk_nodes(node)` (pre-order), `xpr_node_count(node)`,
  `xpr_tree_depth(node)`, `xpr_map_postorder(node, transform)`.

### operators

- `xpr_binary_precedence(op)`, `xpr_is_binary_operator(op)`,
  `xpr_is_unary_operator(op)`, `xpr_is_right_associative(op)`.
- `xpr_apply_unary(op, value)`, `xpr_apply_binary(op, left, right)`: exact
  semantics of every operator; unknown operators raise `XprEvalError`.

### parser

- `xpr_parse_text(source)` and `xpr_parse_tokens(tokens)`: return the tree or
  raise `XprSyntaxError` (`unexpected end of input`, `unexpected '<text>'`,
  `expected ')'`, `expression nested too deeply`) with the offending position.

### functions

- `xpr_builtin_registry()`: new dict of name to `XprFunction(name, minimum,
  maximum, implementation)`; `maximum` is `None` for unbounded.
- `xpr_register_function(registry, name, minimum, maximum, implementation)`:
  returns an extended copy; `ValueError` if the name is taken.
- `xpr_call_function(registry, name, args)`: checks name and arity
  (`XprEvalError`), returns a normalized number. `xpr_describe_arity(function)`.

### evaluator

- `XprScope(variables=None, parent=None)` with `lookup(name)`, `define(name,
  value)`, `child(variables=None)` and `names()`.
- `xpr_evaluate_node(node, scope, registry)`: evaluation errors carry the
  position of the failing node.
- `xpr_evaluate_text(source, variables=None, registry=None)`.
- `xpr_evaluate_definitions(sources, variables=None, registry=None)`: evaluates
  a dict of name to expression text in dependency order and returns a dict of
  results in evaluation order; cycles raise `XprEvalError`.

### folding

- `xpr_fold_constants(node, registry=None)`: replaces constant subtrees by
  their value and removes `x+0`, `0+x`, `x-0`, `x*1`, `1*x`, `x/1`, `x^1` and
  `--x`. Errors and values without a finite decimal form are left unfolded.
  Calls fold only if a registry is given.

### printer

- `xpr_render_minimal(node)`: source text with the fewest parentheses that
  parse back to the same tree.
- `xpr_render_parenthesized(node)`: every operator application bracketed.
- `xpr_render_tree(node, indent=0)`: indented outline, one node per line.

### analysis

- `xpr_free_variables(node)`, `xpr_called_functions(node)` (first-appearance
  order), `xpr_is_constant_expression(node)`, `xpr_operator_census(node)`,
  `xpr_substitute_variable(node, name, replacement)`.
- `xpr_dependency_order(definitions)`: order of a dict of name to tree so every
  definition follows the ones it uses; `XprEvalError("circular definition: a
  -> b -> a")` on cycles.

### derivative

- `xpr_differentiate(node, variable)`: simplified derivative. Supports
  `+ - * /`, `^` with an exponent free of the variable, unary `-` `+`, and the
  calls `sqrt`, `abs` (gives `sign`) and `pow`; anything else raises
  `XprEvalError`.
- `xpr_nth_derivative(node, variable, order)`, `xpr_gradient(node, variables)`.
- Smart constructors `xpr_make_negation`, `xpr_make_sum`,
  `xpr_make_difference`, `xpr_make_product`, `xpr_make_quotient`,
  `xpr_make_power`.

## Example

    from exprcalc import (xpr_parse_text, xpr_evaluate_text, xpr_differentiate,
                          xpr_fold_constants, xpr_render_minimal)

    xpr_evaluate_text("2 * x ^ 2 + 1", {"x": 3})            # 19
    tree = xpr_parse_text("3 * x ^ 2 + 2 * x + (1 + 1)")
    xpr_render_minimal(xpr_fold_constants(tree))            # "3 * x^2 + 2 * x + 2"
    xpr_render_minimal(xpr_differentiate(tree, "x"))        # "6 * x + 2"

Swedish Embedded AB implements exact, auditable formula engines for pricing,
configuration and rule evaluation for its clients. If your team needs
expertise in expression parsing, evaluation and symbolic simplification you
can procure our services by sending an email to info@swedishembedded.com.
