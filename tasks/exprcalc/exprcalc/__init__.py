# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
from .analysis import (
    xpr_called_functions,
    xpr_dependency_order,
    xpr_free_variables,
    xpr_is_constant_expression,
    xpr_operator_census,
    xpr_substitute_variable,
)
from .astnodes import (
    XprBinary,
    XprCall,
    XprNum,
    XprUnary,
    XprVar,
    xpr_child_nodes,
    xpr_is_number,
    xpr_literal_value,
    xpr_map_postorder,
    xpr_node_count,
    xpr_number_node,
    xpr_rebuild_node,
    xpr_tree_depth,
    xpr_walk_nodes,
)
from .derivative import (
    xpr_differentiate,
    xpr_gradient,
    xpr_make_difference,
    xpr_make_negation,
    xpr_make_power,
    xpr_make_product,
    xpr_make_quotient,
    xpr_make_sum,
    xpr_nth_derivative,
)
from .errors import (
    XprError,
    XprEvalError,
    XprSyntaxError,
    xpr_attach_position,
    xpr_describe_error,
    xpr_line_and_column,
    xpr_source_line,
)
from .evaluator import (
    XprScope,
    xpr_evaluate_definitions,
    xpr_evaluate_node,
    xpr_evaluate_text,
)
from .exactnum import (
    xpr_divide_exact,
    xpr_exact_square_root,
    xpr_is_terminating,
    xpr_normalize_number,
    xpr_number_from_text,
    xpr_number_to_text,
    xpr_power_exact,
    xpr_remainder_exact,
)
from .folding import xpr_fold_constants
from .functions import (
    XprFunction,
    xpr_builtin_registry,
    xpr_call_function,
    xpr_describe_arity,
    xpr_register_function,
)
from .operators import (
    xpr_apply_binary,
    xpr_apply_unary,
    xpr_binary_precedence,
    xpr_is_binary_operator,
    xpr_is_right_associative,
    xpr_is_unary_operator,
)
from .parser import XprParser, xpr_parse_text, xpr_parse_tokens
from .printer import xpr_render_minimal, xpr_render_parenthesized, xpr_render_tree
from .tokenizer import XprToken, xpr_check_number_text, xpr_scan_tokens, xpr_token_is
