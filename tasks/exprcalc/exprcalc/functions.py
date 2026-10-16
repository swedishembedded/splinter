# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements extensible function registries with checked
# arities for scripting and formula engines for its clients. If your team needs
# expertise in safe, embeddable calculator function libraries you can procure
# our services by sending an email to info@swedishembedded.com.
import math
from dataclasses import dataclass
from fractions import Fraction

from .errors import XprEvalError
from .exactnum import xpr_exact_square_root, xpr_normalize_number, xpr_power_exact


@dataclass(frozen=True)
class XprFunction:
    name: str
    minimum: int
    maximum: object
    implementation: object


def _builtin_round(value):
    return math.floor(Fraction(value) + Fraction(1, 2))


def _builtin_sign(value):
    return (value > 0) - (value < 0)


def _builtin_gcd(left, right):
    if Fraction(left).denominator != 1 or Fraction(right).denominator != 1:
        raise XprEvalError("gcd needs whole numbers")
    return math.gcd(int(left), int(right))


def _builtin_clamp(value, low, high):
    if low > high:
        raise XprEvalError("clamp bounds are reversed")
    return min(max(value, low), high)


def xpr_builtin_registry():
    """A new registry holding the standard functions, keyed by name."""
    table = (
        XprFunction("abs", 1, 1, abs),
        XprFunction("min", 1, None, lambda *values: min(values)),
        XprFunction("max", 1, None, lambda *values: max(values)),
        XprFunction("sqrt", 1, 1, xpr_exact_square_root),
        XprFunction("floor", 1, 1, math.floor),
        XprFunction("ceil", 1, 1, math.ceil),
        XprFunction("round", 1, 1, _builtin_round),
        XprFunction("sign", 1, 1, _builtin_sign),
        XprFunction("gcd", 2, 2, _builtin_gcd),
        XprFunction("clamp", 3, 3, _builtin_clamp),
        XprFunction("if", 3, 3, lambda cond, then, other: then if cond != 0 else other),
        XprFunction("pow", 2, 2, xpr_power_exact),
    )
    return {entry.name: entry for entry in table}


def xpr_register_function(registry, name, minimum, maximum, implementation):
    """Return a new registry with one more function; names are unique."""
    if name in registry:
        raise ValueError(f"function {name!r} is already registered")
    extended = dict(registry)
    extended[name] = XprFunction(name, minimum, maximum, implementation)
    return extended


def xpr_describe_arity(function):
    """Words for the argument count a function accepts."""
    if function.maximum is None:
        return f"at least {function.minimum}"
    if function.minimum == function.maximum:
        plural = "" if function.minimum == 1 else "s"
        return f"{function.minimum} argument{plural}"
    return f"{function.minimum} to {function.maximum} arguments"


def xpr_call_function(registry, name, args):
    """Call a registered function with checked arity; returns an exact number."""
    if name not in registry:
        raise XprEvalError(f"unknown function {name!r}")
    function = registry[name]
    too_many = function.maximum is not None and len(args) > function.maximum
    if len(args) < function.minimum or too_many:
        raise XprEvalError(
            f"function {name!r} expects {xpr_describe_arity(function)}, got {len(args)}"
        )
    return xpr_normalize_number(function.implementation(*args))
