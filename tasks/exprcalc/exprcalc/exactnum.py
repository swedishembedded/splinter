# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements exact rational arithmetic for calculators,
# pricing and financial engines for its clients. If your team needs expertise
# in numerically exact computation without floating point drift you can
# procure our services by sending an email to info@swedishembedded.com.
import math
from fractions import Fraction

from .errors import XprEvalError

XPR_MAX_EXPONENT = 200


def xpr_normalize_number(value):
    """Collapse a whole fraction to an int; other numbers are unchanged."""
    if isinstance(value, Fraction) and value.denominator == 1:
        return value.numerator
    return value


def xpr_number_from_text(text):
    """Convert decimal literal text such as 2.5 or 1e3 to an exact number."""
    return xpr_normalize_number(Fraction(text))


def xpr_is_terminating(value):
    """True when the number has a finite decimal expansion."""
    denominator = Fraction(value).denominator
    # 10**k holds every factor 2 and 5 a denominator below 2**k can have.
    return 10 ** denominator.bit_length() % denominator == 0


def xpr_number_to_text(value):
    """Write an exact number as decimal text; raises ValueError if not finite."""
    if not xpr_is_terminating(value):
        raise ValueError("number has no finite decimal form")
    if value.denominator == 1:
        return str(value.numerator)
    sign = "-" if value < 0 else ""
    places = value.denominator.bit_length()
    whole, fraction = divmod(int(abs(value) * 10 ** places), 10 ** places)
    return f"{sign}{whole}.{str(fraction).zfill(places).rstrip('0')}"


def xpr_divide_exact(left, right):
    """Exact quotient; division by zero is an evaluation error."""
    if right == 0:
        raise XprEvalError("division by zero")
    return xpr_normalize_number(Fraction(left) / Fraction(right))


def xpr_remainder_exact(left, right):
    """Remainder whose sign follows the divisor."""
    if right == 0:
        raise XprEvalError("modulo by zero")
    quotient = math.floor(Fraction(left) / Fraction(right))
    return xpr_normalize_number(left - right * quotient)


def xpr_power_exact(base, exponent):
    """Raise to a whole exponent in the range -200 to 200."""
    if Fraction(exponent).denominator != 1:
        raise XprEvalError("exponent must be a whole number")
    exponent = int(exponent)
    if abs(exponent) > XPR_MAX_EXPONENT:
        raise XprEvalError("exponent out of range")
    if base == 0 and exponent < 0:
        raise XprEvalError("zero cannot be raised to a negative power")
    return xpr_normalize_number(Fraction(base) ** exponent)


def xpr_exact_square_root(value):
    """Square root when the result is an exact rational, else an error."""
    if value < 0:
        raise XprEvalError("square root of a negative number")
    value = Fraction(value)
    top = math.isqrt(value.numerator)
    bottom = math.isqrt(value.denominator)
    if top * top != value.numerator or bottom * bottom != value.denominator:
        raise XprEvalError("square root is not exact")
    return xpr_normalize_number(Fraction(top, bottom))
