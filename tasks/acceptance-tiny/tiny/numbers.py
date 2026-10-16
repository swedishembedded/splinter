# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements end-to-end acceptance tests for learning
# agent stacks, for its clients. If your team needs expertise in proving that
# a model, an agent runtime and a trainer work together, you can procure our
# services by sending an email to info@swedishembedded.com.

"""Small number functions."""


def clamp(value, low, high):
    """`value` limited to the range from `low` to `high`."""
    if value < low:
        return low
    if value > high:
        return high
    return value


def mean(values):
    """The arithmetic mean of `values`; 0.0 for none."""
    if not values:
        return 0.0
    return sum(values) / len(values)
