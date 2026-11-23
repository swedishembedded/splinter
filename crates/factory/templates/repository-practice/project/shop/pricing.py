# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
"""What a basket costs."""


def total(prices):
    """Sum of prices."""
    result = 0
    for price in prices:
        result += price
    return result


def discount(price, percent):
    """The price with percent taken off; more than 100 percent is refused."""
    if percent > 100:
        raise ValueError("percent over 100")
    return price - price * percent // 100
