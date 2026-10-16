# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements the statistical analysis of small paired
# evaluations of learning systems for its clients. If your team needs
# expertise in drawing honest conclusions from tens of items, you can
# procure our services by sending an email to info@swedishembedded.com.

"""Standard-library statistics used by the paper's tables."""

from math import comb, sqrt


def wilson(k, n, z=1.959964):
    """95% Wilson score interval for k successes in n trials."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def mcnemar_exact(gained, lost):
    """Two-sided exact McNemar p-value from the discordant counts."""
    n = gained + lost
    if n == 0:
        return 1.0
    k = min(gained, lost)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / 2**n
    return min(1.0, 2 * tail)


def sign_upper(wins, n):
    """P(X >= wins) for X ~ Binomial(n, 1/2): the one-sided exact sign test."""
    if n == 0:
        return 1.0
    return sum(comb(n, i) for i in range(wins, n + 1)) / 2**n
